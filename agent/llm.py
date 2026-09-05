"""LLM 接入层。

- DeepSeekLLM：调用 DeepSeek（OpenAI 兼容协议）真实 API。
  优先使用原生 function calling（模型基于工具 Schema 自主决策是否调用工具），
  并带指数退避重试。全部配置统一从 agent/config.py 读取
  （优先级：环境变量 > .env 文件 > 默认值）。
- parse_text_response：LLM 纯文本输出的解析逻辑（无 function calling 时的兼容通道），
  从文本中提取思考过程（thought）、工具调用（tool_calls）或最终答案（answer），
  支持三种协议：整体 JSON、XML 风格标签、纯文本兜底。
- FakeLLM：测试用的可脚本化假 LLM，按队列依次返回预设响应并记录收到的请求，
  用于在离线环境下确定性测试 Agent 循环。
"""

import copy
import json
import re
import time
import uuid

from . import config
from .exceptions import LLMError, ParseError

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_INVOKE_RE = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# 文本协议解析
# ---------------------------------------------------------------------------

def _parse_invoke_args(args_text):
    """解析 <invoke> 标签内的参数文本，容忍 <params> 包裹与 markdown 代码块围栏。"""
    args_text = (args_text or "").strip()
    m = re.fullmatch(r"<params>(.*)</params>", args_text, re.DOTALL)
    if m:
        args_text = m.group(1).strip()
    args_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", args_text).strip()
    if not args_text:
        return {}
    try:
        parsed = json.loads(args_text)
    except json.JSONDecodeError:
        return {"query": args_text}
    return parsed if isinstance(parsed, dict) else {"input": parsed}


def _normalize_tool_calls(raw_calls):
    """把各种形态的工具调用描述统一为 {"id", "name", "args"} 结构。"""
    normalized = []
    for call in raw_calls or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("tool")
        if not name:
            continue
        args = call.get("args", call.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": args}
        normalized.append({
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "name": str(name),
            "args": args,
        })
    return normalized


def parse_text_response(text):
    """解析 LLM 的纯文本输出，提取 thought / tool_calls / answer。

    支持三种协议（按顺序尝试）：
    1. 整体为 JSON：{"thought": ..., "tool_calls": [...], "answer": ...}
    2. XML 风格标签：<thought>..</thought> <invoke name="x">{...}</invoke> <answer>..</answer>
    3. 兜底：整段文本视为最终答案
    """
    text = (text or "").strip()
    if not text:
        return {"thought": "", "tool_calls": [], "answer": ""}

    # 协议一：整体 JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and any(k in data for k in ("thought", "tool_calls", "answer")):
        return {
            "thought": str(data.get("thought", "")),
            "tool_calls": _normalize_tool_calls(data.get("tool_calls")),
            "answer": str(data.get("answer", "")),
        }

    # 协议二：XML 风格标签
    thought_match = _THOUGHT_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    invokes = _INVOKE_RE.findall(text)
    if thought_match or answer_match or invokes:
        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "name": name.strip(),
                "args": _parse_invoke_args(body),
            }
            for name, body in invokes
        ]
        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "tool_calls": tool_calls,
            "answer": answer_match.group(1).strip() if answer_match else "",
        }

    # 协议三：兜底
    return {"thought": "", "tool_calls": [], "answer": text}


# ---------------------------------------------------------------------------
# DeepSeek 真实 API
# ---------------------------------------------------------------------------

class DeepSeekLLM:
    """DeepSeek 真实 API 客户端（OpenAI 兼容协议）。

    配置来源（优先级：显式参数 > config.py（环境变量 / .env）> 默认值）：
    - api_key:     DEEPSEEK_API_KEY（必填）
    - model:       DEEPSEEK_MODEL，默认 deepseek-chat
    - base_url:    DEEPSEEK_BASE_URL，默认 https://api.deepseek.com
    - timeout:     DEEPSEEK_TIMEOUT，默认 60 秒
    - max_retries: DEEPSEEK_MAX_RETRIES，默认 2
    """

    def __init__(self, api_key=None, model=None, base_url=None, timeout=None, max_retries=None):
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.model = model or config.DEEPSEEK_MODEL
        self.base_url = base_url or config.DEEPSEEK_BASE_URL
        self.timeout = config.DEEPSEEK_TIMEOUT if timeout is None else timeout
        self.max_retries = config.DEEPSEEK_MAX_RETRIES if max_retries is None else max_retries
        if not self.api_key:
            raise LLMError(
                "缺少 DeepSeek API Key：请在环境变量或 .env 文件中设置 DEEPSEEK_API_KEY"
                "（可参考 .env.example），或在构造 DeepSeekLLM(api_key=...) 时传入。"
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError("未安装 openai 包，请先执行: pip install openai") from e
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def chat(self, messages, tools=None, session_id=None):
        """调用 LLM。返回统一结构 {"thought", "tool_calls", "answer"}。

        - tools: OpenAI function-calling 格式的工具 Schema 列表；
          LLM 基于该 Schema 自主决策是否调用工具。
        - 网络类异常按指数退避重试，超过次数抛出 LLMError。
        """
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._chat_once(messages, tools)
            except (LLMError, ParseError):
                raise
            except Exception as e:  # 网络/限流等服务端异常，值得重试
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
        raise LLMError(
            f"DeepSeek API 调用失败（已重试 {self.max_retries} 次）: {last_error}"
        )

    def _chat_once(self, messages, tools):
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        return self._parse_api_response(response)

    def _parse_api_response(self, response):
        """解析 API 响应。

        - 走了原生 function calling：tool_calls 中提取 id/name/args，
          content 中的文字作为思考过程（thought）。
        - 未走 function calling：content 交给 parse_text_response 做文本协议解析。
        """
        message = response.choices[0].message
        tool_calls = []
        for tc in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                raise ParseError(f"工具 {tc.function.name} 的参数 JSON 解析失败: {e}")
            if not isinstance(args, dict):
                args = {"input": args}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})

        if tool_calls:
            return {"thought": message.content or "", "tool_calls": tool_calls, "answer": ""}
        return parse_text_response(message.content or "")


# ---------------------------------------------------------------------------
# 测试用 FakeLLM
# ---------------------------------------------------------------------------

class FakeLLM:
    """可脚本化的假 LLM：按队列依次返回预设响应，并完整记录收到的每次请求。

    用于离线、确定性地测试 Agent 循环（工具调用、多轮迭代、追问等）。
    """

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.received = []

    def queue(self, response):
        self.responses.append(response)
        return self

    def chat(self, messages, tools=None, session_id=None):
        self.received.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "session_id": session_id,
        })
        if self.responses:
            return self.responses.pop(0)
        return {"thought": "", "tool_calls": [], "answer": "（FakeLLM 没有更多预设响应）"}


# ---------------------------------------------------------------------------
# 响应构造辅助（测试/示例用）
# ---------------------------------------------------------------------------

def text_response(answer, thought=""):
    """构造一个"直接回答"的 LLM 响应。"""
    return {"thought": thought, "tool_calls": [], "answer": answer}


def tool_call_response(name, args, thought="", call_id=None):
    """构造一个"调用工具"的 LLM 响应。"""
    return {
        "thought": thought,
        "tool_calls": [{
            "id": call_id or f"call_{uuid.uuid4().hex[:12]}",
            "name": name,
            "args": args,
        }],
        "answer": "",
    }
