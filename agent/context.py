"""Context（上下文）管理。

每个 session 拥有独立的消息历史，格式严格遵循 OpenAI Chat 协议，
组装后可直接发给 DeepSeek API：

    {"role": "system",    "content": str}                       # 系统提示词
    {"role": "user",      "content": str}                       # 用户输入
    {"role": "assistant", "content": str|None, "tool_calls": [...]}  # 思考/工具调用请求
    {"role": "tool",      "tool_call_id": str, "content": str}  # 工具执行结果

什么信息进入 context（设计决策）：
- 用户输入、assistant 的工具调用请求、工具执行结果、最终答案 —— 全部保留，
  这是支持"追问"（无论纯对话还是带工具）的最低限度信息；
- Agent 思考过程：仅在伴随工具调用时随 assistant 消息保留（帮助后续轮次
  理解"当时为什么调用工具"）；纯文本回答的思考不单独入库，避免膨胀；
- 工具结果超长时截断，防止单条消息撑爆上下文。

压缩（基础版）：
- 消息数超过 max_messages 时触发；
- 按"用户轮次"整轮切割，保证 assistant(tool_calls) 与 tool 消息对永远完整
  （拆散会导致 API 报错）；
- 被压缩的旧消息拼接为文字摘要，常驻 context 头部（system 区），
  之后每次 LLM 请求自动携带 —— 即 memory 的"召回"。
"""

import copy
import json
import uuid

from .utils import to_display_text, truncate

DEFAULT_SYSTEM_PROMPT = """你是一个乐于助人的 AI 助手，可以调用工具来帮助用户完成任务。

工作准则：
- 需要精确计算、检索信息、查询天气、管理待办事项时，优先调用相应工具，不要凭记忆猜测。
- 每次工具调用后，仔细阅读工具返回结果，再决定：继续调用其他工具，还是直接给出最终回答。
- 回答使用中文，简洁准确。
- 如上下文中存在"早期对话摘要"，请结合摘要理解早期的用户意图与结论。"""


class ContextManager:
    """管理所有 session 的上下文：消息历史、轮次计数、超限压缩。"""

    def __init__(self, system_prompt=DEFAULT_SYSTEM_PROMPT, max_messages=40,
                 keep_recent=16, max_tool_result_chars=1200,
                 summary_entry_chars=160, summary_total_chars=1500):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.max_tool_result_chars = max_tool_result_chars
        self.summary_entry_chars = summary_entry_chars
        self.summary_total_chars = summary_total_chars
        self._contexts = {}

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _get(self, session_id):
        if session_id not in self._contexts:
            self._contexts[session_id] = {
                "messages": [],      # OpenAI 格式消息列表
                "turn_count": 0,     # 用户轮次（收到一次用户输入 = 1 轮）
                "summary": "",       # 压缩产生的早期对话摘要
                "compress_count": 0, # 已触发压缩的次数
            }
        return self._contexts[session_id]

    def _append(self, session_id, message):
        ctx = self._get(session_id)
        ctx["messages"].append(message)
        if len(ctx["messages"]) > self.max_messages:
            self._compress(session_id)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_user_message(self, session_id, content):
        """记录一条用户输入，用户轮次 +1。"""
        ctx = self._get(session_id)
        ctx["turn_count"] += 1
        self._append(session_id, {"role": "user", "content": str(content)})

    def add_assistant_message(self, session_id, thought="", answer="", tool_calls=None):
        """记录 assistant 消息。

        - tool_calls 非空：content 存思考过程（可为 None），并附带 OpenAI 格式的
          tool_calls 数组；
        - tool_calls 为空：content 存最终答案。
        """
        if tool_calls:
            message = {
                "role": "assistant",
                "content": thought or None,
                "tool_calls": [
                    {
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        else:
            message = {"role": "assistant", "content": answer or thought or ""}
        self._append(session_id, message)

    def add_tool_result(self, session_id, tool_call_id, result):
        """记录工具执行结果（role=tool，必须与 tool_call_id 配对），超长自动截断。"""
        text = truncate(to_display_text(result), self.max_tool_result_chars)
        self._append(session_id, {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text,
        })

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get_llm_messages(self, session_id):
        """组装发给 LLM 的完整消息列表：系统提示词 + 摘要 + 近期历史。"""
        ctx = self._get(session_id)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # 加入早期对话摘要（如果有）
        if ctx["summary"]:
            messages.append({
                "role": "system",
                "content": f"[早期对话摘要（自动压缩生成）]\n{ctx['summary']}",
            })
        messages.extend(copy.deepcopy(ctx["messages"]))
        return messages

    def get_history(self, session_id):
        """返回会话历史的可读视图（调试/展示用）。"""
        ctx = self._get(session_id)
        return {
            "turn_count": ctx["turn_count"],
            "summary": ctx["summary"],
            "compress_count": ctx["compress_count"],
            "messages": copy.deepcopy(ctx["messages"]),
        }

    def get_turn_count(self, session_id):
        return self._get(session_id)["turn_count"]

    # ------------------------------------------------------------------
    # 压缩与清理
    # ------------------------------------------------------------------

    def _compress(self, session_id):
        """基础压缩：按用户轮次整轮保留近期消息，旧消息折叠为文字摘要。"""
        ctx = self._get(session_id)
        messages = ctx["messages"]
        if len(messages) <= self.keep_recent:
            return

        # 按"用户轮次"分组：每条 user 消息开启一个新轮次，
        # 其后的 assistant/tool 消息归属同轮（保证消息对完整）。
        turns = []
        for message in messages:
            if message["role"] == "user" or not turns:
                turns.append([message])
            else:
                turns[-1].append(message)

        # 从最近的轮次向前收集，直到接近 keep_recent 上限
        kept = []
        for turn in reversed(turns):
            if kept and len(kept) + len(turn) > self.keep_recent:
                break
            kept = turn + kept

        old = messages[: len(messages) - len(kept)]
        if not old:
            return

        entries = []
        for message in old:
            role = message["role"]
            content = message.get("content") or ""
            if role == "assistant" and message.get("tool_calls"):
                names = ",".join(
                    tc["function"]["name"] for tc in message["tool_calls"]
                )
                content = f"调用工具[{names}] {content}".strip()
            elif role == "tool":
                content = f"工具结果: {content}"
            entries.append(f"{role}: {truncate(content, self.summary_entry_chars)}")

        new_summary = "; ".join(entries)
        summary = f"{ctx['summary']} | {new_summary}" if ctx["summary"] else new_summary
        ctx["summary"] = summary[-self.summary_total_chars:]  # 超长保尾部（最近的）
        ctx["messages"] = kept
        ctx["compress_count"] += 1

    def clear(self, session_id):
        self._contexts.pop(session_id, None)
