"""Session 与核心 Agent Loop。

Agent 基本循环（每个用户输入触发一次）：

    Step 1  接收用户输入，写入 session 独立 context
    Step 2  组装 context（系统提示词 + 摘要 + 历史）调用 LLM，
            由 LLM 基于工具 Schema 自主决策：直接回复 or 调用工具
    Step 3  执行被请求的工具（结果与错误都写回 context）
    Step 4  根据工具结果判断：带着结果继续 loop（回到 Step 2），
            还是 LLM 不再请求工具时返回最终答案给用户

安全阀：
- max_iterations 限制单次请求内循环的最大迭代次数；
- 达到上限后追加一次性提示，强制 LLM 在"不调用工具"的约束下收尾作答。
"""

import uuid

from .context import ContextManager
from .exceptions import AgentError, SessionError
from .llm import DeepSeekLLM
from .logger import TraceLogger
from .tools import build_default_registry


class Session:
    """一个独立的会话窗口：绑定独立的 context、共享的 LLM 与工具注册表。"""

    def __init__(self, session_id, context_manager, llm, tool_registry, logger,
                 max_iterations=8):
        self.session_id = session_id
        self.context_manager = context_manager
        self.llm = llm
        self.tool_registry = tool_registry
        self.logger = logger
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # 核心循环
    # ------------------------------------------------------------------

    def run(self, user_input):
        """执行一次完整的 Agent 循环，返回本次执行的明细。"""
        # Step 1: 接收用户输入
        text = user_input if isinstance(user_input, str) else str(user_input)
        if not text.strip():
            return {
                "answer": "收到空消息，请输入具体问题。",
                "thought": "",
                "tool_calls": [],
                "iterations": 0,
                "status": "success",
            }

        self.logger.user_input(text, self.session_id)
        self.context_manager.add_user_message(self.session_id, text)

        executed_tool_calls = []
        iterations = 0
        answer = None
        last_thought = ""

        try:
            for step in range(1, self.max_iterations + 1):
                iterations = step

                # Step 2: LLM 决策 —— 直接回复，还是调用工具
                messages = self.context_manager.get_llm_messages(self.session_id)
                tools = self.tool_registry.get_function_schemas()
                self.logger.llm_request(len(messages), self.session_id, tools_count=len(tools))
                response = self.llm.chat(messages, tools=tools, session_id=self.session_id)
                self.logger.llm_response(response, self.session_id)

                tool_calls = response.get("tool_calls") or []
                if tool_calls:
                    # Step 3: 调用工具（含参数校验与异常兜底）
                    self.context_manager.add_assistant_message(
                        self.session_id,
                        thought=response.get("thought", ""),
                        tool_calls=tool_calls,
                    )
                    for tc in tool_calls:
                        executed_tool_calls.append(self._execute_tool(tc))
                    # Step 4: 工具结果已写入 context，继续下一轮迭代
                    continue

                # LLM 未请求工具 —— 这就是最终答案
                last_thought = response.get("thought", "")
                answer = response.get("answer") or last_thought or ""
                self.context_manager.add_assistant_message(self.session_id, answer=answer)
                break

            # 循环耗尽仍未得到答案：强制收尾
            if answer is None:
                answer = self._force_final_answer()

            self.logger.final_answer(answer, self.session_id)
            return {
                "answer": answer,
                "thought": last_thought,
                "tool_calls": executed_tool_calls,
                "iterations": iterations,
                "status": "success",
            }
        except Exception as e:
            self.logger.error(f"{type(e).__name__}: {e}", self.session_id)
            raise

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_call):
        """执行单个工具调用。

        工具执行失败不让循环崩溃：错误信息以工具结果的形式写回 context，
        让 LLM 在下一轮迭代中看到错误并自行决定如何处理（换参数重试 / 直接回答）。
        """
        name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"

        self.logger.tool_call(name, args, self.session_id)
        try:
            result = self.tool_registry.execute(name, args, session_id=self.session_id)
            status = "ok"
        except AgentError as e:
            result = {"error": str(e)}
            status = "error"
        except Exception as e:  # 工具内部的意外异常，双保险
            result = {"error": f"工具内部意外异常 {type(e).__name__}: {e}"}
            status = "error"
        self.logger.tool_result(name, result, self.session_id, status=status)

        # 无论成败都写回 context（保证与 assistant.tool_calls 配对完整）
        self.context_manager.add_tool_result(self.session_id, call_id, result)
        return {
            "id": call_id,
            "name": name,
            "args": args,
            "result": result,
            "status": status,
        }

    # ------------------------------------------------------------------
    # 超限收尾
    # ------------------------------------------------------------------

    def _force_final_answer(self):
        """达到最大迭代次数后，禁止工具调用，强制 LLM 基于已有信息作答。"""
        self.logger.info(
            f"达到最大迭代次数 {self.max_iterations}，强制生成最终回答", self.session_id
        )
        messages = self.context_manager.get_llm_messages(self.session_id)
        # 一次性提示（不写入 context）：要求不再调用工具、直接作答
        messages.append({
            "role": "user",
            "content": "（系统提示：已达到本轮最大工具调用次数，请立即基于以上已有信息"
                       "给出最终回答，不要再调用任何工具。）",
        })
        response = self.llm.chat(messages, tools=None, session_id=self.session_id)
        answer = (
            response.get("answer")
            or response.get("thought")
            or "（已达最大工具调用轮数，未能生成最终回答。）"
        )
        self.context_manager.add_assistant_message(self.session_id, answer=answer)
        return answer


class SessionManager:
    """会话管理：创建/获取/删除 session。

    - 每个 session 拥有独立的 context（ContextManager 按 session_id 隔离存储），
      同一用户的多个窗口互不影响，且可随时断点续聊；
    - LLM 客户端与工具注册表全局共享（无会话状态）。
    """

    def __init__(self, llm=None, tool_registry=None, context_manager=None,
                 logger=None, max_iterations=8):
        self.logger = logger or TraceLogger()
        self.tool_registry = tool_registry or build_default_registry()
        self.context_manager = context_manager or ContextManager()
        self.max_iterations = max_iterations
        self._default_llm = llm
        self.sessions = {}

    def _resolve_llm(self, llm):
        """确定会话使用的 LLM：显式传入 > 管理器默认 > 懒加载 DeepSeek 真实 API。"""
        effective = llm or self._default_llm
        if effective is None:
            effective = DeepSeekLLM()  # 未配置 DEEPSEEK_API_KEY 时抛出带指引的 LLMError
            self._default_llm = effective
        return effective

    def create_session(self, llm=None):
        session_id = uuid.uuid4().hex[:8]
        session = Session(
            session_id=session_id,
            context_manager=self.context_manager,
            llm=self._resolve_llm(llm),
            tool_registry=self.tool_registry,
            logger=self.logger,
            max_iterations=self.max_iterations,
        )
        self.sessions[session_id] = session
        self.logger.trace("session_created", {"session_id": session_id}, session_id)
        return session_id

    def chat(self, session_id, user_input):
        """便捷入口：在指定会话中执行一次 Agent 循环。"""
        return self.get_session(session_id).run(user_input)

    def get_session(self, session_id):
        if session_id not in self.sessions:
            raise SessionError(session_id, "session 不存在")
        return self.sessions[session_id]

    def has_session(self, session_id):
        return session_id in self.sessions

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.context_manager.clear(session_id)
            self.logger.trace("session_deleted", {"session_id": session_id}, session_id)

    def get_all_sessions(self):
        return list(self.sessions.keys())
