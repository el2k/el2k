"""AgentRuntime：对外的运行时门面。

组合 SessionManager / ContextManager / TraceLogger，提供：
- 会话生命周期（创建/切换/列出/结束）
- chat：一次完整的 Agent 循环（含异常兜底，出错时返回 status=error 而不是崩溃）
- trace / 日志查询
"""

from .session import SessionManager


class AgentRuntime:
    def __init__(self, llm=None, session_manager=None, max_iterations=8):
        self.session_manager = session_manager or SessionManager(
            llm=llm, max_iterations=max_iterations
        )
        self.context_manager = self.session_manager.context_manager
        self.logger = self.session_manager.logger

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    def start_session(self, llm=None):
        """创建一个新会话窗口，返回 session_id。可传入自定义 LLM（测试用）。"""
        return self.session_manager.create_session(llm=llm)

    def list_sessions(self):
        return self.session_manager.get_all_sessions()

    def end_session(self, session_id):
        self.session_manager.delete_session(session_id)

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------

    def chat(self, session_id, user_input):
        """在指定会话中执行一次 Agent 循环。

        返回：
            {"session_id", "answer", "thought", "tool_calls",
             "iterations", "turn", "status"}
        任何内部异常都被捕获并转为 status=error 的结构化结果，
        保证调用方（CLI/API）不崩溃。
        """
        try:
            session = self.session_manager.get_session(session_id)
        except Exception as e:
            return self._error_result(session_id, e)

        try:
            result = session.run(user_input)
            return {
                "session_id": session_id,
                "answer": result["answer"],
                "thought": result.get("thought", ""),
                "tool_calls": result["tool_calls"],
                "iterations": result["iterations"],
                "turn": self.context_manager.get_turn_count(session_id),
                "status": "success",
            }
        except Exception as e:
            self.logger.error(f"runtime error: {type(e).__name__}: {e}", session_id)
            return self._error_result(session_id, e)

    def _error_result(self, session_id, error):
        return {
            "session_id": session_id,
            "answer": f"（处理出错：{error}）",
            "thought": "",
            "tool_calls": [],
            "iterations": 0,
            "turn": 0,
            "status": "error",
        }

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def get_history(self, session_id):
        """查看某个会话的上下文（消息历史 + 摘要 + 轮次）。"""
        return self.context_manager.get_history(session_id)

    def get_traces(self, session_id):
        """查看某个会话的结构化执行 trace。"""
        return self.logger.get_traces(session_id)

    def get_logs(self, session_id=None):
        """查看文本日志（可按会话过滤）。"""
        return self.logger.get_logs(session_id)
