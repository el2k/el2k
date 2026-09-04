from agent.session import SessionManager, Session
from agent.context import ContextManager

class AgentRuntime:
    def __init__(self, session_manager=None):
        self.session_manager = session_manager or SessionManager()
        self.context_manager = self.session_manager.context_manager
        self.logger = self.session_manager.logger

    def start_session(self):
        session_id = self.session_manager.create_session()
        return session_id

    def start_session_with_llm(self, llm):
        session_id = self.session_manager.create_session(llm=llm)
        return session_id

    def chat(self, session_id, user_input):
        session = self.session_manager.get_session(session_id)
        self.logger.info(f"Processing request in session {session_id}", session_id)
        try:
            response, tool_calls = session.run(user_input)
            return {
                "session_id": session_id,
                "response": response,
                "tool_calls": tool_calls,
                "turn": self.context_manager.get_turn_count(session_id),
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}", session_id)
            return {
                "session_id": session_id,
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "turn": self.context_manager.get_turn_count(session_id),
                "status": "error"
            }

    def list_sessions(self):
        return self.session_manager.get_all_sessions()

    def end_session(self, session_id):
        self.session_manager.delete_session(session_id)

    def get_logs(self, session_id=None):
        return self.logger.get_logs(session_id)