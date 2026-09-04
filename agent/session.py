import uuid
from agent.context import ContextManager
from agent.llm import LLMInterface, MockLLM
from agent.tool_registry import ToolRegistry, get_default_registry
from agent.logger import TraceLogger
from agent.exceptions import SessionError

reg = get_default_registry()

class Session:
    def __init__(self, session_id, context_manager, llm, tool_registry, logger):
        self.session_id = session_id
        self.context_manager = context_manager
        self.llm = llm
        self.tool_registry = tool_registry
        self.logger = logger

    def run(self, user_input):
        self.context_manager.add_message(self.session_id, "user", user_input)
        self.logger.info(f"User input: {user_input}", self.session_id)

        messages = self.context_manager.get_messages(self.session_id)
        self.logger.llm_request(messages, self.session_id)

        response = self.llm.chat(messages, self.session_id)
        thought = response.get("thought", "")
        tool_calls = response.get("tool_calls", [])
        answer = response.get("answer", "")

        self.context_manager.add_message(self.session_id, "assistant", answer)
        if thought:
            self.context_manager.add_message(self.session_id, "assistant", thought)

        if tool_calls:
            for tc in tool_calls:
                tool_name = tc["name"]
                args = tc["args"]
                self.logger.tool_call(tool_name, args, self.session_id)
                try:
                    result = self.tool_registry.execute(tool_name, args)
                    self.logger.tool_result(tool_name, result, self.session_id)
                    self.context_manager.add_tool_result(self.session_id, tool_name, result)
                    self.context_manager.add_message(self.session_id, "tool", f"{tool_name}({args}) -> {result}")
                except Exception as e:
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    self.logger.error(error_msg, self.session_id)
                    self.context_manager.add_message(self.session_id, "tool", error_msg)

            messages_after_tools = self.context_manager.get_messages(self.session_id)
            follow_up = self.llm.chat(messages_after_tools, self.session_id)
            final_answer = follow_up.get("answer", "")
            if follow_up.get("thought"):
                self.context_manager.add_message(self.session_id, "assistant", follow_up["thought"])
            self.context_manager.add_message(self.session_id, "assistant", final_answer)
            return final_answer, tool_calls

        return answer, tool_calls

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.tool_registry = ToolRegistry()
        self.logger = TraceLogger()
        self.context_manager = ContextManager(max_turns=20)
        self._setup_default_tools()
        self._init_llm()

    def _setup_default_tools(self):
        self.tool_registry.register_tool(reg.get("calculator"))
        self.tool_registry.register_tool(reg.get("search"))
        self.tool_registry.register_tool(reg.get("todo"))
        self.tool_registry.register_tool(reg.get("weather"))
        self.logger.info("Default tools registered", "system")

    def _init_llm(self):
        self._default_llm = MockLLM()
        self._default_llm._tool_registry = self.tool_registry

    def create_session(self, llm=None):
        session_id = str(uuid.uuid4())[:8]
        llm = llm or self._default_llm
        llm._tool_registry = self.tool_registry
        session = Session(session_id, self.context_manager, llm, self.tool_registry, self.logger)
        self.sessions[session_id] = session
        self.logger.info(f"Created session {session_id}", session_id)
        return session_id

    def get_session(self, session_id):
        if session_id not in self.sessions:
            raise SessionError(session_id, "Session not found")
        return self.sessions[session_id]

    def has_session(self, session_id):
        return session_id in self.sessions

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.context_manager.clear(session_id)

    def get_all_sessions(self):
        return list(self.sessions.keys())