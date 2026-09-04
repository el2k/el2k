class AgentError(Exception):
    pass

class ToolNotFoundError(AgentError):
    def __init__(self, tool_name):
        super().__init__(f"Tool '{tool_name}' not found in registry")

class ToolExecutionError(AgentError):
    def __init__(self, tool_name, reason):
        super().__init__(f"Tool '{tool_name}' execution failed: {reason}")

class LLMError(AgentError):
    def __init__(self, message):
        super().__init__(f"LLM error: {message}")

class SessionError(AgentError):
    def __init__(self, session_id, message):
        super().__init__(f"Session '{session_id}' error: {message}")

class ContextOverflowError(AgentError):
    def __init__(self, max_turns):
        super().__init__(f"Context exceeds maximum turns ({max_turns})")

class ParseError(AgentError):
    def __init__(self, message):
        super().__init__(f"Parse error: {message}")