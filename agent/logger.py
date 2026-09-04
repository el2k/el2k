import datetime
import os

class TraceLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"agent_{datetime.date.today()}.log")

    def _write(self, level, message, session_id=None):
        timestamp = datetime.datetime.now().isoformat()
        sid = f"[{session_id}] " if session_id else ""
        line = f"{timestamp} [{level}] {sid}{message}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def trace(self, message, session_id=None):
        self._write("TRACE", message, session_id)

    def info(self, message, session_id=None):
        self._write("INFO", message, session_id)

    def warn(self, message, session_id=None):
        self._write("WARN", message, session_id)

    def error(self, message, session_id=None):
        self._write("ERROR", message, session_id)

    def tool_call(self, tool_name, args, session_id=None):
        self._write("TOOL_CALL", f"{tool_name}({args})", session_id)

    def tool_result(self, tool_name, result, session_id=None):
        self._write("TOOL_RESULT", f"{tool_name} -> {result}", session_id)

    def llm_request(self, messages, session_id=None):
        self._write("LLM_REQUEST", f"Messages count: {len(messages)}", session_id)

    def llm_response(self, response, session_id=None):
        self._write("LLM_RESPONSE", f"{response}", session_id)

    def get_logs(self, session_id=None):
        if not os.path.exists(self.log_file):
            return []
        logs = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if session_id is None or f"[{session_id}]" in line:
                    logs.append(line.strip())
        return logs

    def clear_logs(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)