"""执行日志与工具调用 trace。

TraceLogger 同时输出两路：
1. 结构化 trace（内存，按 session 隔离，有界 deque）——供程序化查询与测试断言；
2. 文本日志（logs/agent_YYYY-MM-DD.log）——供人工排查。

trace 事件类型：
    session_created / session_deleted / user_input / llm_request / llm_response /
    tool_call / tool_result / final_answer / error / info
"""

import datetime
import json
import os
from collections import defaultdict, deque

from .utils import to_display_text, truncate


class TraceLogger:
    def __init__(self, log_dir="logs", file_logging=True, max_traces_per_session=1000):
        self.file_logging = file_logging
        self.max_traces_per_session = max_traces_per_session
        self._traces = defaultdict(lambda: deque(maxlen=max_traces_per_session))
        if file_logging:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(
                log_dir, f"agent_{datetime.date.today()}.log"
            )
        else:
            self.log_file = None

    # ------------------------------------------------------------------
    # 核心
    # ------------------------------------------------------------------

    def trace(self, event, data=None, session_id=None, level="INFO"):
        """记录一条结构化 trace，同时落盘一行文本日志。"""
        entry = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "event": event,
            "session_id": session_id,
            "data": data if data is not None else {},
        }
        if session_id is not None:
            self._traces[session_id].append(entry)
        if self.file_logging:
            sid = f"[{session_id}] " if session_id else ""
            payload = json.dumps(entry["data"], ensure_ascii=False, default=str)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{entry['time']} [{level}] {sid}{event} {payload}\n")
        return entry

    # ------------------------------------------------------------------
    # 语义化事件
    # ------------------------------------------------------------------

    def user_input(self, text, session_id):
        self.trace("user_input", {"input": truncate(text, 300)}, session_id)

    def llm_request(self, message_count, session_id, tools_count=0):
        self.trace(
            "llm_request",
            {"messages": message_count, "tools": tools_count},
            session_id,
        )

    def llm_response(self, response, session_id):
        tool_calls = response.get("tool_calls") or []
        self.trace(
            "llm_response",
            {
                "thought": truncate(response.get("thought", ""), 200),
                "tool_calls": [tc.get("name") for tc in tool_calls],
                "answer": truncate(response.get("answer", ""), 300),
            },
            session_id,
        )

    def tool_call(self, name, args, session_id):
        self.trace("tool_call", {"name": name, "args": args}, session_id)

    def tool_result(self, name, result, session_id, status="ok"):
        self.trace(
            "tool_result",
            {
                "name": name,
                "status": status,
                "result": truncate(to_display_text(result), 500),
            },
            session_id,
        )

    def final_answer(self, answer, session_id):
        self.trace("final_answer", {"answer": truncate(answer, 500)}, session_id)

    def error(self, message, session_id=None):
        self.trace("error", {"message": truncate(str(message), 500)},
                   session_id, level="ERROR")

    def info(self, message, session_id=None):
        self.trace("info", {"message": str(message)}, session_id)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_traces(self, session_id):
        """返回某个 session 的结构化 trace 列表（时间顺序）。"""
        return [dict(entry) for entry in self._traces.get(session_id, [])]

    def get_logs(self, session_id=None):
        """从文本日志文件中读取（可按 session 过滤）日志行。"""
        if not self.log_file or not os.path.exists(self.log_file):
            return []
        lines = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if session_id is None or f"[{session_id}]" in line:
                    lines.append(line.rstrip("\n"))
        return lines
