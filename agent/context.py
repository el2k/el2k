from collections import deque
from .exceptions import ContextOverflowError

class ContextManager:
    def __init__(self, max_turns=20):
        self.max_turns = max_turns
        self._contexts = {}

    def get_context(self, session_id):
        if session_id not in self._contexts:
            self._contexts[session_id] = {
                "messages": deque(maxlen=self.max_turns * 2),
                "turn_count": 0,
                "summary": ""
            }
        return self._contexts[session_id]

    def add_message(self, session_id, role, content):
        ctx = self.get_context(session_id)
        ctx["messages"].append({"role": role, "content": content})
        ctx["turn_count"] += 1
        if ctx["turn_count"] > self.max_turns:
            self._compress(session_id)

    def get_messages(self, session_id):
        ctx = self.get_context(session_id)
        return list(ctx["messages"])

    def add_tool_result(self, session_id, tool_name, result):
        ctx = self.get_context(session_id)
        ctx["messages"].append({
            "role": "tool",
            "content": f"Tool '{tool_name}' result: {result}"
        })

    def _compress(self, session_id):
        ctx = self.get_context(session_id)
        messages = ctx["messages"]
        if len(messages) < 1:
            return ""

        if len(messages) < 4:
            ctx["summary"] = "; ".join([m.get("content", "") for m in messages])
            return ctx["summary"]

        old_messages = list(messages)[:len(messages)//2]
        recent_messages = list(messages)[len(messages)//2:]

        summary_parts = []
        for msg in old_messages:
            content = msg.get("content", "")
            if len(content) > 100:
                summary_parts.append(content[:100] + "...")
            else:
                summary_parts.append(content)

        ctx["summary"] = "; ".join(summary_parts[-5:])
        ctx["messages"] = deque(recent_messages, maxlen=self.max_turns * 2)
        return ctx["summary"]

    def get_turn_count(self, session_id):
        ctx = self.get_context(session_id)
        return ctx["turn_count"]

    def clear(self, session_id):
        if session_id in self._contexts:
            del self._contexts[session_id]