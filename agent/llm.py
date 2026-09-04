import json
import re
from .exceptions import LLMError
from .logger import TraceLogger

class LLMInterface:
    def __init__(self, api_key=None, model="gpt-4o-mini", base_url=None):
        self.client = None
        self.model = model
        self.base_url = base_url
        self.use_mock = False
        self._tool_registry = None
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.use_mock = True

    def _parse_response(self, text):
        text = text.strip()
        try:
            parsed = json.loads(text)
            return {
                "thought": parsed.get("thought", ""),
                "tool_calls": parsed.get("tool_calls", []),
                "answer": parsed.get("answer", "")
            }
        except json.JSONDecodeError:
            pass

        thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
        tool_call_match = re.search(r'<invoke name="([^"]+)">(.*?)</invoke>', text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

        if thought_match or tool_call_match or answer_match:
            return {
                "thought": thought_match.group(1).strip() if thought_match else "",
                "tool_calls": self._parse_tool_calls(tool_call_match.group(1).strip()) if tool_call_match else [],
                "answer": answer_match.group(1).strip() if answer_match else ""
            }

        return {"thought": "", "tool_calls": [], "answer": text}

    def _parse_tool_calls(self, text):
        calls = []
        pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
        for match in re.finditer(pattern, text, re.DOTALL):
            name = match.group(1)
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {"query": args_str}
            calls.append({"name": name, "args": args})
        return calls

    def chat(self, messages, session_id=None):
        if self.use_mock:
            return self._mock_chat(messages, session_id)
        return self._real_chat(messages, session_id)

    def _real_chat(self, messages, session_id=None):
        from .exceptions import LLMError
        try:
            tools = self.get_function_schemas() or None
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None
            )
            content = response.choices[0].message.content or ""
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                parsed_calls = []
                for tc in tool_calls:
                    parsed_calls.append({"name": tc.function.name, "args": json.loads(tc.function.arguments)})
                return {"thought": content, "tool_calls": parsed_calls, "answer": ""}
            return self._parse_response(content)
        except Exception as e:
            raise LLMError(str(e))

    def _mock_chat(self, messages, session_id=None):
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        response_text = self._generate_mock_response(last_user_msg, messages)
        logger = TraceLogger()
        logger.llm_response(response_text, session_id)
        return self._parse_response(response_text)

    def _generate_mock_response(self, user_msg, messages):
        user_lower = user_msg.lower()
        if any(kw in user_lower for kw in ["calculate", "计算", "what is", "plus", "minus", "multiply", "divide", "2 +", "3 *", "5 -"]):
            return '<thought>Let me calculate that for you.</thought><answer>Here is the calculation result.</answer>'
        if any(kw in user_lower for kw in ["search", "搜索", "find", "lookup"]):
            return '<thought>Searching for information.</thought><answer>Here are the search results.</answer>'
        if any(kw in user_lower for kw in ["weather", "温度", "forecast"]):
            city = "Beijing"
            for city_name in ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu"]:
                if city_name.lower() in user_lower:
                    city = city_name
                    break
            return f'<thought>Let me check the weather for {city}.</thought><answer>Here is the weather forecast for {city}.</answer>'
        if any(kw in user_lower for kw in ["todo", "task", "任务", "待办"]):
            if any(kw in user_lower for kw in ["add", "创建", "记录", "new"]):
                return '<thought>I will add this task to the todo list.</thought><answer>Task added to your todo list.</answer>'
            return '<thought>Let me check the todo list.</thought><answer>Here are your todos.</answer>'
        if any(kw in user_lower for kw in ["bye", "goodbye", "exit", "quit", "再见", "退出"]):
            return '<thought>Goodbye! Have a great day.</thought><answer>Goodbye! Have a great day.</answer>'
        return '<thought>Let me think about that.</thought><answer>That is an interesting question. Let me think more about it.</answer>'

    def get_function_schemas(self):
        if self._tool_registry:
            return self._tool_registry.get_function_schemas()
        return None

class MockLLM(LLMInterface):
    def __init__(self, model="mock-llm"):
        self._tool_registry = None
        self.model = model
        self.use_mock = True
        self.client = None
        self.base_url = None