from .exceptions import ToolNotFoundError

_default_registry = None

def get_default_registry():
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry

class Tool:
    def __init__(self, name, description, schema, func):
        self.name = name
        self.description = description
        self.schema = schema
        self.func = func

    def __repr__(self):
        return f"Tool(name='{self.name}', description='{self.description}')"

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name=None, description=None, schema=None):
        if callable(name) and description is None and schema is None:
            func = name
            return func
        def decorator(func):
            tool_name = name or func.__name__
            self._tools[tool_name] = Tool(tool_name, description or "", schema or {}, func)
            return func
        return decorator

    def register_tool(self, tool):
        if not isinstance(tool, Tool):
            raise ValueError("Can only register Tool instances")
        self._tools[tool.name] = tool

    def get(self, name):
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def has(self, name):
        return name in self._tools

    def get_all(self):
        return list(self._tools.values())

    def get_function_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            }
            for tool in self._tools.values()
        ]

    def execute(self, name, args):
        tool = self.get(name)
        return tool.func(**args)