"""工具注册机制。

每个工具是一个 Tool 实例，包含四个要素：
- name        工具名（LLM 调用时使用）
- description 工具用途描述（LLM 据此判断何时该用这个工具）
- schema      参数 JSON Schema（OpenAI function-calling 格式，LLM 据此构造参数）
- func        工具的 Python 实现

工作方式：
- registry.get_function_schemas() 把全部工具导出为 OpenAI tools 格式，
  随每次 LLM 请求下发，由 LLM 自主决策是否调用、如何传参；
- registry.execute(name, args, session_id) 执行工具：
  * 轻量校验 required 参数（缺参直接报错，不执行）；
  * 若工具函数声明了 session_id 形参，则自动注入当前会话 ID
    （todo 这类有状态工具借此实现"按会话隔离存储"）；
  * 工具内部异常统一包装为 ToolExecutionError，向上层提供统一错误接口。
"""

import inspect
from typing import Any

from .exceptions import ToolExecutionError, ToolNotFoundError


class Tool:
    """工具 = 名称 + 描述 + 参数 Schema + 实现函数。"""

    def __init__(self, name, description, schema, func):
        self.name = name
        self.description = description
        self.schema = schema or {"type": "object"}
        self.func = func

    def __repr__(self):
        return f"Tool(name={self.name!r}, description={self.description!r})"

    def validate_args(self, args):
        """轻量参数校验：检查 schema.required 中的字段是否齐全。"""
        args = args or {}
        required = self.schema.get("required", []) or []
        missing = [key for key in required if key not in args]
        if missing:
            raise ToolExecutionError(self.name, f"缺少必填参数: {', '.join(missing)}")
        return args


class ToolRegistry:
    """工具注册表：注册、查询、导出 Schema、执行。"""

    def __init__(self):
        self._tools = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register_tool(self, tool):
        """注册一个 Tool 实例（同名重复注册时后者覆盖前者）。"""
        if not isinstance(tool, Tool):
            raise ValueError("register_tool 只接受 Tool 实例")
        self._tools[tool.name] = tool

    def register(self, name=None, description=None, schema=None):
        """装饰器注册：@reg.register(name=..., description=..., schema=...)"""

        def decorator(func):
            self.register_tool(Tool(
                name=name or func.__name__,
                description=description or "",
                schema=schema,
                func=func,
            ))
            return func

        return decorator

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, name):
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def has(self, name):
        return name in self._tools

    def get_all(self):
        return list(self._tools.values())

    def get_function_schemas(self):
        """导出 OpenAI function-calling 格式的工具 Schema 列表。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
            for tool in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def execute(self, name, args=None, session_id=None):
        """按名称执行工具。

        - name 不存在        -> ToolNotFoundError
        - 缺少必填参数       -> ToolExecutionError
        - 工具内部抛异常     -> ToolExecutionError（包装原始错误）
        - 工具声明 session_id 形参 -> 自动注入当前会话 ID
        """
        tool = self.get(name)
        args = tool.validate_args(args)

        kwargs = dict[Any, Any](args)
        if "session_id" in inspect.signature(tool.func).parameters:
            kwargs.setdefault("session_id", session_id)

        try:
            return tool.func(**kwargs)
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(name, f"{type(e).__name__}: {e}") from e
