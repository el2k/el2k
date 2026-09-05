"""内置工具集合与默认注册表构建。

每个工具模块导出一个配置好的 Tool 实例（名称 + 描述 + 参数 Schema + 实现），
build_default_registry() 创建全新 ToolRegistry 并注册全部内置工具。
不使用模块级全局注册表，避免多处共享同一可变状态。
"""

from ..tool_registry import ToolRegistry
from .calculator import TOOL as calculator_tool
from .search import TOOL as search_tool
from .todo import TOOL as todo_tool
from .weather import TOOL as weather_tool

BUILTIN_TOOLS = [calculator_tool, search_tool, todo_tool, weather_tool]


def build_default_registry():
    """创建注册了全部内置工具的新 ToolRegistry。"""
    registry = ToolRegistry()
    for tool in BUILTIN_TOOLS:
        registry.register_tool(tool)
    return registry
