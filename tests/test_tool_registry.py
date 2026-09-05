"""工具注册机制测试：注册、Schema 导出、执行、session_id 注入、参数校验。"""

import pytest

from agent.exceptions import ToolExecutionError, ToolNotFoundError
from agent.tool_registry import Tool, ToolRegistry
from agent.tools import build_default_registry
from agent.tools.todo import todo


@pytest.fixture
def registry():
    return build_default_registry()


# ---------------------------------------------------------------------------
# 注册与查询
# ---------------------------------------------------------------------------

def test_builtin_tools_registered(registry):
    names = [t.name for t in registry.get_all()]
    assert set(names) == {"calculator", "search", "todo", "weather"}


def test_each_tool_has_name_description_schema():
    for tool in build_default_registry().get_all():
        assert tool.name
        assert tool.description, f"{tool.name} 缺少描述"
        assert tool.schema.get("type") == "object"
        assert callable(tool.func)


def test_register_decorator():
    registry = ToolRegistry()

    @registry.register(
        name="echo",
        description="原样返回输入",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    def echo(text):
        return {"echo": text}

    assert registry.has("echo")
    assert registry.execute("echo", {"text": "hi"}) == {"echo": "hi"}


def test_register_tool_instance():
    registry = ToolRegistry()
    registry.register_tool(Tool("noop", "什么都不做", {"type": "object"}, lambda: "done"))
    assert registry.execute("noop") == "done"


def test_register_rejects_non_tool():
    with pytest.raises(ValueError):
        ToolRegistry().register_tool("not a tool")


def test_get_unknown_tool_raises():
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().get("no_such_tool")


# ---------------------------------------------------------------------------
# Schema 导出（OpenAI function-calling 格式）
# ---------------------------------------------------------------------------

def test_get_function_schemas_format(registry):
    schemas = registry.get_function_schemas()
    assert len(schemas) == 4
    for schema in schemas:
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] in {"calculator", "search", "todo", "weather"}
        assert func["description"]
        assert isinstance(func["parameters"], dict)


def test_calculator_schema_has_required_expression(registry):
    schema = {s["function"]["name"]: s for s in registry.get_function_schemas()}
    assert schema["calculator"]["function"]["parameters"]["required"] == ["expression"]
    assert schema["weather"]["function"]["parameters"]["required"] == ["city"]


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def test_execute_calculator(registry):
    result = registry.execute("calculator", {"expression": "6 * 7"})
    assert result["result"] == 42


def test_execute_with_session_id_injection():
    """声明了 session_id 形参的工具，执行时自动注入当前会话。"""
    registry = build_default_registry()
    sid = "reg-test-session"
    result = registry.execute("todo", {"action": "add", "task": "注入测试"}, session_id=sid)
    assert result["status"] == "added"
    listed = registry.execute("todo", {"action": "list"}, session_id=sid)
    assert [t["task"] for t in listed["tasks"]] == ["注入测试"]


def test_execute_missing_required_arg():
    registry = build_default_registry()
    with pytest.raises(ToolExecutionError, match="缺少必填参数"):
        registry.execute("calculator", {})  # 缺 expression


def test_execute_wraps_internal_errors():
    registry = ToolRegistry()

    @registry.register(name="boom", description="总是抛错",
                       schema={"type": "object", "properties": {}, "required": []})
    def boom():
        raise RuntimeError("炸了")

    with pytest.raises(ToolExecutionError, match="炸了"):
        registry.execute("boom")


def test_execute_unknown_tool():
    with pytest.raises(ToolNotFoundError):
        build_default_registry().execute("nope", {})
