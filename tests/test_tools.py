import pytest
import sys
import os
sys.path.insert(0, r'D:\Desktop\EL2K')

from agent.tool_registry import Tool, ToolRegistry
from agent.tools.calculator import calculator
from agent.tools.search import search
from agent.tools.todo import todo
from agent.tools.weather import weather

def test_tool_registration():
    reg = ToolRegistry()
    tool = Tool("test_tool", "A test tool", {"type": "object"}, lambda x: x)
    reg.register_tool(tool)
    assert reg.has("test_tool")
    assert len(reg.get_all()) == 1

def test_tool_registration_decorator():
    reg = ToolRegistry()
    @reg.register(name="my_calc", description="Test calculator", schema={"type": "object"})
    def my_calc(x):
        return x * 2
    assert reg.has("my_calc")
    assert reg.get("my_calc").name == "my_calc"

def test_calculator_tool():
    result = calculator(expression="2 + 3 * 4")
    assert result == 14

def test_calculator_with_sqrt():
    result = calculator(expression="sqrt(16)")
    assert result == 4.0

def test_calculator_with_power():
    result = calculator(expression="2 ** 10")
    assert result == 1024

def test_calculator_error():
    result = calculator(expression="1 / 0")
    assert "Error" in str(result)

def test_search_tool():
    result = search(query="test query", num_results=3)
    assert result["query"] == "test query"
    assert len(result["results"]) == 3
    assert result["total"] == 3

def test_search_default_results():
    result = search(query="python")
    assert len(result["results"]) == 5

def test_todo_add():
    import os
    todo_file = os.path.join(r"D:\Desktop\EL2K", ".todo_data.json")
    if os.path.exists(todo_file):
        os.remove(todo_file)
    result = todo(action="add", task="Test task")
    assert result["status"] == "added"
    assert result["total"] == 1

def test_todo_list():
    result = todo(action="list")
    assert result["status"] == "ok"
    assert result["total"] >= 1

def test_todo_remove():
    result = todo(action="add", task="To be removed")
    assert result["status"] == "added"
    todo_file = os.path.join(r"D:\Desktop\EL2K", ".todo_data.json")
    with open(todo_file, "r") as f:
        import json
        tasks = json.load(f)
    idx = len(tasks) - 1
    result = todo(action="remove", index=idx)
    assert result["status"] == "removed"

def test_weather_tool():
    result = weather(city="Beijing")
    assert result["city"] == "Beijing"
    assert "temperature" in result
    assert "condition" in result

def test_weather_unknown_city():
    result = weather(city="UnknownCity")
    assert result["city"] == "UnknownCity"
    assert "temperature" in result
    assert result["source"] == "mock"

def test_get_function_schemas():
    reg = ToolRegistry()
    reg.register_tool(Tool("calc", "Calculator", {"type": "object"}, calculator))
    schemas = reg.get_function_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "calc"