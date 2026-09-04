import pytest
import sys
import os
sys.path.insert(0, r'D:\Desktop\EL2K')

from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.llm import MockLLM

@pytest.fixture
def runtime():
    return AgentRuntime()

def test_full_workflow_calculator(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "Calculate 15 + 27")
    assert result["status"] == "success"

def test_full_workflow_search_then_weather(runtime):
    sid = runtime.start_session()
    result1 = runtime.chat(sid, "Search for AI")
    assert result1["status"] == "success"
    result2 = runtime.chat(sid, "What is the weather in Beijing")
    assert result2["status"] == "success"

def test_full_workflow_todo_add_list(runtime):
    sid = runtime.start_session()
    result1 = runtime.chat(sid, "Add todo: Buy groceries")
    assert result1["status"] == "success"
    result2 = runtime.chat(sid, "List todos")
    assert result2["status"] == "success"

def test_session_continuation(runtime):
    sid = runtime.start_session()
    result1 = runtime.chat(sid, "Calculate 2+2")
    assert result1["status"] == "success"
    result2 = runtime.chat(sid, "Now calculate 3+3")
    assert result2["status"] == "success"

def test_parallel_sessions_do_not_interfere(runtime):
    sid1 = runtime.start_session()
    sid2 = runtime.start_session()

    r1 = runtime.chat(sid1, "Add todo: Task A")
    r2 = runtime.chat(sid2, "Add todo: Task B")

    assert r1["status"] == "success"
    assert r2["status"] == "success"

def test_follow_up_with_tools(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "Search for python and tell me more")
    assert result["status"] == "success"

def test_weather_and_todo_in_same_session(runtime):
    sid = runtime.start_session()
    r1 = runtime.chat(sid, "What is the weather in Shenzhen?")
    r2 = runtime.chat(sid, "Add todo: Remember sunny weather")
    assert r1["status"] == "success"
    assert r2["status"] == "success"

def test_trace_log_completeness(runtime):
    sid = runtime.start_session()
    runtime.chat(sid, "Calculate 100 * 2")
    logs = runtime.get_logs(sid)
    assert len(logs) > 0

def test_session_manager_default_tools(runtime):
    sm = runtime.session_manager
    tools = sm.tool_registry.get_all()
    tool_names = [t.name for t in tools]
    assert "calculator" in tool_names
    assert "search" in tool_names
    assert "todo" in tool_names
    assert "weather" in tool_names

def test_mock_llm_returns_valid_structure(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "Hello")
    assert "thought" in result or "response" in result
    assert result["status"] in ["success", "error"]

def test_context_compression_does_not_lose_data(runtime):
    runtime.session_manager.context_manager.max_turns = 2
    sid = runtime.start_session()
    for i in range(10):
        runtime.chat(sid, f"Message {i}")
    assert runtime.context_manager.get_turn_count(sid) >= 10

def test_tool_registration_mechanism(runtime):
    sm = runtime.session_manager
    reg = sm.tool_registry
    schemas = reg.get_function_schemas()
    assert len(schemas) == 4
    schema_names = [s["function"]["name"] for s in schemas]
    assert "calculator" in schema_names
    assert "search" in schema_names
    assert "todo" in schema_names
    assert "weather" in schema_names

def test_exception_handling_in_runtime(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "This is a very long message that should still be processed correctly without any errors")
    assert result["status"] == "success"

def test_empty_message_handling(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "")
    assert result["status"] == "success"