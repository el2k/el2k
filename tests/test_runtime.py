import pytest
import sys
import os
sys.path.insert(0, r'D:\Desktop\EL2K')

from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.llm import MockLLM
from agent.tool_registry import ToolRegistry, Tool

@pytest.fixture
def runtime():
    return AgentRuntime()

def test_runtime_creation(runtime):
    assert runtime is not None
    assert runtime.session_manager is not None

def test_runtime_start_session(runtime):
    sid = runtime.start_session()
    assert sid is not None
    assert runtime.session_manager.has_session(sid)

def test_runtime_chat(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "Hello there")
    assert result["status"] == "success"
    assert "response" in result
    assert "session_id" in result

def test_runtime_chat_calculator(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "Calculate 2+2")
    assert result["status"] == "success"

def test_runtime_chat_weather(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "What is the weather in Beijing")
    assert result["status"] == "success"

def test_runtime_chat_search(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "Search for python programming")
    assert result["status"] == "success"

def test_runtime_chat_todo(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "Add todo: test task")
    assert result["status"] == "success"

def test_runtime_multiple_sessions(runtime):
    sid1 = runtime.start_session()
    sid2 = runtime.start_session()

    r1 = runtime.chat(sid1, "Calculate 5*5")
    r2 = runtime.chat(sid2, "What is the weather in Shanghai")

    assert r1["status"] == "success"
    assert r2["status"] == "success"
    assert r1["session_id"] == sid1
    assert r2["session_id"] == sid2

def test_runtime_independent_sessions(runtime):
    sid1 = runtime.start_session()
    sid2 = runtime.start_session()

    runtime.chat(sid1, "Add todo: session1 task")
    runtime.chat(sid2, "List todos")

    assert runtime.context_manager.get_turn_count(sid1) > 0
    assert runtime.context_manager.get_turn_count(sid2) > 0

def test_runtime_end_session(runtime):
    sid = runtime.start_session()
    assert runtime.session_manager.has_session(sid)
    runtime.end_session(sid)
    assert not runtime.session_manager.has_session(sid)

def test_runtime_list_sessions(runtime):
    sid1 = runtime.start_session()
    sid2 = runtime.start_session()
    sessions = runtime.list_sessions()
    assert len(sessions) >= 2

def test_runtime_turn_count(runtime):
    sid = runtime.start_session()
    assert runtime.context_manager.get_turn_count(sid) == 0
    runtime.chat(sid, "Hello")
    assert runtime.context_manager.get_turn_count(sid) > 0

def test_runtime_error_handling(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "")
    assert result["status"] == "success"

def test_runtime_custom_llm(runtime):
    sid = runtime.start_session_with_llm(MockLLM())
    result = runtime.chat(sid, "Hello with mock LLM")
    assert result["status"] == "success"

def test_runtime_get_logs(runtime):
    sid = runtime.start_session()
    runtime.chat(sid, "Test logging")
    logs = runtime.get_logs(sid)
    assert len(logs) > 0

def test_runtime_max_turns(runtime):
    runtime.session_manager.context_manager.max_turns = 3
    sid = runtime.start_session()
    for i in range(5):
        runtime.chat(sid, f"Message {i}")
    assert runtime.context_manager.get_turn_count(sid) >= 5

def test_runtime_tool_trace(runtime):
    sid = runtime.start_session()
    runtime.chat(sid, "Calculate 10+20")
    logs = runtime.get_logs(sid)
    has_tool_call = any("TOOL_CALL" in l for l in logs)
    has_tool_result = any("TOOL_RESULT" in l for l in logs)
    assert has_tool_call or has_tool_result