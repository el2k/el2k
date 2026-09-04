import pytest
import sys
import os
sys.path.insert(0, r'D:\Desktop\EL2K')

from agent.session import SessionManager, Session
from agent.context import ContextManager
from agent.llm import MockLLM, LLMInterface
from agent.tool_registry import ToolRegistry
from agent.logger import TraceLogger

@pytest.fixture
def session_manager():
    sm = SessionManager()
    return sm

def test_create_session(session_manager):
    sid = session_manager.create_session()
    assert sid is not None
    assert len(sid) == 8
    assert session_manager.has_session(sid)

def test_independent_sessions(session_manager):
    sid1 = session_manager.create_session()
    sid2 = session_manager.create_session()
    assert sid1 != sid2

def test_session_independent_context(session_manager):
    sid1 = session_manager.create_session()
    sid2 = session_manager.create_session()

    session1 = session_manager.get_session(sid1)
    session2 = session_manager.get_session(sid2)

    cm = session_manager.context_manager
    cm.add_message(sid1, "user", "Session 1 message")
    cm.add_message(sid2, "user", "Session 2 message")

    msgs1 = cm.get_messages(sid1)
    msgs2 = cm.get_messages(sid2)

    assert len(msgs1) == 1
    assert len(msgs2) == 1
    assert msgs1[0]["content"] == "Session 1 message"
    assert msgs2[0]["content"] == "Session 2 message"

def test_session_chat(session_manager):
    sid = session_manager.create_session()
    result = session_manager.chat(sid, "Hello")
    assert result["status"] == "success"
    assert "response" in result

def test_session_chat_with_tools(session_manager):
    sid = session_manager.create_session()
    result = session_manager.chat(sid, "Calculate 2+2")
    assert result["status"] == "success"

def test_search_in_session(session_manager):
    sid = session_manager.create_session()
    result = session_manager.chat(sid, "Search for python")
    assert result["status"] == "success"
    assert "response" in result

def test_weather_in_session(session_manager):
    sid = session_manager.create_session()
    result = session_manager.chat(sid, "What is the weather in Shanghai")
    assert result["status"] == "success"

def test_todo_in_session(session_manager):
    sid = session_manager.create_session()
    result = session_manager.chat(sid, "Add todo: learn agent")
    assert result["status"] == "success"

def test_delete_session(session_manager):
    sid = session_manager.create_session()
    assert session_manager.has_session(sid)
    session_manager.delete_session(sid)
    assert not session_manager.has_session(sid)

def test_context_turn_count(session_manager):
    sid = session_manager.create_session()
    cm = session_manager.context_manager
    assert cm.get_turn_count(sid) == 0
    cm.add_message(sid, "user", "test")
    assert cm.get_turn_count(sid) == 1

def test_session_not_found(session_manager):
    with pytest.raises(Exception):
        session_manager.get_session("nonexistent")

def test_multiple_sessions_independent_todos(session_manager):
    sid1 = session_manager.create_session()
    sid2 = session_manager.create_session()

    session_manager.chat(sid1, "Add todo: task from session 1")
    session_manager.chat(sid2, "List todos")

    assert session_manager.context_manager.get_turn_count(sid1) > 0
    assert session_manager.context_manager.get_turn_count(sid2) > 0

def test_session_list(session_manager):
    sid1 = session_manager.create_session()
    sid2 = session_manager.create_session()
    sessions = session_manager.list_sessions()
    assert len(sessions) >= 2
    assert sid1 in sessions
    assert sid2 in sessions