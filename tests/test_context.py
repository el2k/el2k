import pytest
import sys
import os
sys.path.insert(0, r'D:\Desktop\EL2K')

from agent.context import ContextManager
from agent.llm import MockLLM
from agent.tool_registry import ToolRegistry
from agent.logger import TraceLogger

@pytest.fixture
def context_manager():
    return ContextManager(max_turns=5)

def test_context_creation(context_manager):
    sid = "test_session_1"
    ctx = context_manager.get_context(sid)
    assert ctx is not None
    assert len(ctx["messages"]) == 0
    assert ctx["turn_count"] == 0

def test_add_message(context_manager):
    sid = "test_msg"
    context_manager.add_message(sid, "user", "Hello")
    messages = context_manager.get_messages(sid)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"

def test_add_multiple_messages(context_manager):
    sid = "test_multi"
    context_manager.add_message(sid, "user", "Msg 1")
    context_manager.add_message(sid, "assistant", "Reply 1")
    context_manager.add_message(sid, "user", "Msg 2")
    messages = context_manager.get_messages(sid)
    assert len(messages) == 3

def test_add_tool_result(context_manager):
    sid = "test_tool"
    context_manager.add_message(sid, "user", "Calculate 2+2")
    context_manager.add_tool_result(sid, "calculator", {"result": 4})
    messages = context_manager.get_messages(sid)
    assert len(messages) == 2
    assert messages[1]["role"] == "tool"

def test_context_overflow_and_compression(context_manager):
    sid = "test_compress"
    for i in range(15):
        context_manager.add_message(sid, "user", f"Message {i}")
        context_manager.add_message(sid, "assistant", f"Reply {i}")

    assert context_manager.get_turn_count(sid) > 5
    summary = context_manager._compress(sid)
    assert summary is not None
    assert len(summary) > 0

def test_context_max_turns_limit(context_manager):
    sid = "test_max"
    for i in range(30):
        context_manager.add_message(sid, "user", f"Msg {i}")

    messages = context_manager.get_messages(sid)
    max_len = context_manager.max_turns * 2
    assert len(messages) <= max_len

def test_context_clear(context_manager):
    sid = "test_clear"
    context_manager.add_message(sid, "user", "Hello")
    assert len(context_manager.get_messages(sid)) == 1
    context_manager.clear(sid)
    ctx = context_manager.get_context(sid)
    assert len(ctx["messages"]) == 0
    assert ctx["turn_count"] == 0

def test_different_sessions_independent(context_manager):
    sid1 = "session_a"
    sid2 = "session_b"
    context_manager.add_message(sid1, "user", "A message")
    context_manager.add_message(sid2, "user", "B message")

    msgs1 = context_manager.get_messages(sid1)
    msgs2 = context_manager.get_messages(sid2)

    assert msgs1[0]["content"] == "A message"
    assert msgs2[0]["content"] == "B message"

def test_mock_llm_parse_response():
    llm = MockLLM()
    text = '<thought>Let me calculate.</thought><answer>The answer is 42</answer>'
    result = llm._parse_response(text)
    assert result["thought"] == "Let me calculate."
    assert result["answer"] == "The answer is 42"
    assert result["tool_calls"] == []

def test_mock_llm_parse_tool_calls():
    llm = MockLLM()
    text = '<thought>Let me search.</thought><invoke name="search"><params>{"query": "test"}</params></invoke><answer>Results found.</answer>'
    result = llm._parse_response(text)
    assert len(result["tool_calls"]) > 0
    assert result["tool_calls"][0]["name"] == "search"

def test_mock_llm_chat_with_messages():
    llm = MockLLM()
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    result = llm.chat(messages, "test_session")
    assert "thought" in result
    assert "answer" in result

def test_context_tool_result_integration(context_manager):
    sid = "test_integration"
    context_manager.add_message(sid, "user", "Calculate")
    context_manager.add_tool_result(sid, "calculator", 4)
    messages = context_manager.get_messages(sid)
    tool_msg = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msg) == 1
    assert "calculator" in tool_msg[0]["content"]