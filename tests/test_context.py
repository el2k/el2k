"""ContextManager 测试：消息格式、轮次计数、压缩与摘要、隔离性。"""

from agent.context import ContextManager


SID = "ctx-test-session"


def _fill_turns(cm, sid, n_turns=1, user_prefix="msg"):
    """每个"用户轮次"写入 user + assistant 一对消息。"""
    for i in range(n_turns):
        cm.add_user_message(sid, f"{user_prefix}-{i}")
        cm.add_assistant_message(sid, answer=f"reply-{i}")


# ---------------------------------------------------------------------------
# 消息格式（OpenAI 规范）
# ---------------------------------------------------------------------------

def test_user_and_assistant_message_format():
    cm = ContextManager()
    cm.add_user_message(SID, "你好")
    cm.add_assistant_message(SID, answer="你好，有什么可以帮你？")
    messages = cm.get_llm_messages(SID)
    # 第 1 条是系统提示词
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "你好"}
    assert messages[2] == {"role": "assistant", "content": "你好，有什么可以帮你？"}


def test_assistant_tool_call_message_format():
    cm = ContextManager()
    cm.add_assistant_message(
        SID,
        thought="需要计算",
        tool_calls=[{"id": "call_1", "name": "calculator", "args": {"expression": "1+1"}}],
    )
    messages = cm.get_llm_messages(SID)
    msg = messages[1]
    assert msg["role"] == "assistant"
    assert msg["content"] == "需要计算"
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "calculator"
    assert tc["function"]["arguments"] == '{"expression": "1+1"}'


def test_tool_result_message_pairs_with_tool_call_id():
    cm = ContextManager()
    cm.add_assistant_message(
        SID, tool_calls=[{"id": "call_9", "name": "search", "args": {"query": "x"}}]
    )
    cm.add_tool_result(SID, "call_9", {"total": 3})
    messages = cm.get_llm_messages(SID)
    tool_msg = messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_9"
    assert '"total": 3' in tool_msg["content"]


def test_long_tool_result_truncated():
    cm = ContextManager(max_tool_result_chars=50)
    cm.add_tool_result(SID, "call_x", "x" * 500)
    content = cm.get_llm_messages(SID)[1]["content"]
    assert len(content) < 100
    assert "已截断" in content


# ---------------------------------------------------------------------------
# 轮次计数（最大轮次限制的基础）
# ---------------------------------------------------------------------------

def test_turn_count_tracks_user_messages():
    cm = ContextManager()
    assert cm.get_turn_count(SID) == 0
    _fill_turns(cm, SID, n_turns=3)
    assert cm.get_turn_count(SID) == 3
    # assistant / tool 消息不计入轮次
    cm.add_assistant_message(SID, answer="extra")
    cm.add_tool_result(SID, "c1", "r")
    assert cm.get_turn_count(SID) == 3


# ---------------------------------------------------------------------------
# 压缩与摘要（memory 召回）
# ---------------------------------------------------------------------------

def test_compression_creates_summary_and_bounds_messages():
    cm = ContextManager(max_messages=12, keep_recent=6)
    sid = "compress-1"
    _fill_turns(cm, sid, n_turns=10)  # 20 条消息 > 12
    history = cm.get_history(sid)
    assert history["compress_count"] >= 1
    assert len(history["messages"]) <= 12
    assert history["summary"] != ""


def test_summary_is_injected_into_llm_messages():
    cm = ContextManager(max_messages=8, keep_recent=4)
    sid = "compress-2"
    _fill_turns(cm, sid, n_turns=8)
    messages = cm.get_llm_messages(sid)
    summary_msgs = [m for m in messages if m["role"] == "system" and "自动压缩生成" in m["content"]]
    assert len(summary_msgs) == 1
    # 摘要里应能追溯到被压缩掉的早期内容
    assert "msg-0" in summary_msgs[0]["content"]
    # 近期消息保留原文
    contents = [m.get("content", "") for m in messages]
    assert any("msg-7" in c for c in contents)


def test_compression_keeps_tool_pairs_intact():
    """压缩按整轮切割，assistant(tool_calls) 与 tool 消息对永远不被拆散。"""
    cm = ContextManager(max_messages=10, keep_recent=6)
    sid = "compress-3"
    for i in range(8):
        cm.add_user_message(sid, f"q{i}")
        cm.add_assistant_message(
            sid,
            tool_calls=[{"id": f"c{i}", "name": "calculator", "args": {"expression": f"{i}+1"}}],
        )
        cm.add_tool_result(sid, f"c{i}", {"result": i + 1})
        cm.add_assistant_message(sid, answer=f"a{i}")

    messages = cm.get_history(sid)["messages"]
    for idx, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                follow = messages[idx + 1: idx + 1 + len(msg["tool_calls"])]
                assert any(
                    m["role"] == "tool" and m["tool_call_id"] == tc["id"] for m in follow
                ), "tool 消息与 tool_call_id 配对被打散"


def test_compression_preserves_recent_messages_verbatim():
    cm = ContextManager(max_messages=10, keep_recent=6)
    sid = "compress-4"
    _fill_turns(cm, sid, n_turns=10)
    kept = cm.get_history(sid)["messages"]
    # 最近的完整轮次原文保留
    assert kept[-1]["content"] == "reply-9"
    assert kept[-2]["content"] == "msg-9"


# ---------------------------------------------------------------------------
# 会话隔离与清理
# ---------------------------------------------------------------------------

def test_sessions_are_independent():
    cm = ContextManager()
    cm.add_user_message("s1", "窗口1的消息")
    cm.add_user_message("s2", "窗口2的消息")
    m1 = [m["content"] for m in cm.get_llm_messages("s1") if m["role"] == "user"]
    m2 = [m["content"] for m in cm.get_llm_messages("s2") if m["role"] == "user"]
    assert m1 == ["窗口1的消息"]
    assert m2 == ["窗口2的消息"]


def test_clear():
    cm = ContextManager()
    cm.add_user_message(SID, "hello")
    cm.clear(SID)
    assert cm.get_turn_count(SID) == 0
    assert cm.get_history(SID)["messages"] == []


def test_get_llm_messages_does_not_mutate_history():
    cm = ContextManager()
    cm.add_user_message(SID, "keep me intact")
    msgs = cm.get_llm_messages(SID)
    msgs.append({"role": "user", "content": "pollution"})
    assert len(cm.get_history(SID)["messages"]) == 1
