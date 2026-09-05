"""AgentRuntime 测试：多会话隔离、trace、异常兜底、上下文压缩联动。"""

from agent.llm import FakeLLM, text_response, tool_call_response


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------

def test_start_and_list_sessions(runtime):
    sid1 = runtime.start_session()
    sid2 = runtime.start_session()
    assert sid1 != sid2
    assert set(runtime.list_sessions()) >= {sid1, sid2}


def test_end_session(runtime):
    sid = runtime.start_session()
    runtime.end_session(sid)
    assert sid not in runtime.list_sessions()


def test_chat_unknown_session_returns_error(runtime):
    result = runtime.chat("ghost-session", "hello")
    assert result["status"] == "error"
    assert "不存在" in result["answer"]


# ---------------------------------------------------------------------------
# 两个窗口独立（需求场景：查天气记待办 / 写周报记待办）
# ---------------------------------------------------------------------------

def test_two_windows_independent_sessions(runtime):
    """窗口1、窗口2 各自记待办，互不可见；且窗口内可断点续聊。"""
    # 窗口 1：记待办"买牛奶"
    llm1 = FakeLLM([
        tool_call_response("todo", {"action": "add", "task": "买牛奶"}),
        text_response("已记录待办：买牛奶"),
    ])
    window1 = runtime.start_session(llm=llm1)
    r1 = runtime.chat(window1, "帮我记个待办：买牛奶")
    assert r1["status"] == "success"
    assert r1["tool_calls"][0]["status"] == "ok"

    # 窗口 2：记待办"写周报"
    llm2 = FakeLLM([
        tool_call_response("todo", {"action": "add", "task": "写周报"}),
        text_response("已记录待办：写周报"),
    ])
    window2 = runtime.start_session(llm=llm2)
    r2 = runtime.chat(window2, "帮我记个待办：写周报")
    assert r2["status"] == "success"

    # 窗口 2 查看待办：只有"写周报"，没有窗口 1 的"买牛奶"
    llm2.queue(tool_call_response("todo", {"action": "list"}))
    llm2.queue(text_response("你有一条待办：写周报"))
    r2_list = runtime.chat(window2, "我的待办有哪些")
    listed_tasks = [t["task"] for t in r2_list["tool_calls"][0]["result"]["tasks"]]
    assert listed_tasks == ["写周报"]

    # 窗口 1 查看待办：只有"买牛奶"
    llm1.queue(tool_call_response("todo", {"action": "list"}))
    llm1.queue(text_response("你有一条待办：买牛奶"))
    r1_list = runtime.chat(window1, "我的待办有哪些")
    listed_tasks_1 = [t["task"] for t in r1_list["tool_calls"][0]["result"]["tasks"]]
    assert listed_tasks_1 == ["买牛奶"]


def test_two_windows_independent_contexts(runtime):
    """两个窗口的消息历史互不污染。"""
    llm1 = FakeLLM([text_response("窗口1的回答")])
    llm2 = FakeLLM([text_response("窗口2的回答")])
    w1 = runtime.start_session(llm=llm1)
    w2 = runtime.start_session(llm=llm2)
    runtime.chat(w1, "窗口1的问题")
    runtime.chat(w2, "窗口2的问题")
    h1 = [m["content"] for m in runtime.get_history(w1)["messages"] if m["role"] == "user"]
    h2 = [m["content"] for m in runtime.get_history(w2)["messages"] if m["role"] == "user"]
    assert h1 == ["窗口1的问题"]
    assert h2 == ["窗口2的问题"]


def test_session_continuation(runtime):
    """同一窗口随时接着聊：历史完整保留。"""
    llm = FakeLLM([
        text_response("第一轮回答"),
        text_response("第二轮回答"),
    ])
    sid = runtime.start_session(llm=llm)
    runtime.chat(sid, "第一轮问题")
    result = runtime.chat(sid, "第二轮问题")
    assert result["status"] == "success"
    assert result["turn"] == 2


# ---------------------------------------------------------------------------
# Trace 与日志
# ---------------------------------------------------------------------------

def test_trace_records_full_loop(runtime):
    """一次带工具的对话，trace 覆盖全部关键事件。"""
    llm = FakeLLM([
        tool_call_response("calculator", {"expression": "10*2"}),
        text_response("结果是 20"),
    ])
    sid = runtime.start_session(llm=llm)
    runtime.chat(sid, "计算 10*2")
    events = [t["event"] for t in runtime.get_traces(sid)]
    for expected in ["user_input", "llm_request", "llm_response",
                     "tool_call", "tool_result", "final_answer"]:
        assert expected in events, f"trace 缺少事件 {expected}"
    # tool_call 事件带参数
    tool_call_event = next(t for t in runtime.get_traces(sid) if t["event"] == "tool_call")
    assert tool_call_event["data"]["name"] == "calculator"
    assert tool_call_event["data"]["args"] == {"expression": "10*2"}


def test_get_logs_from_file(runtime):
    llm = FakeLLM([text_response("好的")])
    sid = runtime.start_session(llm=llm)
    runtime.chat(sid, "记日志")
    logs = runtime.get_logs(sid)
    assert any("user_input" in line for line in logs)


# ---------------------------------------------------------------------------
# 异常兜底
# ---------------------------------------------------------------------------

def test_runtime_catches_llm_errors(runtime):
    from agent.exceptions import LLMError

    class ExplodingLLM(FakeLLM):
        def chat(self, messages, tools=None, session_id=None):
            raise LLMError("DeepSeek 服务不可用")

    sid = runtime.start_session(llm=ExplodingLLM())
    result = runtime.chat(sid, "hello")
    assert result["status"] == "error"
    assert "DeepSeek 服务不可用" in result["answer"]
    # 错误也进入 trace
    assert any(t["event"] == "error" for t in runtime.get_traces(sid))


def test_empty_input(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "")
    assert result["status"] == "success"
    assert result["iterations"] == 0


# ---------------------------------------------------------------------------
# 上下文压缩联动（长对话）
# ---------------------------------------------------------------------------

def test_long_conversation_compresses_but_continues(runtime):
    """持续对话触发压缩后仍可正常继续（记住早期状态靠摘要）。"""
    llm = FakeLLM()
    sid = runtime.start_session(llm=llm)
    runtime.context_manager.max_messages = 8
    runtime.context_manager.keep_recent = 4
    for i in range(10):
        llm.queue(text_response(f"回答{i}"))
        result = runtime.chat(sid, f"问题{i}")
        assert result["status"] == "success"

    history = runtime.get_history(sid)
    assert history["compress_count"] >= 1
    assert history["summary"] != ""
    assert history["turn_count"] == 10
    # LLM 每次请求都能看到摘要（memory 召回）
    last_messages = llm.received[-1]["messages"]
    assert any("早期对话摘要" in m["content"] for m in last_messages if m["role"] == "system")
