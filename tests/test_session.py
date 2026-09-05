"""核心 Agent Loop 测试（FakeLLM 驱动，确定性验证 Step 1-4）。"""

import pytest

from agent.exceptions import LLMError, ToolNotFoundError
from agent.llm import FakeLLM, text_response, tool_call_response


@pytest.fixture
def sm():
    from agent.session import SessionManager
    return SessionManager(llm=FakeLLM(), max_iterations=5)


# ---------------------------------------------------------------------------
# Step 2 判断：直接回复 vs 调用工具
# ---------------------------------------------------------------------------

def test_direct_answer_without_tools(sm):
    """LLM 直接回答：一次迭代、零工具调用。"""
    llm = FakeLLM([text_response("你好！有什么可以帮你？")])
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "你好")
    assert result["answer"] == "你好！有什么可以帮你？"
    assert result["tool_calls"] == []
    assert result["iterations"] == 1
    assert result["status"] == "success"


def test_tools_schema_passed_to_llm(sm):
    """LLM 请求携带全部工具 Schema，由其自主决策。"""
    llm = FakeLLM([text_response("好的")])
    sid = sm.create_session(llm=llm)
    sm.chat(sid, "hi")
    tools = llm.received[0]["tools"]
    names = {t["function"]["name"] for t in tools}
    assert names == {"calculator", "search", "todo", "weather"}


# ---------------------------------------------------------------------------
# Step 3/4：工具执行与循环继续
# ---------------------------------------------------------------------------

def test_tool_result_fed_back_into_next_llm_request(sm):
    """工具结果写入 context，并出现在下一次 LLM 请求中（Step 4 继续循环）。"""
    llm = FakeLLM([
        tool_call_response("calculator", {"expression": "2+3"}, thought="需要计算"),
        text_response("2+3 等于 5"),
    ])
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "计算 2+3")

    assert result["answer"] == "2+3 等于 5"
    assert result["iterations"] == 2
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["result"]["result"] == 5
    assert result["tool_calls"][0]["status"] == "ok"

    # 第二次 LLM 请求中应包含 tool 消息，且内容是计算结果
    second_messages = llm.received[1]["messages"]
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "5" in tool_msgs[0]["content"]
    # tool 消息与 assistant.tool_calls 的 id 配对
    call_id = tool_msgs[0]["tool_call_id"]
    assistant_msgs = [m for m in second_messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert any(tc["id"] == call_id for tc in assistant_msgs[0]["tool_calls"])


def test_multiple_tool_calls_in_one_response(sm):
    """一次响应中请求多个工具：全部执行、全部写回。"""
    llm = FakeLLM([
        {
            "thought": "同时查两个城市",
            "tool_calls": [
                {"id": "c1", "name": "weather", "args": {"city": "北京"}},
                {"id": "c2", "name": "weather", "args": {"city": "上海"}},
            ],
            "answer": "",
        },
        text_response("北京 22 度晴，上海 28 度小雨"),
    ])
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "北京和上海的天气")
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["result"]["temperature"] == 22
    assert result["tool_calls"][1]["result"]["temperature"] == 28
    # 两个 tool 消息都进入第二轮请求
    tool_msgs = [m for m in llm.received[1]["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}


def test_chained_tool_calls_across_iterations(sm):
    """跨迭代的连续工具调用（第 2 轮继续要工具）。"""
    llm = FakeLLM([
        tool_call_response("search", {"query": "agent 是什么"}),
        tool_call_response("weather", {"city": "北京"}),
        text_response("综合结果：今天适合研究 agent"),
    ])
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "搜一下 agent 并查北京天气")
    assert result["iterations"] == 3
    assert [tc["name"] for tc in result["tool_calls"]] == ["search", "weather"]
    # 第三轮请求包含前两轮的全部 tool 消息
    tool_msgs = [m for m in llm.received[2]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2


# ---------------------------------------------------------------------------
# 追问（context 记住之前状态）
# ---------------------------------------------------------------------------

def test_pure_conversational_followup(sm):
    """纯对话追问：上一轮的用户输入与回答都在第二轮请求的 context 中。"""
    llm = FakeLLM([
        text_response("珠穆朗玛峰高 8848.86 米"),
        text_response("换算后约 29029 英尺"),
    ])
    sid = sm.create_session(llm=llm)
    sm.chat(sid, "珠峰多高？")
    result = sm.chat(sid, "换算成英尺是多少？")
    assert result["answer"] == "换算后约 29029 英尺"

    second_messages = llm.received[1]["messages"]
    user_contents = [m["content"] for m in second_messages if m["role"] == "user"]
    assert "珠峰多高？" in user_contents
    assistant_contents = [m["content"] for m in second_messages if m["role"] == "assistant"]
    assert any("8848.86" in c for c in assistant_contents)


def test_followup_with_tools(sm):
    """带工具的追问：新问题触发新的工具调用，且能引用旧上下文。"""
    llm = FakeLLM([
        text_response("好的，已了解你的需求"),
        tool_call_response("calculator", {"expression": "8848.86 * 3.28084"}),
        text_response("8848.86 米约等于 29029 英尺"),
    ])
    sid = sm.create_session(llm=llm)
    sm.chat(sid, "珠峰高 8848.86 米，记住这个数")
    result = sm.chat(sid, "帮我精确换算成英尺")
    assert result["tool_calls"][0]["name"] == "calculator"
    assert result["iterations"] == 2
    # 追问轮的请求包含第一轮的完整历史
    followup_messages = llm.received[1]["messages"]
    assert any(m.get("content") == "珠峰高 8848.86 米，记住这个数"
               for m in followup_messages if m["role"] == "user")


# ---------------------------------------------------------------------------
# 安全阀：最大迭代限制
# ---------------------------------------------------------------------------

def test_max_iterations_forces_final_answer():
    from agent.session import SessionManager
    sm = SessionManager(llm=FakeLLM(), max_iterations=3)
    llm = FakeLLM([
        tool_call_response("calculator", {"expression": "1+1"}),
        tool_call_response("calculator", {"expression": "2+2"}),
        tool_call_response("calculator", {"expression": "3+3"}),
        text_response("基于已完成的计算，最终结论是 6"),  # 强制收尾的响应
    ])
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "没完没了地算")
    assert result["iterations"] == 3
    assert result["answer"] == "基于已完成的计算，最终结论是 6"
    assert len(result["tool_calls"]) == 3
    # 强制收尾请求不应携带工具
    assert llm.received[3]["tools"] is None


# ---------------------------------------------------------------------------
# 异常处理
# ---------------------------------------------------------------------------

def test_unknown_tool_error_fed_back_to_llm(sm):
    """工具不存在：错误写回 context，循环不崩溃，LLM 看到错误后继续作答。"""
    llm = FakeLLM([
        tool_call_response("nonexistent_tool", {"x": 1}),
        text_response("该工具不可用，我直接回答"),
    ])
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "试试那个不存在的工具")
    assert result["status"] == "success"
    assert result["tool_calls"][0]["status"] == "error"
    tool_msgs = [m for m in llm.received[1]["messages"] if m["role"] == "tool"]
    assert "nonexistent_tool" in tool_msgs[0]["content"]


def test_bad_tool_args_error_recovery(sm):
    """LLM 传参错误：工具返回错误信息，模型仍可完成回答。"""
    llm = FakeLLM([
        tool_call_response("calculator", {}),  # 缺 expression
        text_response("参数不对，我换个方式回答"),
    ])
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "算个数")
    assert result["tool_calls"][0]["status"] == "error"
    assert "缺少必填参数" in result["tool_calls"][0]["result"]["error"]
    assert result["answer"] == "参数不对，我换个方式回答"


def test_llm_error_propagates(sm):
    """LLM 层错误向上传播（由 runtime 统一兜底）。"""
    class ExplodingLLM(FakeLLM):
        def chat(self, messages, tools=None, session_id=None):
            raise LLMError("API 挂了")

    sid = sm.create_session(llm=ExplodingLLM())
    with pytest.raises(LLMError):
        sm.chat(sid, "hi")


def test_empty_input_short_circuits(sm):
    llm = FakeLLM()
    sid = sm.create_session(llm=llm)
    result = sm.chat(sid, "   ")
    assert result["status"] == "success"
    assert result["iterations"] == 0
    assert llm.received == []  # 不应调用 LLM


# ---------------------------------------------------------------------------
# Session 管理
# ---------------------------------------------------------------------------

def test_session_not_found_raises(sm):
    from agent.exceptions import SessionError
    with pytest.raises(SessionError):
        sm.get_session("no-such-session")


def test_session_ids_unique(sm):
    assert sm.create_session() != sm.create_session()


def test_session_delete_clears_context(sm):
    sid = sm.create_session()
    sm.context_manager.add_user_message(sid, "hello")
    sm.delete_session(sid)
    assert not sm.has_session(sid)
    assert sm.context_manager.get_turn_count(sid) == 0
