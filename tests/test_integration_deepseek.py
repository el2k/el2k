"""真实 DeepSeek API 集成测试。

运行条件：配置 DEEPSEEK_API_KEY（环境变量，或项目根目录 .env 文件）
    export DEEPSEEK_API_KEY=sk-xxx
    pytest tests/test_integration_deepseek.py -v -m llm

未配置 Key 时自动跳过（离线环境跑其余测试不受影响）。
"""

import pytest

from agent import AgentRuntime, DeepSeekLLM, config

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not config.DEEPSEEK_API_KEY,
        reason="需要配置 DEEPSEEK_API_KEY（环境变量或 .env 文件）才能运行真实 API 集成测试",
    ),
]


@pytest.fixture(scope="module")
def runtime():
    return AgentRuntime(llm=DeepSeekLLM())


def _find_tool_call(result, name):
    return next((tc for tc in result["tool_calls"] if tc["name"] == name), None)


# ---------------------------------------------------------------------------
# 直接回复（不调用工具）
# ---------------------------------------------------------------------------

def test_direct_reply_without_tools(runtime):
    result = runtime.chat(runtime.start_session(), "用一句话介绍你自己，不要调用任何工具。")
    assert result["status"] == "success"
    assert result["answer"].strip()
    assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# 工具调用：calculator
# ---------------------------------------------------------------------------

def test_calculator_tool_via_real_llm(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "请用计算器精确计算 (123 * 456) + 789 等于多少，直接给出数值。")
    assert result["status"] == "success"
    call = _find_tool_call(result, "calculator")
    assert call is not None, f"期望调用 calculator，实际: {[t['name'] for t in result['tool_calls']]}"
    assert call["result"]["result"] == 56967
    assert "56967" in result["answer"]


# ---------------------------------------------------------------------------
# 工具调用：weather + todo（需求场景）
# ---------------------------------------------------------------------------

def test_weather_tool_via_real_llm(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "查一下上海现在的天气怎么样？温度是多少？")
    assert result["status"] == "success"
    call = _find_tool_call(result, "weather")
    assert call is not None
    assert call["result"]["city"] in ("上海", "Shanghai")


def test_todo_tool_via_real_llm(runtime):
    sid = runtime.start_session()
    result = runtime.chat(sid, "帮我在待办里加一条：明天上午review PR。")
    assert result["status"] == "success"
    call = _find_tool_call(result, "todo")
    assert call is not None
    assert call["status"] == "ok"


# ---------------------------------------------------------------------------
# 多轮：纯对话追问 / 带工具追问
# ---------------------------------------------------------------------------

def test_conversational_followup_with_tool(runtime):
    """先给模型一个数字，再追问并要求用计算器计算（带工具的追问）。"""
    sid = runtime.start_session()
    r1 = runtime.chat(sid, "请记住这个幸运数字：21。简单回复收到即可。")
    assert r1["status"] == "success"
    r2 = runtime.chat(sid, "把我刚才说的幸运数字乘以 2，用计算器算出精确结果。")
    assert r2["status"] == "success"
    call = _find_tool_call(r2, "calculator")
    assert call is not None, "追问轮应触发 calculator 工具调用"
    assert call["result"]["result"] == 42
    assert "42" in r2["answer"]


# ---------------------------------------------------------------------------
# Session 隔离（真实 API 场景）
# ---------------------------------------------------------------------------

def test_two_sessions_independent_todo(runtime):
    """窗口 1、窗口 2 各记各的待办，互不可见。"""
    w1 = runtime.start_session()
    w2 = runtime.start_session()

    r1 = runtime.chat(w1, "帮我加一条待办：窗口一的任务A，加完简短确认。")
    r2 = runtime.chat(w2, "帮我加一条待办：窗口二的任务B，加完简短确认。")
    assert r1["status"] == "success" and r2["status"] == "success"

    r1_list = runtime.chat(w1, "列出我当前的全部待办。")
    assert r1_list["status"] == "success"
    listed = []
    for tc in r1_list["tool_calls"]:
        if tc["name"] == "todo" and isinstance(tc["result"].get("tasks"), list):
            listed = [t["task"] for t in tc["result"]["tasks"]]
    assert listed == ["窗口一的任务A"], f"窗口1应只见任务A，实际: {listed}"


# ---------------------------------------------------------------------------
# Trace 完整性（真实链路）
# ---------------------------------------------------------------------------

def test_trace_complete_on_real_llm(runtime):
    sid = runtime.start_session()
    runtime.chat(sid, "计算 sqrt(144) 加 8 等于多少？")
    events = [t["event"] for t in runtime.get_traces(sid)]
    assert "user_input" in events and "final_answer" in events
