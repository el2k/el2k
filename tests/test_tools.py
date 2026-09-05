"""内置工具单元测试：calculator / search / weather / todo。"""

import uuid

from agent.tools.calculator import calculator
from agent.tools.search import search
from agent.tools.todo import todo
from agent.tools.weather import weather


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

def test_calculator_basic_arithmetic():
    assert calculator("2 + 3 * 4")["result"] == 14
    assert calculator("(1 + 2) * 5")["result"] == 15
    assert calculator("10 / 4")["result"] == 2.5


def test_calculator_functions():
    assert calculator("sqrt(16)")["result"] == 4.0
    assert calculator("2 ** 10")["result"] == 1024
    assert calculator("abs(-7)")["result"] == 7


def test_calculator_error_returns_error_not_raises():
    result = calculator("1 / 0")
    assert "error" in result


def test_calculator_rejects_injection():
    # __builtins__ 已清空，仅白名单函数可用
    result = calculator("__import__('os').system('ls')")
    assert "error" in result


# ---------------------------------------------------------------------------
# search (mock)
# ---------------------------------------------------------------------------

def test_search_default_results():
    result = search("python")
    assert result["query"] == "python"
    assert result["total"] == 5
    assert len(result["results"]) == 5
    assert result["source"] == "mock"


def test_search_num_results_clamped():
    assert len(search("x", num_results=2)["results"]) == 2
    assert len(search("x", num_results=99)["results"]) == 5  # 上限 5


# ---------------------------------------------------------------------------
# weather (mock)
# ---------------------------------------------------------------------------

def test_weather_known_city():
    result = weather("北京")
    assert result["city"] == "北京"
    assert result["temperature"] == 22
    assert result["condition"] == "晴"
    assert result["source"] == "mock"


def test_weather_english_and_case_insensitive():
    assert weather("Shanghai")["temperature"] == 28
    assert weather("beijing")["temperature"] == 22


def test_weather_unknown_city_random():
    result = weather("Atlantis")
    assert result["source"] == "mock"
    assert -10 <= result["temperature"] <= 40
    assert "condition" in result and "humidity" in result


# ---------------------------------------------------------------------------
# todo（按 session 隔离）
# ---------------------------------------------------------------------------

def _sid():
    return f"test-{uuid.uuid4().hex[:8]}"


def test_todo_add_and_list():
    sid = _sid()
    assert todo("add", task="买牛奶", session_id=sid)["status"] == "added"
    assert todo("add", task="写周报", session_id=sid)["status"] == "added"
    listed = todo("list", session_id=sid)
    assert listed["total"] == 2
    assert [t["task"] for t in listed["tasks"]] == ["买牛奶", "写周报"]


def test_todo_remove_and_clear():
    sid = _sid()
    todo("add", task="task0", session_id=sid)
    todo("add", task="task1", session_id=sid)
    removed = todo("remove", index=0, session_id=sid)
    assert removed["status"] == "removed"
    assert removed["task"] == "task0"
    assert todo("list", session_id=sid)["total"] == 1
    assert todo("clear", session_id=sid)["status"] == "cleared"
    assert todo("list", session_id=sid)["total"] == 0


def test_todo_sessions_are_isolated():
    """核心需求：窗口 1 / 窗口 2 的待办互不影响。"""
    sid1, sid2 = _sid(), _sid()
    todo("add", task="窗口1的待办", session_id=sid1)
    todo("add", task="窗口2的待办", session_id=sid2)
    tasks1 = [t["task"] for t in todo("list", session_id=sid1)["tasks"]]
    tasks2 = [t["task"] for t in todo("list", session_id=sid2)["tasks"]]
    assert tasks1 == ["窗口1的待办"]
    assert tasks2 == ["窗口2的待办"]


def test_todo_errors():
    sid = _sid()
    assert "error" in todo("add", session_id=sid)              # 缺 task
    assert "error" in todo("remove", index=0, session_id=sid)  # 空列表删除
    assert "error" in todo("jump", session_id=sid)             # 未知 action
