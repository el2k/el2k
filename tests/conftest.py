"""pytest 公共配置：项目根目录入 sys.path + 公共 fixtures。"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.llm import FakeLLM  # noqa: E402
from agent.runtime import AgentRuntime  # noqa: E402
from agent.session import SessionManager  # noqa: E402


@pytest.fixture
def fake_llm():
    """空的 FakeLLM，测试中自行 queue 响应。"""
    return FakeLLM()


@pytest.fixture
def runtime():
    """使用 FakeLLM 的 AgentRuntime（离线可跑）。"""
    return AgentRuntime(llm=FakeLLM())


@pytest.fixture
def session_manager():
    """使用 FakeLLM 的 SessionManager（离线可跑）。"""
    return SessionManager(llm=FakeLLM())
