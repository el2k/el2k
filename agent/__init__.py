"""EL2K Mini Agent：一个从零实现的轻量级 Agent Runtime。

不依赖任何 Agent 框架（langgraph / openhands / ...），核心循环、工具注册、
会话与上下文管理均为手写实现。LLM 通过 DeepSeek（OpenAI 兼容）真实 API 接入。
"""

from .context import ContextManager
from .exceptions import (
    AgentError,
    ContextOverflowError,
    LLMError,
    ParseError,
    SessionError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .llm import (
    DeepSeekLLM,
    FakeLLM,
    parse_text_response,
    text_response,
    tool_call_response,
)
from .logger import TraceLogger
from .runtime import AgentRuntime
from .session import Session, SessionManager
from .tool_registry import Tool, ToolRegistry
from .tools import build_default_registry

__all__ = [
    "AgentRuntime",
    "Session",
    "SessionManager",
    "ContextManager",
    "Tool",
    "ToolRegistry",
    "build_default_registry",
    "DeepSeekLLM",
    "FakeLLM",
    "parse_text_response",
    "text_response",
    "tool_call_response",
    "TraceLogger",
    "AgentError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "LLMError",
    "SessionError",
    "ContextOverflowError",
    "ParseError",
]
