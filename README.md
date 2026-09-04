# Minimal Agent Framework

一个从零实现的轻量级 Agent 运行时框架。

## 架构

```
agent/
├── __init__.py        # 包入口
├── exceptions.py      # 自定义异常
├── logger.py          # TraceLogger 追踪日志
├── tool_registry.py   # 工具注册机制
├── tools/             # 内置工具
│   ├── calculator.py  # 计算器
│   ├── search.py      # 搜索 (mock)
│   ├── todo.py        # 待办事项管理
│   └── weather.py     # 天气查询 (mock)
├── llm.py             # LLM 接口 + MockLLM
├── context.py         # 上下文管理（轮次限制 + 压缩）
├── session.py         # Session 管理（独立会话）
└── runtime.py         # AgentRuntime 核心循环
```

## 快速开始

```python
from agent import AgentRuntime

runtime = AgentRuntime()

# 创建独立会话
session1 = runtime.start_session()
session2 = runtime.start_session()

# 在不同会话中交互
runtime.chat(session1, "计算 2+3")
runtime.chat(session2, "查一下北京的天气")

# 独立上下文，互不影响
runtime.chat(session1, "帮我加个待办：买牛奶")
runtime.chat(session2, "我的待办是什么")  # 空
```

## 测试

```bash
pip install openai pytest
pytest tests/ -v
```

## 功能特性

- **从零实现**：不依赖 langgraph/openhands/openclaw
- **工具注册机制**：名称、描述、参数 Schema，LLM 自主决策调用
- **会话管理**：每个会话独立上下文，支持多窗口并行
- **上下文管理**：最大轮次限制 + 自动压缩
- **Trace 日志**：完整的工具调用链和执行日志
- **异常处理**：完善的错误处理和恢复机制