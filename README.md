# EL2K Mini Agent

一个**从零实现**的轻量级 Agent Runtime：不依赖 langgraph / openhands / openclaw 等 Agent 框架，核心循环（ReAct 式 Loop）、工具注册、Session 管理、Context 压缩均为手写实现。LLM 通过 **DeepSeek 真实 API**（OpenAI 兼容协议）接入，基于工具 Schema 自主决策调用工具。

## 目录

- [快速开始](#快速开始)
- [系统设计](#系统设计)
- [Agent 基本循环](#agent-基本循环)
- [工具系统](#工具系统)
- [Session 管理](#session-管理)
- [Context 与 Memory 管理](#context-与-memory-管理)
- [异常处理与 Trace](#异常处理与-trace)
- [测试](#测试)
- [AI Prompt 与问题解决记录](#ai-prompt-与问题解决记录)

## 快速开始

### 安装

```bash
pip install -r requirements.txt   # openai + pytest
```

### 配置 DeepSeek API Key

在 https://platform.deepseek.com/ 申请 Key 后：

```bash
export DEEPSEEK_API_KEY=sk-xxxxxx
# 可选配置：
# export DEEPSEEK_MODEL=deepseek-chat        # 默认 deepseek-chat
# export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 运行交互式 CLI

```bash
python main.py
```

```
==============================================================
  EL2K Mini Agent - DeepSeek 真实 API 驱动
==============================================================
当前会话: 3f1a2b4c
[Agent] > 帮我查下北京的天气，再记个待办：明天带伞
  [思考] 需要先查天气，再记录待办
  [工具] weather({"city": "北京"}) -> {"city": "北京", "temperature": 22, ...} [ok]
  [工具] todo({"action": "add", "task": "明天带伞"}) -> {"status": "added", ...} [ok]
  [回答] 北京今天 22 度、晴。已为你添加待办"明天带伞"。
  [统计] 用户轮次 1 | 本轮 LLM 迭代 3 次 | 状态 success
```

CLI 命令：`/new` 新建会话、`/switch <id>` 断点续聊、`/sessions` 列出会话、`/history` 查看上下文、`/trace` 查看执行 trace、`/quit` 退出。

### 作为库使用

```python
from agent import AgentRuntime, DeepSeekLLM

runtime = AgentRuntime(llm=DeepSeekLLM())

# 用户 A 的两个独立窗口
window1 = runtime.start_session()
window2 = runtime.start_session()

runtime.chat(window1, "查一下北京的天气，并记待办：明天带伞")
runtime.chat(window2, "帮我写周报开头，并记待办：周五前提交")

# 两个窗口的 context 与 todo 数据完全隔离，可随时接着聊
runtime.chat(window1, "我的待办有哪些")   # 只会看到"明天带伞"
```

## 系统设计

```
agent/
├── llm.py             # LLM 接入层
│   ├── DeepSeekLLM        # 真实 API（OpenAI 兼容）+ function calling + 指数退避重试
│   ├── parse_text_response# 文本协议解析（thought / tool_calls / answer）
│   └── FakeLLM            # 可脚本化假 LLM（离线确定性测试）
├── session.py         # Session + SessionManager + 核心 Agent Loop
├── context.py         # ContextManager：消息历史、轮次计数、压缩与摘要
├── tool_registry.py   # Tool（name+description+schema+func）与注册/执行
├── tools/             # calculator / search(mock) / weather(mock) / todo
├── logger.py          # TraceLogger：结构化 trace（内存）+ 文本日志（文件）
├── runtime.py         # AgentRuntime 门面：会话生命周期 + chat + 观测
├── exceptions.py      # 异常体系
└── utils.py           # 截断 / 序列化小工具

main.py                # 交互式 CLI 入口
tests/                 # 单元测试 + FakeLLM 集成测试 + 真实 API 集成测试
```

分层职责：

| 层 | 模块 | 职责 |
|---|---|---|
| 接入层 | `llm.py` | 调用 DeepSeek API，解析出统一的 `{thought, tool_calls, answer}` |
| 决策层 | `session.py` | Agent Loop：决定何时调工具、何时返回答案 |
| 状态层 | `context.py` | 每个会话独立的消息历史、压缩、摘要 |
| 能力层 | `tool_registry.py` + `tools/` | 工具注册、Schema 导出、参数校验与执行 |
| 门面层 | `runtime.py` | 组装一切，对外提供 `chat()` 与观测接口 |

## Agent 基本循环

`agent/session.py` 的 `Session.run()` 实现了标准的四步循环（对应作业要求）：

```
Step 1  接收用户输入，写入本 session 的 context
Step 2  组装 context（系统提示词 + 摘要 + 历史）与全部工具 Schema 调用 LLM，
        由 LLM 自主决策：直接回复 or 调用工具
Step 3  若请求工具：逐个执行（结果与错误都写回 context）
Step 4  根据工具结果判断：
        - LLM 下一轮仍请求工具  -> 继续 loop（回到 Step 2，携带工具结果）
        - LLM 不再请求工具      -> 返回最终答案给用户
```

安全阀：`max_iterations`（默认 8）限制单次请求内的最大迭代次数；达到上限后追加一次性"禁止再调工具"的提示，强制 LLM 基于已有信息收尾作答，防止死循环。

LLM 输出的解析逻辑（`llm.py`）：

- **主通道**：DeepSeek 原生 function calling —— 从 `message.tool_calls` 提取 `id/name/args`，`message.content` 作为思考过程；
- **兼容通道**：模型未走 function calling 时，对纯文本做协议解析，支持整体 JSON（`{"thought":..., "tool_calls":[...], "answer":...}`）、XML 风格标签（`<thought>/<invoke name="...">/<answer>`）、纯文本兜底（整段视为答案）三种形态。

## 工具系统

每个工具是一个 `Tool` 实例，四要素：**名称、描述、参数 JSON Schema、实现函数**。

- `registry.get_function_schemas()` 导出 OpenAI function-calling 格式的 Schema 列表，随每次 LLM 请求下发——LLM 完全基于 Schema 自主决策是否调用、如何传参；
- `registry.execute(name, args, session_id)` 执行前做轻量 required 参数校验；工具内部异常统一包装为 `ToolExecutionError`；
- **session_id 注入机制**：工具函数若声明了 `session_id` 形参，注册表执行时自动注入当前会话 ID。todo 工具借此实现"按会话隔离存储"，保证多窗口待办互不影响。

内置 4 个工具：

| 工具 | 说明 | 数据源 |
|---|---|---|
| `calculator` | 白名单算术表达式求值（禁用内建，防注入） | 本地计算 |
| `search` | 网络搜索 | mock |
| `weather` | 城市天气 | mock |
| `todo` | 待办 add/list/remove/clear | 内存，按 session 隔离 |

## Session 管理

- 每次调用 `runtime.start_session()` 创建一个独立会话窗口（8 位 ID）；
- **ContextManager 按 session_id 隔离存储**消息历史：窗口 1 与窗口 2 的对话、待办完全独立；
- 会话常驻内存，用户可随时 `/switch <id>` 回到任意窗口**断点续聊**；
- `delete_session` 同时清理会话与上下文。

需求场景验证（见 `tests/test_runtime.py::test_two_windows_independent_sessions`）：窗口 1 查天气记待办、窗口 2 写周报记待办——两边各自的 todo 列表互相不可见。

## Context 与 Memory 管理

### 哪些信息进入 context

| 信息 | 是否保留 | 原因 |
|---|---|---|
| 用户输入 | 保留 | 追问的理解基础 |
| assistant 的工具调用请求 | 保留（含 tool_calls 结构） | 记录"做过什么"，OpenAI 协议要求与 tool 消息配对 |
| 工具执行结果 | 保留（单条超 1200 字符截断） | 后续轮次引用工具结论的依据 |
| 最终答案 | 保留 | 纯对话追问的基础 |
| 思考过程（thought） | 仅在伴随工具调用时保留 | 帮助后续轮次理解"当时为什么调工具"；纯文本回答的思考不单独入库，避免膨胀 |

### 最大轮次限制（两层）

1. **单次请求内的循环上限**：`max_iterations=8`，超限强制收尾（防死循环）；
2. **会话历史的消息上限**：`max_messages=40`，超限触发压缩。

### 压缩（基础版）与 Memory 的召回时机、放置方式

```
消息数 > max_messages 时触发压缩：

  按"用户轮次"整轮切割历史（绝不从中间截断，
  保证 assistant(tool_calls) 与 tool 消息对完整——拆散会导致 API 报错）
          │
          ├── 旧轮次  -> 折叠为文字摘要（每条截断 160 字符，总长上限 1500）
          └── 近期轮次（约 keep_recent=16 条）-> 原文保留

发给 LLM 的最终 context：
  [system] 系统提示词
  [system] [早期对话摘要（自动压缩生成）] ...   <- memory 的载体
  [user/assistant/tool ...] 近期消息原文
```

**召回时机**：无需显式检索——每次调用 LLM 前（`ContextManager.get_llm_messages`）组装 context 时，摘要**始终随请求自动携带**。这是最简单可靠的"always-in-context"召回策略，保证任何一轮对话都能引用最早期的状态。

**放置方式**：摘要以一条 `role=system` 的消息放在系统提示词之后、近期历史之前。选 system 区而不是混入 user 消息，是为了与真实用户输入明确区分，并利用 system 消息的高权重让模型优先采信。

**压缩后的连续性**：多次压缩时新摘要与旧摘要用 `|` 拼接、超长保尾部（最近的历史优先），持续对话中模型始终能感知完整会话脉络（验证：`tests/test_runtime.py::test_long_conversation_compresses_but_continues`）。

## 异常处理与 Trace

### 异常体系（`exceptions.py`）

| 异常 | 触发场景 | 处理方式 |
|---|---|---|
| `LLMError` | Key 缺失 / API 调用失败（重试 2 次后） | 向上传播，runtime 兜底转为 `status=error` |
| `ParseError` | 工具参数 JSON 非法 | 不重试，直接抛出 |
| `ToolNotFoundError` | LLM 请求了不存在的工具 | 错误写回 context，LLM 看到后自行调整 |
| `ToolExecutionError` | 缺必填参数 / 工具内部异常 | 同上，循环不崩溃 |
| `SessionError` | 访问不存在的会话 | runtime 返回结构化错误 |

关键设计：**工具失败不让循环崩溃**——错误以工具结果的形式写回 context，下一轮 LLM 能看到错误并自行决定换参数重试还是直接回答（验证：`tests/test_session.py::test_unknown_tool_error_fed_back_to_llm`）。`runtime.chat()` 捕获一切异常返回 `status=error` 的结构化结果，CLI/API 调用方永不崩溃。

### Trace 与执行日志（`logger.py`）

双路输出：

1. **结构化 trace**（内存，按 session 隔离）：事件包括 `session_created / user_input / llm_request / llm_response / tool_call / tool_result / final_answer / error`，可通过 `runtime.get_traces(session_id)` 程序化查询（测试断言即基于此）；
2. **文本日志**（`logs/agent_YYYY-MM-DD.log`）：每条 trace 同步落盘一行，供人工排查。

## 测试

```bash
# 离线全量测试（FakeLLM 驱动，无需 API Key，85 个用例）
pytest tests/ -v

# 仅排除真实 API 测试
pytest tests/ -v -m "not llm"

# 真实 DeepSeek API 集成测试（需要 Key，7 个用例）
export DEEPSEEK_API_KEY=sk-xxx
pytest tests/test_integration_deepseek.py -v -m llm
```

覆盖矩阵：

| 测试文件 | 覆盖点 |
|---|---|
| `test_tools.py` | 4 个工具单元测试：正常路径、错误路径、todo 会话隔离、calculator 防注入 |
| `test_tool_registry.py` | 注册机制（装饰器/实例）、Schema 导出格式、required 校验、session_id 注入、异常包装 |
| `test_context.py` | OpenAI 消息格式、tool_call_id 配对、轮次计数、压缩触发、摘要注入、消息对完整性、会话隔离 |
| `test_llm.py` | 文本协议解析（JSON/XML/兜底/多工具/容错）、FakeLLM、DeepSeekLLM stub 测试（原生 tool_calls 解析、重试、Key 校验） |
| `test_config.py` | 配置集中管理：.env 加载与解析、环境变量优先于 .env、refresh 读取、describe 脱敏 |
| `test_session.py` | **核心循环**：直接回复、工具结果回传下一轮请求、多工具并发、跨迭代链式调用、纯对话追问、带工具追问、最大迭代强制收尾、错误恢复 |
| `test_runtime.py` | 双窗口独立（todo + context）、断点续聊、trace 完整性、runtime 异常兜底、长对话压缩联动 |
| `test_integration_deepseek.py` | **真实 API**：直接回复、calculator/weather/todo 工具调用、带工具追问、双会话待办隔离、trace 完整性 |

FakeLLM 的设计要点：**脚本化响应队列 + 完整记录每次收到的请求**，使得"工具结果是否进入了下一轮 LLM 请求的 context"这类核心断言可以确定性验证，而不依赖真实模型的行为波动。

## AI Prompt 与问题解决记录

本项目在 AI 辅助下开发，以下是关键 Prompt 摘要与开发过程中实际遇到并解决的问题记录。

### 关键 AI Prompt（摘要）

1. **重构指令**：「这个项目目前有些业务没有正确实现……LLM 是自己构造的一个，应该使用 api 调用的，目前我打算使用 deepseek 的 api，然后 loop 的判断逻辑也没有实现，也没有 schema 的结构等等，帮我重新修改完善这个项目」——附完整的作业要求（从零实现 / 基本循环 / 工具 / session 管理 / context 管理 / 测试用例）。
2. **架构决策追问**：「消息格式不符合 OpenAI/DeepSeek API 规范（tool 消息缺 tool_call_id、assistant 的 thought/answer 被拆成两条消息），真实 API 会直接报错」——要求 AI 先诊断旧实现的缺陷清单再动手。
3. **测试策略**：「用可脚本化的 FakeLLM 做确定性测试，真实 API 测试单独打 marker，没有 Key 时自动跳过」。

### 问题解决记录

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 1 | 旧实现的"工具调用链路"实际从未走通 | MockLLM 的预设响应只含 thought/answer，从不产生 tool_calls | 引入 DeepSeekLLM 真实 API + 原生 function calling；测试改用可脚本化 FakeLLM |
| 2 | Loop 是"一次工具 + 一次追问"的伪循环 | `Session.run()` 执行完工具后固定只再调一次 LLM，没有"根据工具结果判断继续/返回"的迭代结构 | 重写为 `for step in range(max_iterations)` 的真循环：LLM 仍请求工具就继续，否则返回答案；超限强制收尾 |
| 3 | 消息格式与 OpenAI/DeepSeek 协议不符 | tool 消息缺 `tool_call_id`；assistant 的 thought 和 answer 被拆成两条独立消息；tool_calls 未按协议序列化 | ContextManager 严格按 OpenAI 规范组装：`assistant.tool_calls[]` + `tool.tool_call_id` 配对 |
| 4 | 压缩可能把 assistant(tool_calls) 与 tool 消息拆散 | 旧压缩按消息条数对半切，不看消息对边界 | 改为按"用户轮次"整轮切割，消息对永远完整 |
| 5 | 两个窗口的待办互相污染 | todo 数据持久化在全局共享的 `.todo_data.json` | 内存存储按 session_id 键控；ToolRegistry 支持向声明了 `session_id` 形参的工具自动注入会话 ID |
| 6 | 测试硬编码 Windows 路径且调用不存在的方法 | 旧测试 `sys.path.insert(0, r'D:\Desktop\EL2K')`，`session_manager.chat()` 从未实现 | `conftest.py` 用项目相对路径注入 sys.path；补齐 `SessionManager.chat()`；全部测试重写 |
| 7 | 工具失败会中断整轮对话 | 旧实现工具异常仅记日志，LLM 感知不到 | 错误以工具结果形式写回 context，模型下一轮可见并可自行恢复 |
| 8 | 开发环境中文 locale 导致源文件编码混乱 | 编辑工具在 zh_CN locale 下将中文内容写成 GBK | 落盘前统一转 UTF-8（这也解释了为何个别提交需要编码修复脚本） |

### 遗留说明

- todo 为内存存储，进程重启后清空（demo 场景下可接受；持久化只需在 `tools/todo.py` 换成带 session 维度的文件/DB 存储）；
- 压缩为"截断拼接"的基础实现（作业明确不要求复杂压缩），生产可替换为 LLM 摘要或向量检索。
