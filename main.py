"""交互式 CLI 入口。

用法：
    export DEEPSEEK_API_KEY=sk-xxx     # 必填，https://platform.deepseek.com/ 获取
                                       # 或：cp .env.example .env 并填入 Key
    python main.py

会话命令：
    /new            新建会话并切换过去
    /switch <id>    切换到指定会话（断点续聊）
    /sessions       列出全部会话
    /history        查看当前会话的上下文（消息历史/摘要/轮次）
    /trace          查看当前会话最近的结构化执行 trace
    /help           帮助
    /quit           退出
其他输入会作为用户消息进入当前会话的 Agent 循环。
"""

import sys

from agent import AgentRuntime, DeepSeekLLM, LLMError, config
from agent.utils import to_display_text, truncate

HELP = """命令列表：
  /new            新建会话并切换
  /switch <id>    切换会话（断点续聊）
  /sessions       列出全部会话
  /history        查看当前会话上下文
  /trace          查看当前会话执行 trace
  /help           帮助
  /quit           退出"""


def _print_chat_result(result):
    if result.get("thought"):
        print(f"  [思考] {result['thought']}")
    for tc in result.get("tool_calls", []):
        print(f"  [工具] {tc['name']}({to_display_text(tc['args'])}) "
              f"-> {truncate(to_display_text(tc['result']), 200)} [{tc['status']}]")
    print(f"  [回答] {result['answer']}")
    print(f"  [统计] 用户轮次 {result.get('turn', 0)} | "
          f"本轮 LLM 迭代 {result.get('iterations', 0)} 次 | "
          f"状态 {result.get('status')}")


def _print_history(runtime, session_id):
    history = runtime.get_history(session_id)
    print(f"用户轮次: {history['turn_count']} | 压缩次数: {history['compress_count']}")
    if history["summary"]:
        print(f"[早期对话摘要] {truncate(history['summary'], 300)}")
    for msg in history["messages"]:
        role = msg["role"]
        if role == "assistant" and msg.get("tool_calls"):
            names = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
            print(f"  [{role}] (请求调用工具: {names}) {msg.get('content') or ''}")
        elif role == "tool":
            print(f"  [tool#{msg['tool_call_id']}] {truncate(msg['content'], 120)}")
        else:
            print(f"  [{role}] {truncate(msg.get('content') or '', 200)}")

# 打印最近 trace 入口, trace 指 Agent 执行过程中记录的事件日志
def _print_trace(runtime, session_id, limit=15):
    traces = runtime.get_traces(session_id)[-limit:]
    if not traces:
        print("（暂无 trace）")
        return
    for entry in traces:
        print(f"  {entry['time']} {entry['event']:<14} {entry['data']}")


def main():
    try:
        llm = DeepSeekLLM()
    except LLMError as e:
        print(f"[启动失败] {e}")
        print("获取 API Key：https://platform.deepseek.com/")
        sys.exit(1)

    runtime = AgentRuntime(llm=llm)
    current = runtime.start_session()

    print("=" * 62)
    print("  EL2K Mini Agent - DeepSeek 真实 API 驱动")
    print(f"  [配置] {config.describe()}")
    print("=" * 62)
    print(f"当前会话: {current}")
    print(HELP)

    while True:
        try:
            user_input = input("\n[Agent] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("再见！")
            break
        if user_input == "/help":
            print(HELP)
            continue
        if user_input == "/new":
            current = runtime.start_session()
            print(f"已新建并切换到会话: {current}")
            continue
        if user_input == "/sessions":
            sessions = runtime.list_sessions()
            print(f"共 {len(sessions)} 个会话:")
            for sid in sessions:
                marker = "  <-- 当前" if sid == current else ""
                print(f"  {sid}{marker}")
            continue
        if user_input.startswith("/switch"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or parts[1] not in runtime.list_sessions():
                print("用法: /switch <存在的会话id>（/sessions 查看）")
                continue
            current = parts[1]
            print(f"已切换到会话: {current}")
            continue
        if user_input == "/history":
            _print_history(runtime, current)
            continue
        if user_input == "/trace":
            _print_trace(runtime, current)
            continue

        result = runtime.chat(current, user_input)
        _print_chat_result(result)


if __name__ == "__main__":
    main()
