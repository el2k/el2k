"""todo 工具：待办事项管理。

存储按 session 隔离（内存 dict，key 为 session_id，由 ToolRegistry 自动注入），
因此同一用户开的多个窗口各自维护独立的待办列表，互不影响。
"""
# 这是一个演示工具，展示如何管理会话状态。生产环境建议使用 Redis 存储。
from ..tool_registry import Tool

# session_id -> [{"task": str, "done": bool}, ...]
_store = {}


def _tasks_of(session_id):
    key = session_id if session_id is not None else "_global"
    return _store.setdefault(key, [])


def todo(action, task=None, index=None, session_id=None):
    """管理当前会话的待办列表：add / list / remove / clear。"""
    tasks = _tasks_of(session_id)

    if action == "add":
        if not task or not str(task).strip():
            return {"error": "add 操作需要提供 task 参数"}
        tasks.append({"task": str(task).strip(), "done": False})
        return {"status": "added", "task": str(task).strip(), "total": len(tasks)}

    if action == "list":
        return {"status": "ok", "tasks": list(tasks), "total": len(tasks)}

    if action == "remove":
        if index is None or not (0 <= int(index) < len(tasks)):
            return {"error": f"无效的 index（当前共 {len(tasks)} 条），请先 list 查看"}
        removed = tasks.pop(int(index))
        return {"status": "removed", "task": removed["task"], "total": len(tasks)}

    if action == "clear":
        tasks.clear()
        return {"status": "cleared", "total": 0}

    return {"error": f"未知 action: {action}，支持 add/list/remove/clear"}


TOOL = Tool(
    name="todo",
    description="管理当前会话的待办事项。action 支持 add（新增，需 task）、list（查看）、remove（按 index 删除）、clear（清空）。",
    schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "remove", "clear"],
                "description": "要执行的操作",
            },
            "task": {
                "type": "string",
                "description": "待办内容（add 时必填）",
            },
            "index": {
                "type": "integer",
                "description": "要删除的待办序号（remove 时使用，0 起）",
            },
        },
        "required": ["action"],
    },
    func=todo,
)
