import json
import os
from agent.tool_registry import get_default_registry, Tool

reg = get_default_registry()

@reg.register(
    name="todo",
    description="Manage your todo list. Supports adding, listing, and removing tasks.",
    schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "remove", "clear"],
                "description": "Action to perform: add, list, remove, or clear"
            },
            "task": {
                "type": "string",
                "description": "Task description (required for 'add', optional for 'remove')"
            },
            "index": {
                "type": "integer",
                "description": "Task index to remove (0-based)"
            }
        },
        "required": ["action"]
    }
)
def todo(action, task=None, index=None):
    todo_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".todo_data.json")

    if os.path.exists(todo_file):
        with open(todo_file, "r") as f:
            tasks = json.load(f)
    else:
        tasks = []

    if action == "add":
        if not task:
            return {"status": "error", "message": "Task description required for 'add'"}
        tasks.append({"task": task, "done": False})
        with open(todo_file, "w") as f:
            json.dump(tasks, f, indent=2)
        return {"status": "added", "task": task, "total": len(tasks)}

    elif action == "list":
        return {"status": "ok", "tasks": tasks, "total": len(tasks)}

    elif action == "remove":
        if index is None or index < 0 or index >= len(tasks):
            return {"status": "error", "message": f"Invalid index. Use 'list' to see tasks."}
        removed = tasks.pop(index)
        with open(todo_file, "w") as f:
            json.dump(tasks, f, indent=2)
        return {"status": "removed", "task": removed["task"], "total": len(tasks)}

    elif action == "clear":
        tasks.clear()
        with open(todo_file, "w") as f:
            json.dump(tasks, f, indent=2)
        return {"status": "cleared", "total": 0}

    return {"status": "error", "message": f"Unknown action: {action}"}