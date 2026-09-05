"""calculator 工具：白名单算术表达式求值。"""

import math

from ..tool_registry import Tool


def calculator(expression):
    """计算数学表达式，支持 + - * / ** % 与 sqrt/abs/min/max/round/pow 等函数。"""
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "math": math,
        "sqrt": math.sqrt,
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "pow": pow,
    }
    try:
        result = eval(expression, safe_globals, safe_locals)  # noqa: S307 白名单求值
    except Exception as e:
        return {"error": f"无法计算表达式 '{expression}': {e}"}
    if isinstance(result, float):
        result = round(result, 10)
    return {"expression": expression, "result": result}


TOOL = Tool(
    name="calculator",
    description=(
        "计算数学表达式。支持 +、-、*、/、**（幂）、%，"
        "以及 sqrt/abs/min/max/round 等函数。"
        "例如 '2 + 3 * 4'、'sqrt(16)'、'2 ** 10'。"
    ),
    schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，例如 '2 + 3 * 4'、'sqrt(16)'",
            }
        },
        "required": ["expression"],
    },
    func=calculator,
)
