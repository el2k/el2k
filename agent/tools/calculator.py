import math
from agent.tool_registry import get_default_registry, Tool

reg = get_default_registry()

@reg.register(
    name="calculator",
    description="Perform arithmetic calculations. Supports +, -, *, /, **, sqrt, and basic expressions.",
    schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g. '2 + 3 * 4', 'sqrt(16)', '2 ** 10'"
            }
        },
        "required": ["expression"]
    }
)
def calculator(expression):
    safe_expr = expression.replace("sqrt", "math.sqrt")
    allowed_names = {"math": math}
    try:
        result = eval(safe_expr, {"__builtins__": {}}, allowed_names)
        if isinstance(result, float):
            return round(result, 10)
        return result
    except Exception as e:
        return f"Error: {str(e)}"