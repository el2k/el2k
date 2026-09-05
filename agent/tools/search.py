"""search 工具：mock 网络搜索，返回构造好的演示结果。"""

from ..tool_registry import Tool


def search(query, num_results=5):
    """按关键词搜索（mock 数据源）。"""
    try:
        n = max(1, min(int(num_results), 5))
    except (TypeError, ValueError):
        n = 5
    results = [
        {
            "title": f"{query} - 相关结果 {i + 1}",
            "url": f"https://example.com/search?q={query}&r={i + 1}",
            "snippet": f"关于「{query}」的第 {i + 1} 条演示搜索结果（mock 数据源）。",
        }
        for i in range(n)
    ]
    return {"query": query, "results": results, "total": len(results), "source": "mock"}


TOOL = Tool(
    name="search",
    description="搜索互联网信息（当前为 mock 数据源，返回演示用搜索结果）。",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "num_results": {
                "type": "integer",
                "description": "返回结果数量，1-5，默认 5",
            },
        },
        "required": ["query"],
    },
    func=search,
)
