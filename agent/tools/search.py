from agent.tool_registry import get_default_registry, Tool

reg = get_default_registry()

@reg.register(
    name="search",
    description="Search the web for information. Returns mock results for demo purposes.",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
)
def search(query, num_results=5):
    mock_results = [
        {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i+1}", "snippet": f"This is mock search result {i+1} about '{query}'"}
        for i in range(min(num_results, 5))
    ]
    return {"query": query, "results": mock_results, "total": len(mock_results)}