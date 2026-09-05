"""weather 工具：mock 天气查询。"""

import random

from ..tool_registry import Tool

_MOCK_WEATHER = {
    "Beijing": {"temperature": 22, "condition": "晴", "humidity": 45},
    "北京": {"temperature": 22, "condition": "晴", "humidity": 45},
    "Shanghai": {"temperature": 28, "condition": "小雨", "humidity": 78},
    "上海": {"temperature": 28, "condition": "小雨", "humidity": 78},
    "Guangzhou": {"temperature": 32, "condition": "多云", "humidity": 82},
    "广州": {"temperature": 32, "condition": "多云", "humidity": 82},
    "Shenzhen": {"temperature": 30, "condition": "局部多云", "humidity": 75},
    "深圳": {"temperature": 30, "condition": "局部多云", "humidity": 75},
    "Chengdu": {"temperature": 20, "condition": "雾", "humidity": 88},
    "成都": {"temperature": 20, "condition": "雾", "humidity": 88},
}


def weather(city):
    """查询指定城市的天气（mock 数据源，未知城市返回随机演示数据）。"""
    data = _MOCK_WEATHER.get(city)
    if data is None:
        # 兼容 "beijing " / "Beijing市" 等大小写差异
        normalized = str(city).strip().lower()
        for known, known_data in _MOCK_WEATHER.items():
            if known.lower() == normalized:
                data = known_data
                break
    if data is not None:
        return {"city": city, **data, "source": "mock"}
    return {
        "city": city,
        "temperature": random.randint(-10, 40),
        "condition": random.choice(["晴", "多云", "小雨", "雪", "大风"]),
        "humidity": random.randint(20, 95),
        "source": "mock",
    }


TOOL = Tool(
    name="weather",
    description="查询指定城市当前的天气（温度、天气状况、湿度）。数据源为 mock 演示数据。",
    schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如 '北京'、'Shanghai'、'深圳'",
            }
        },
        "required": ["city"],
    },
    func=weather,
)
