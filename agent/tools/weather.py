import random
from agent.tool_registry import get_default_registry, Tool

reg = get_default_registry()

@reg.register(
    name="weather",
    description="Get weather information for a city. Returns mock weather data.",
    schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name to get weather for, e.g. 'Beijing', 'Shanghai'"
            }
        },
        "required": ["city"]
    }
)
def weather(city):
    mock_weather = {
        "Beijing": {"temperature": 22, "condition": "Sunny", "humidity": 45},
        "Shanghai": {"temperature": 28, "condition": "Rainy", "humidity": 78},
        "Guangzhou": {"temperature": 32, "condition": "Cloudy", "humidity": 82},
        "Shenzhen": {"temperature": 30, "condition": "Partly Cloudy", "humidity": 75},
        "Chengdu": {"temperature": 20, "condition": "Foggy", "humidity": 88},
    }
    city_data = mock_weather.get(city)
    if city_data:
        return {"city": city, **city_data, "source": "mock"}
    return {
        "city": city,
        "temperature": random.randint(-10, 40),
        "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]),
        "humidity": random.randint(20, 95),
        "source": "mock"
    }