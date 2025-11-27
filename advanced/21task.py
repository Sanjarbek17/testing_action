from unittest.mock import MagicMock, patch
import json

class WeatherAPI:
    def get_weather(self, city):
        pass
    
class WeatherService:
    def __init__(self, api_client):
        self.api = api_client
    
    def get_temperature(self, city):
        response = self.api.get_weather(city)
        if response['status'] == 'success':
            return response['data']['temperature']
        return None
    
    def is_raining(self, city):
        response = self.api.get_weather(city)
        if response['status'] == 'success':
            return response['data']['condition'] == 'rain'
        return False

mock_api = MagicMock()
mock_api.get_weather.return_value = {
    'status': 'success',
    'data': {
        'temperature': 22,
        'condition': 'rain'
    }
}

service = WeatherService(mock_api)
temp = service.get_temperature("New York")
raining = service.is_raining("New York")

print(f"Temperature: {temp}°C")
print(f"Is it raining? {'Yes' if raining else 'No'}")

assert mock_api.get_weather.call_count == 2
print("✓ All assertions passed!")