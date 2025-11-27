import asyncio
from unittest.mock import AsyncMock

async def fetch_data(api):
    data = await api.get_users()
    return data

async def test_async():
    mock_api = AsyncMock()
    mock_api.get_users.return_value = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    
    result = await fetch_data(mock_api)
    print(f"Fetched Data: {result}")  # Output: Fetched Data: [{'
    mock_api.get_users.assert_called_once()
    
asyncio.run(test_async())
