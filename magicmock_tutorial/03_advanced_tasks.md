# MagicMock Advanced Tasks 🔥

## Task 13: Chained Method Calls
**Objective**: Mock methods that return objects with their own methods

**Instructions**:
1. Create a mock that supports chaining (e.g., `db.query().filter().all()`)
2. Set return values at each level
3. Verify the chain works

**Example to complete**:
```python
from unittest.mock import MagicMock

# Mock a database query chain
db = MagicMock()
db.query.return_value.filter.return_value.all.return_value = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
]

results = db.query("User").filter(age=25).all()
print(f"Found {len(results)} users")
for user in results:
    print(f"  - {user['name']}")
```

**Expected Output**:
```
Found 2 users
  - Alice
  - Bob
```

---

## Task 14: Context Manager Mocking
**Objective**: Mock objects used with `with` statement

**Instructions**:
1. Mock a file object or similar context manager
2. Use `__enter__` and `__exit__` magic methods
3. Test the context manager behavior

**Example to complete**:
```python
from unittest.mock import MagicMock, mock_open, patch

# Method 1: Using mock_open
mock_file = mock_open(read_data="Hello from mock file!")

with patch('builtins.open', mock_file):
    with open('somefile.txt', 'r') as f:
        content = f.read()
        print(f"Content: {content}")

# Method 2: Manual context manager
mock_db = MagicMock()
mock_db.__enter__.return_value = mock_db
mock_db.execute.return_value = [{"result": "success"}]

with mock_db as db:
    result = db.execute("SELECT * FROM users")
    print(f"Query result: {result}")
```

---

## Task 15: PropertyMock
**Objective**: Mock property getters and setters

**Instructions**:
1. Import `PropertyMock`
2. Mock a class property
3. Test getting and setting the property

**Example to complete**:
```python
from unittest.mock import MagicMock, PropertyMock

class User:
    def __init__(self):
        self._name = ""
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value

user = User()
type(user).name = PropertyMock(return_value="Mocked Name")

print(f"User name: {user.name}")

# Verify property was accessed
type(user).name.assert_called()
```

---

## Task 16: AsyncMock (Python 3.8+)
**Objective**: Mock asynchronous functions

**Instructions**:
1. Import `AsyncMock`
2. Create an async mock
3. Use it in an async context

**Example to complete**:
```python
import asyncio
from unittest.mock import AsyncMock

async def fetch_data(api):
    data = await api.get_users()
    return data

async def test_async():
    mock_api = AsyncMock()
    mock_api.get_users.return_value = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]
    
    result = await fetch_data(mock_api)
    print(f"Fetched {len(result)} users")
    mock_api.get_users.assert_called_once()

# Run the test
asyncio.run(test_async())
```

---

## Task 17: Patch Multiple Targets
**Objective**: Patch multiple functions/methods simultaneously

**Instructions**:
1. Use `patch` as context manager for multiple targets
2. Test code that depends on multiple external dependencies

**Example to complete**:
```python
from unittest.mock import patch

def process_payment(amount):
    from payment_gateway import charge_card
    from email_service import send_receipt
    from logger import log_transaction
    
    transaction_id = charge_card(amount)
    send_receipt(transaction_id)
    log_transaction(transaction_id, amount)
    return transaction_id

# Mock all dependencies
with patch('payment_gateway.charge_card', return_value="TX123") as mock_charge, \
     patch('email_service.send_receipt') as mock_email, \
     patch('logger.log_transaction') as mock_log:
    
    result = process_payment(99.99)
    
    print(f"Transaction: {result}")
    mock_charge.assert_called_once_with(99.99)
    mock_email.assert_called_once_with("TX123")
    mock_log.assert_called_once_with("TX123", 99.99)
```

---

## Task 18: Custom Mock Subclass
**Objective**: Create a custom mock class with predefined behavior

**Instructions**:
1. Subclass `MagicMock`
2. Add custom methods or override behavior
3. Use your custom mock in tests

**Example to complete**:
```python
from unittest.mock import MagicMock

class DatabaseMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = []
    
    def insert(self, item):
        self._data.append(item)
        return len(self._data)
    
    def get_all(self):
        return self._data
    
    def count(self):
        return len(self._data)

db = DatabaseMock()
db.insert({"name": "Alice"})
db.insert({"name": "Bob"})

print(f"Database has {db.count()} records")
print(f"Records: {db.get_all()}")
```

---

## Task 19: Testing with pytest and Mock
**Objective**: Integrate MagicMock with pytest fixtures

**Instructions**:
1. Create a pytest test file
2. Use fixtures to provide mocks
3. Test a function with mocked dependencies

**Example to complete**:
```python
import pytest
from unittest.mock import MagicMock, patch

class EmailService:
    def send(self, to, subject, body):
        # Actual implementation would send email
        raise NotImplementedError("Real email sending")

def notify_user(user_id, message):
    email_service = EmailService()
    email = f"user{user_id}@example.com"
    email_service.send(email, "Notification", message)
    return True

@pytest.fixture
def mock_email_service():
    with patch('__main__.EmailService') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

def test_notify_user(mock_email_service):
    result = notify_user(123, "Hello!")
    
    assert result is True
    mock_email_service.send.assert_called_once_with(
        "user123@example.com",
        "Notification",
        "Hello!"
    )
    print("✓ Test passed!")

# Run with: pytest -v thisfile.py
```

---

## Task 20: Mock with ANY and call Matchers
**Objective**: Use flexible argument matching in assertions

**Instructions**:
1. Import `ANY`, `call` from unittest.mock
2. Assert calls with partial matching
3. Use `call` objects to check call lists

**Example to complete**:
```python
from unittest.mock import MagicMock, ANY, call

mock = MagicMock()

# Make various calls
mock.process(user_id=1, timestamp=1234567890, data={"key": "value"})
mock.process(user_id=2, timestamp=1234567999, data={"key": "other"})

# Assert with ANY for timestamp (we don't care about exact value)
mock.process.assert_any_call(user_id=1, timestamp=ANY, data={"key": "value"})

# Check all calls
expected_calls = [
    call(user_id=1, timestamp=ANY, data=ANY),
    call(user_id=2, timestamp=ANY, data=ANY)
]
mock.process.assert_has_calls(expected_calls)

print("✓ All assertions passed!")
```

---

## Task 21: Real-World Scenario - API Client Testing
**Objective**: Mock an external API client in a realistic scenario

**Instructions**:
Complete this real-world example of testing a service that calls an external API

**Example to complete**:
```python
from unittest.mock import MagicMock, patch
import json

class WeatherAPI:
    def get_weather(self, city):
        # Would make real HTTP request
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
            return response['data']['conditions'] == 'rain'
        return False

# Test it
mock_api = MagicMock()
mock_api.get_weather.return_value = {
    'status': 'success',
    'data': {
        'temperature': 22,
        'conditions': 'sunny',
        'humidity': 65
    }
}

service = WeatherService(mock_api)
temp = service.get_temperature("New York")
raining = service.is_raining("New York")

print(f"Temperature: {temp}°C")
print(f"Is raining: {raining}")

# Verify API was called
assert mock_api.get_weather.call_count == 2
print("✓ Test completed!")
```

---

## ✅ Completion Checklist
- [ ] Completed Task 13: Chained method calls
- [ ] Completed Task 14: Context managers
- [ ] Completed Task 15: PropertyMock
- [ ] Completed Task 16: AsyncMock
- [ ] Completed Task 17: Multiple patches
- [ ] Completed Task 18: Custom mock subclass
- [ ] Completed Task 19: pytest integration
- [ ] Completed Task 20: ANY and call matchers
- [ ] Completed Task 21: Real-world API testing

**Congratulations!** 🎉 You've completed all MagicMock tasks!

## Next Steps
- Apply these skills to your actual project tests
- Explore `pytest-mock` plugin for even more features
- Read the official documentation: https://docs.python.org/3/library/unittest.mock.html
- Practice writing tests for your real code using these techniques
