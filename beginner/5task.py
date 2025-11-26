from unittest.mock import MagicMock

user_mock = MagicMock()
user_mock.name = "Alice"
user_mock.age = 30
user_mock.email = "alice@example.com"

print(f"User Name: {user_mock.name}")
print(f"User Age: {user_mock.age}")
print(f"User Email: {user_mock.email}")