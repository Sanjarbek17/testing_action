from unittest.mock import MagicMock

mock = MagicMock()

mock.send_email("user@example.com", "hello")


result = mock.send_email.assert_called_with(
    "user@example.com", "hello"
)

print(result)