from unittest.mock import MagicMock

mock = MagicMock()

mock.get_name.return_value = "Sanjarbek"

result  = mock.get_name()
print(f"Name {result}")