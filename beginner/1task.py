from unittest.mock import MagicMock

mock = MagicMock()

mock.hello('world')

print(mock.hello.called)