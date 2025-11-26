from unittest.mock import MagicMock

mock = MagicMock()

mock.increment()
mock.increment()
mock.increment()
mock.increment()

print(f"Called times: {mock.increment.call_count}")