from unittest.mock import MagicMock

mock = MagicMock()
mock.risky_operation.side_effect = ValueError("something went wrong!")

try:
    mock.risky_operation()
except ValueError as e:
    print(f"Caught an exception: {e}")