from unittest.mock import MagicMock

def custom_logic(x):
    return x * 2

mock  = MagicMock()
mock.double.side_effect = custom_logic

result = mock.double(9)
print(f"Result of custom logic: {result}")