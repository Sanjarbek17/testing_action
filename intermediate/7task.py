from unittest.mock import MagicMock

mock = MagicMock()
mock.get_number.side_effect = [1,2,3,4]

print(mock.get_number())  # Output: 1
print(mock.get_number())  # Output: 2
print(mock.get_number())  # Output: 3
print(mock.get_number())  # Output: 4