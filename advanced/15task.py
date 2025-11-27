from unittest.mock import patch, PropertyMock

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
with patch.object(type(user), 'name', new_callable=PropertyMock) as mock_name:
    mock_name.return_value = "Mocked Name"
    print(f"User name: {user.name}")  # Output: User name: Mocked Name
    result = mock_name.assert_called()
    print(result)