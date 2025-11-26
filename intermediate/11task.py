from unittest.mock import MagicMock

class User:
    def login(self):
        pass

    def logout(self):
        pass
    
mock_user = MagicMock(spec=User)
# mock_user.login()
# print("Login called:", mock_user.login.called)

mock_user.invalid_method()
print("Invalid method called:", mock_user.invalid_method.called)