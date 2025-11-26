from unittest.mock import patch

def get_user_from_db(user_id):
    # Simulate a database call
    return {"id": user_id, "name": "John Doe"}

@patch('__main__.get_user_from_db')
def test_get_user_from_db(mock_get_user):
    mock_get_user.return_value = {"id": 1, "name": "Mocked User"}

    user = get_user_from_db(1)
    print(f"User fetched: {user}")
    mock_get_user.assert_called_once_with(1)

test_get_user_from_db()