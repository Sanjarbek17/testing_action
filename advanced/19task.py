import pytest
from unittest.mock import MagicMock, patch


class EmailService:
    def send(self, to, subject, body):
        # Actual implementation would send email
        raise NotImplementedError("Real email sending")


def notify_user(user_id, message):
    email_service = EmailService()
    email = f"user{user_id}@example.com"
    email_service.send(email, "Notification", message)
    return True


@pytest.fixture
def mock_email_service():
    with patch(f"{__name__}.EmailService") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


def test_notify_user(mock_email_service):
    result = notify_user(123, "Hello!")

    assert result is True
    mock_email_service.send.assert_called_once_with(
        "user123@example.com", "Notification", "Hello!"
    )
    print("✓ Test passed!")


# Run with: pytest -v thisfile.py
