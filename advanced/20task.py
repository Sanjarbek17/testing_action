from unittest.mock import MagicMock, ANY, call

mock = MagicMock()

mock.process(user_id=1, timestamp=1234567890, data={"key": "value"})
mock.process(user_id=2, timestamp=1234567891, data={"key": "another_value"})

mock.process.assert_any_call(user_id=1, timestamp=ANY, data={"key": "value"})

expected_calls = [
    call(user_id=1, timestamp=ANY, data={"key": "value"}),
    call(user_id=2, timestamp=ANY, data={"key": "another_value"}),
]

mock.process.assert_has_calls(expected_calls)
print("✓ All assertions passed!")