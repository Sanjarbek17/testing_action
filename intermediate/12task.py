from unittest.mock import MagicMock

mock = MagicMock()
mock.log("Error", "File not found")
mock.log("Warning", "Low disk space")
mock.log("Info", "Operation completed")

for call in mock.log.call_args_list:
    print(f"Called with args: {call.args}")