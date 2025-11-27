from unittest.mock import MagicMock, mock_open, patch

mock_file = mock_open(read_data="Hello from mock file!")

with patch('builtins.open', mock_file):
    with open('somefile.txt', 'r') as f:
        content = f.read()
        print(f"File content: {content}")  # Output: File content: Hello from mock file!

mock_db = MagicMock()
mock_db.__enter__.return_value = mock_db
mock_db.execute.return_value = [{"result": "success"}]

with mock_db as db:
    result = db.execute("SELECT * FROM table")
    print(f"DB Query Result: {result}")  # Output: DB Query Result: [{'result': 'success'}]