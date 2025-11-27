from unittest.mock import MagicMock

db = MagicMock()
db.query.return_value.filter.return_value.all.return_value = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
]

result = db.query("Users").filter(name="Alice").all()
print(result)  # Output: [{'id': 1, 'name': 'Alice'},