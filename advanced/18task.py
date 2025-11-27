from unittest.mock import MagicMock

class DatabaseMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = []
    
    def insert(self, item):
        self._data.append(item)
        return len(self._data)
    
    def get_all(self):
        return self._data
    
    def count(self):
        return len(self._data)
    
db = DatabaseMock()
db.insert({"name": "Alice"})
db.insert({"name": "Bob"})

print(f"All Records: {db.get_all()}")  # Output: All Records: [{'name': 'Alice'}, {'name': 'Bob'}]
print(f"Total Records: {db.count()}")   # Output: Total Records: