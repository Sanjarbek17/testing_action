# MagicMock Intermediate Tasks 🚀

## Task 6: Side Effects with Functions
**Objective**: Make a mock execute custom logic when called

**Instructions**:
1. Create a function that will be the side effect
2. Assign it to `side_effect`
3. Call the mock and see your function execute

**Example to complete**:
```python
from unittest.mock import MagicMock

def custom_logic(x):
    return x * 2

mock = MagicMock()
mock.double.side_effect = custom_logic

result = mock.double(5)
print(f"Result: {result}")
```

**Expected Output**: `Result: 10`

---

## Task 7: Side Effects with Lists
**Objective**: Return different values on consecutive calls

**Instructions**:
1. Create a mock with a list of return values
2. Call it multiple times
3. Each call returns the next value in the list

**Example to complete**:
```python
from unittest.mock import MagicMock

mock = MagicMock()
mock.get_number.side_effect = [1, 2, 3, 4]

print(mock.get_number())  # First call
print(mock.get_number())  # Second call
print(mock.get_number())  # Third call
```

**Expected Output**: 
```
1
2
3
```

---

## Task 8: Raising Exceptions
**Objective**: Make a mock raise an exception

**Instructions**:
1. Set `side_effect` to an exception class
2. Use try/except to catch it when called

**Example to complete**:
```python
from unittest.mock import MagicMock

mock = MagicMock()
mock.risky_operation.side_effect = ValueError("Something went wrong!")

try:
    mock.risky_operation()
except ValueError as e:
    print(f"Caught: {e}")
```

**Expected Output**: `Caught: Something went wrong!`

---

## Task 9: Patching Functions
**Objective**: Replace a real function with a mock temporarily

**Instructions**:
1. Create a simple function in a file
2. Use `@patch` decorator to replace it in tests
3. Verify the mock was used instead

**Example to complete**:
```python
from unittest.mock import patch

# Assume you have this function somewhere
def get_user_from_db(user_id):
    # Expensive database call
    return {"id": user_id, "name": "Real User"}

# Now test it
@patch('__main__.get_user_from_db')
def test_user_retrieval(mock_get_user):
    mock_get_user.return_value = {"id": 1, "name": "Mock User"}
    
    user = get_user_from_db(1)
    print(f"Got user: {user['name']}")
    mock_get_user.assert_called_once_with(1)

test_user_retrieval()
```

**Expected Output**: `Got user: Mock User`

---

## Task 10: Mocking Object Methods
**Objective**: Mock a method on a class instance

**Instructions**:
1. Create a simple class
2. Create an instance
3. Replace one of its methods with a mock

**Example to complete**:
```python
from unittest.mock import MagicMock

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

calc = Calculator()
calc.multiply = MagicMock(return_value=100)

print(f"2 + 3 = {calc.add(2, 3)}")        # Real method
print(f"2 * 3 = {calc.multiply(2, 3)}")   # Mocked method
```

**Expected Output**: 
```
2 + 3 = 5
2 * 3 = 100
```

---

## Task 11: Spec Parameter
**Objective**: Restrict mock to only have methods/attributes of a real class

**Instructions**:
1. Create a class with specific methods
2. Create a mock with `spec` parameter
3. Try calling a method that doesn't exist (it should fail)

**Example to complete**:
```python
from unittest.mock import MagicMock

class User:
    def login(self):
        pass
    
    def logout(self):
        pass

mock_user = MagicMock(spec=User)
mock_user.login()  # This works
print("Login called successfully")

# Try this - it will raise AttributeError
# mock_user.invalid_method()
```

---

## Task 12: Call Arguments
**Objective**: Inspect what arguments were used in calls

**Instructions**:
1. Create a mock
2. Call it with different arguments
3. Use `call_args_list` to see all calls

**Example to complete**:
```python
from unittest.mock import MagicMock

mock = MagicMock()
mock.log("Error", "File not found")
mock.log("Warning", "Low memory")
mock.log("Info", "Process started")

for call in mock.log.call_args_list:
    print(f"Log called with: {call}")
```

---

## ✅ Completion Checklist
- [ ] Completed Task 6
- [ ] Completed Task 7
- [ ] Completed Task 8
- [ ] Completed Task 9
- [ ] Completed Task 10
- [ ] Completed Task 11
- [ ] Completed Task 12

**Next**: Move to `03_advanced_tasks.md` when ready!
