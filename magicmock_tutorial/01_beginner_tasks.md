# MagicMock Beginner Tasks 🎯

## Task 1: Basic Mock Creation
**Objective**: Learn to create a simple MagicMock object

**Instructions**:
1. Import `MagicMock` from `unittest.mock`
2. Create a mock object
3. Call a method on it (any name you want)
4. Print what was called

**Example to complete**:
```python
from unittest.mock import MagicMock

# Create your mock here
mock = # TODO

# Call a method named 'hello'
# TODO

# Check if hello was called
print(mock.hello.called)
```

**Expected Output**: `True`

---

## Task 2: Setting Return Values
**Objective**: Make a mock return a specific value

**Instructions**:
1. Create a MagicMock
2. Set `return_value` for a method
3. Call the method and verify it returns what you set

**Example to complete**:
```python
from unittest.mock import MagicMock

mock = MagicMock()
mock.get_name.return_value = # TODO: Return "Alice"

result = mock.get_name()
print(f"Name: {result}")
```

**Expected Output**: `Name: Alice`

---

## Task 3: Checking Method Calls
**Objective**: Verify that a method was called with specific arguments

**Instructions**:
1. Create a mock
2. Call a method with some arguments
3. Use `assert_called_with()` to verify

**Example to complete**:
```python
from unittest.mock import MagicMock

mock = MagicMock()
mock.send_email("user@example.com", "Hello!")

# TODO: Assert that send_email was called with correct arguments
mock.send_email.assert_called_with(# TODO)
```

**Success message**: No error = test passed!

---

## Task 4: Counting Method Calls
**Objective**: Track how many times a method was called

**Instructions**:
1. Create a mock
2. Call a method multiple times
3. Check the `call_count` attribute

**Example to complete**:
```python
from unittest.mock import MagicMock

mock = MagicMock()

# Call the method 3 times
mock.increment()
mock.increment()
mock.increment()

print(f"Called {mock.increment.call_count} times")
```

**Expected Output**: `Called 3 times`

---

## Task 5: Mock Attributes
**Objective**: Set attributes on a mock object

**Instructions**:
1. Create a MagicMock
2. Set some attributes directly
3. Access and print them

**Example to complete**:
```python
from unittest.mock import MagicMock

user_mock = MagicMock()
user_mock.name = "Bob"
user_mock.age = 30
user_mock.email = "bob@example.com"

print(f"{user_mock.name} is {user_mock.age} years old")
```

**Expected Output**: `Bob is 30 years old`

---

## ✅ Completion Checklist
- [ ] Completed Task 1
- [ ] Completed Task 2
- [ ] Completed Task 3
- [ ] Completed Task 4
- [ ] Completed Task 5

**Next**: Move to `02_intermediate_tasks.md` when ready!
