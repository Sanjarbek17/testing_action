# MagicMock Learning Path 🎓

Welcome to your progressive MagicMock tutorial! This guide will take you from beginner to advanced mocking techniques in Python.

## 📚 What is MagicMock?

MagicMock is a powerful testing tool from Python's `unittest.mock` module that allows you to:
- Replace real objects/functions with fake ones during tests
- Control what values are returned
- Track how your code uses these objects
- Test code without external dependencies (databases, APIs, files, etc.)

## 🎯 Learning Structure

This tutorial is organized into 3 levels with **21 hands-on tasks**. Complete one task at a time before moving to the next!

### 📗 Level 1: Beginner (Tasks 1-5)
**File**: `01_beginner_tasks.md`

Learn the fundamentals:
- Creating mock objects
- Setting return values
- Checking if methods were called
- Counting calls
- Mocking attributes

**Time estimate**: 30-45 minutes

---

### 📘 Level 2: Intermediate (Tasks 6-12)
**File**: `02_intermediate_tasks.md`

Build on the basics:
- Side effects (custom logic & exceptions)
- Using `@patch` decorator
- Mocking class methods
- Using `spec` parameter
- Inspecting call arguments

**Time estimate**: 1-1.5 hours

---

### 📕 Level 3: Advanced (Tasks 13-21)
**File**: `03_advanced_tasks.md`

Master complex scenarios:
- Chained method calls
- Context managers
- PropertyMock & AsyncMock
- Multiple patches
- Custom mock subclasses
- pytest integration
- Flexible argument matching
- Real-world API testing

**Time estimate**: 2-3 hours

---

## 🚀 How to Use This Tutorial

1. **Start with Task 1** in `01_beginner_tasks.md`
2. **Read the objective** and instructions carefully
3. **Complete the code** in the examples
4. **Run your code** to verify it works
5. **Check the expected output** matches
6. **Move to the next task** only after succeeding
7. **Mark tasks complete** in the checklist

## 💡 Tips for Success

- **Type the code yourself** - don't just copy/paste
- **Experiment** - try changing values and see what happens
- **Run each example** - learning by doing is key
- **Take breaks** - these concepts build on each other
- **Ask questions** - if stuck, research or ask for help
- **Keep notes** - write down your "aha!" moments

## 🔧 Setup

Make sure you have Python 3.7+ installed. MagicMock comes built-in with Python, so no installation needed!

```bash
# Verify your Python version
python3 --version

# Create a test file to try examples
touch test_mock_practice.py
```

## 📝 Practice Files Structure

```
magicmock_tutorial/
├── README.md                    # This file (overview)
├── 01_beginner_tasks.md        # Tasks 1-5
├── 02_intermediate_tasks.md    # Tasks 6-12
└── 03_advanced_tasks.md        # Tasks 13-21
```

## 🎯 Your Progress

Track your journey:

- [ ] Completed Beginner Level (5 tasks)
- [ ] Completed Intermediate Level (7 tasks)
- [ ] Completed Advanced Level (9 tasks)

**Total**: 21 tasks to master MagicMock!

## 📖 Additional Resources

- [Official Python Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Real Python: Understanding Mock](https://realpython.com/python-mock-library/)
- [pytest-mock Plugin](https://github.com/pytest-dev/pytest-mock)

---

## 🏁 Ready to Start?

Open `01_beginner_tasks.md` and begin with Task 1!

**Remember**: Complete one task at a time. Master each concept before moving forward.

Good luck! 🚀
