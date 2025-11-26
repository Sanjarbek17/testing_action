"""
Dummy test file to test the GitHub Actions workflow.
"""


def test_addition():
    """Test that basic addition works."""
    assert 1 + 1 == 2


def test_subtraction():
    """Test that basic subtraction works."""
    assert 5 - 3 == 2


def test_multiplication():
    """Test that basic multiplication works."""
    assert 3 * 4 == 12


def test_division():
    """Test that basic division works."""
    assert 10 / 2 == 5


def test_string_concatenation():
    """Test that string concatenation works."""
    result = "Hello" + " " + "World"
    assert result == "Hello World"


def test_list_operations():
    """Test that list operations work."""
    my_list = [1, 2, 3]
    my_list.append(4)
    assert len(my_list) == 4
    assert my_list[-1] == 4
