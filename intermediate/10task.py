from unittest.mock import MagicMock

class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
    
calc = Calculator()
calc.multiply = MagicMock(return_value=100)

print(f"2 + 3 = {calc.add(2, 3)}")          # Should print 5
print(f"4 * 5 = {calc.multiply(4, 5)}")
