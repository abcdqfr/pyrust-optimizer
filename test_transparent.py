"""
Simple test file for transparent optimization.
"""

def slow_loop_function(n):
    """A function with a slow loop that should be optimized."""
    result = 0
    for i in range(n):
        for j in range(100):
            result += i * j
    return result

def math_heavy_function(data):
    """A function with heavy mathematical operations."""
    result = []
    for item in data:
        # Nested mathematical operations
        value = item ** 2
        value = value * 3.14159
        value = value / 2.71828
        result.append(value)
    return result

def normal_function():
    """A normal function that doesn't need optimization."""
    return "This function is fast enough!"

if __name__ == "__main__":
    # Test the functions
    print("Testing functions...")
    print(f"slow_loop_function(10): {slow_loop_function(10)}")
    print(f"math_heavy_function([1, 2, 3]): {math_heavy_function([1, 2, 3])}")
    print(f"normal_function(): {normal_function()}")
