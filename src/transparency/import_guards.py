"""
Import Guards for PyRust Optimizer Transparent Integration.

This module provides smart import guards that try Rust optimizations first
and gracefully fall back to original Python implementations.
"""

import importlib
import sys
import traceback
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import functools


class ImportGuard:
    """
    Smart import guard that handles Rust optimization fallback.

    This class provides decorators and utilities to create functions that
    automatically try Rust implementations first and fall back to Python
    if the Rust version is unavailable or fails.
    """

    def __init__(self, module_name: str = "_pyrust_optimized"):
        self.module_name = module_name
        self.rust_module = None
        self.rust_available = False
        self.failed_functions = set()
        self.performance_stats = {}

        # Try to import Rust module
        self._try_import_rust_module()

    def _try_import_rust_module(self):
        """Attempt to import the Rust optimization module."""
        try:
            # Try absolute import first
            try:
                self.rust_module = importlib.import_module(self.module_name)
            except ImportError:
                # Try relative import with current package
                self.rust_module = importlib.import_module(f".{self.module_name}", package=__name__.split('.')[0])

            self.rust_available = True
            print(f"✅ Loaded Rust optimizations from {self.module_name}")
        except ImportError as e:
            self.rust_available = False
            print(f"⚠️ Rust optimizations not available: {e}")
        except Exception as e:
            self.rust_available = False
            print(f"❌ Error loading Rust module: {e}")

    def create_guarded_function(self, func_name: str, python_implementation: Callable) -> Callable:
        """
        Create a function with Rust optimization guard.

        Args:
            func_name: Name of the function in the Rust module
            python_implementation: Original Python function to fall back to

        Returns:
            Guarded function that tries Rust first, then Python
        """

        @functools.wraps(python_implementation)
        def guarded_function(*args, **kwargs):
            # Skip Rust if we know it's failed before
            if func_name in self.failed_functions:
                return self._call_python_with_stats(func_name, python_implementation, *args, **kwargs)

            # Try Rust implementation first
            if self.rust_available and hasattr(self.rust_module, func_name):
                try:
                    rust_func = getattr(self.rust_module, func_name)
                    result = self._call_rust_with_stats(func_name, rust_func, *args, **kwargs)
                    return result
                except Exception as e:
                    # Mark as failed and fall back to Python
                    self.failed_functions.add(func_name)
                    print(f"⚠️ Rust optimization failed for {func_name}, falling back to Python: {e}")

            # Fall back to Python implementation
            return self._call_python_with_stats(func_name, python_implementation, *args, **kwargs)

        return guarded_function

    def _call_rust_with_stats(self, func_name: str, rust_func: Callable, *args, **kwargs):
        """Call Rust function and track performance statistics."""
        import time
        start_time = time.time()

        try:
            result = rust_func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Update performance stats
            if func_name not in self.performance_stats:
                self.performance_stats[func_name] = {
                    'rust_calls': 0,
                    'python_calls': 0,
                    'rust_total_time': 0.0,
                    'python_total_time': 0.0
                }

            stats = self.performance_stats[func_name]
            stats['rust_calls'] += 1
            stats['rust_total_time'] += execution_time

            return result

        except Exception as e:
            # Remove from stats if it failed
            execution_time = time.time() - start_time
            raise e

    def _call_python_with_stats(self, func_name: str, python_func: Callable, *args, **kwargs):
        """Call Python function and track performance statistics."""
        import time
        start_time = time.time()

        result = python_func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Update performance stats
        if func_name not in self.performance_stats:
            self.performance_stats[func_name] = {
                'rust_calls': 0,
                'python_calls': 0,
                'rust_total_time': 0.0,
                'python_total_time': 0.0
            }

        stats = self.performance_stats[func_name]
        stats['python_calls'] += 1
        stats['python_total_time'] += execution_time

        return result

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report showing Rust vs Python usage."""
        report = {
            'rust_available': self.rust_available,
            'total_functions': len(self.performance_stats),
            'failed_functions': list(self.failed_functions),
            'function_stats': {}
        }

        for func_name, stats in self.performance_stats.items():
            func_report = {
                'rust_calls': stats['rust_calls'],
                'python_calls': stats['python_calls'],
                'total_calls': stats['rust_calls'] + stats['python_calls']
            }

            # Calculate average times
            if stats['rust_calls'] > 0:
                func_report['rust_avg_time'] = stats['rust_total_time'] / stats['rust_calls']
            if stats['python_calls'] > 0:
                func_report['python_avg_time'] = stats['python_total_time'] / stats['python_calls']

            # Calculate speedup if both have been called
            if stats['rust_calls'] > 0 and stats['python_calls'] > 0:
                rust_avg = stats['rust_total_time'] / stats['rust_calls']
                python_avg = stats['python_total_time'] / stats['python_calls']
                if rust_avg > 0:
                    func_report['actual_speedup'] = python_avg / rust_avg

            # Calculate usage percentage
            total_calls = func_report['total_calls']
            if total_calls > 0:
                func_report['rust_usage_percent'] = (stats['rust_calls'] / total_calls) * 100
                func_report['python_usage_percent'] = (stats['python_calls'] / total_calls) * 100

            report['function_stats'][func_name] = func_report

        return report

    def print_performance_report(self):
        """Print a formatted performance report."""
        report = self.get_performance_report()

        print("\n🔥 PyRust Optimizer Performance Report")
        print("=" * 45)
        print(f"Rust Module Available: {report['rust_available']}")
        print(f"Total Functions Tracked: {report['total_functions']}")

        if report['failed_functions']:
            print(f"Failed Rust Functions: {', '.join(report['failed_functions'])}")

        print("\n📊 Function Performance:")
        print("-" * 45)

        for func_name, stats in report['function_stats'].items():
            print(f"\n🎯 {func_name}:")
            print(f"   Total Calls: {stats['total_calls']}")
            print(f"   Rust Calls: {stats['rust_calls']} ({stats.get('rust_usage_percent', 0):.1f}%)")
            print(f"   Python Calls: {stats['python_calls']} ({stats.get('python_usage_percent', 0):.1f}%)")

            if 'rust_avg_time' in stats:
                print(f"   Rust Avg Time: {stats['rust_avg_time']:.6f}s")
            if 'python_avg_time' in stats:
                print(f"   Python Avg Time: {stats['python_avg_time']:.6f}s")
            if 'actual_speedup' in stats:
                print(f"   Actual Speedup: {stats['actual_speedup']:.2f}x")

    def reset_failed_functions(self):
        """Reset the list of failed functions to retry Rust optimizations."""
        self.failed_functions.clear()
        print("♻️ Reset failed functions - will retry Rust optimizations")

    def force_python_mode(self):
        """Force all functions to use Python implementation."""
        self.rust_available = False
        print("🐍 Forced Python mode - all functions will use Python implementation")

    def reload_rust_module(self):
        """Attempt to reload the Rust optimization module."""
        try:
            if self.rust_module:
                importlib.reload(self.rust_module)
            else:
                self._try_import_rust_module()
            print("🔄 Reloaded Rust optimization module")
        except Exception as e:
            print(f"❌ Failed to reload Rust module: {e}")


class GuardedFunctionGenerator:
    """
    Generates guarded function code for transparent optimization.

    This class helps create the Python code that includes import guards
    for transparent Rust optimization fallback.
    """

    @staticmethod
    def generate_guarded_function_code(func_name: str, params: str, docstring: str,
                                     original_body: str, rust_module: str = "_pyrust_optimized") -> str:
        """
        Generate code for a function with Rust optimization guards.

        Args:
            func_name: Name of the function
            params: Function parameters as string
            docstring: Function docstring
            original_body: Original Python function body
            rust_module: Name of the Rust module to import from

        Returns:
            Complete function code with guards
        """

        # Extract parameter names for Rust function call
        param_names = GuardedFunctionGenerator._extract_param_names(params)

        guarded_code = f'''def {func_name}({params}):
    """{docstring}"""
    if _RUST_AVAILABLE:
        try:
            return _rust_{func_name}({param_names})
        except Exception as e:
            # Fallback to original implementation
            import traceback
            print(f"⚠️ Rust optimization failed for {func_name}: {{e}}")
            if _PYRUST_DEBUG:
                traceback.print_exc()

    # Original Python implementation (preserved)
{GuardedFunctionGenerator._indent_code(original_body)}'''

        return guarded_code

    @staticmethod
    def generate_import_header(rust_functions: List[str], rust_module: str = "_pyrust_optimized") -> str:
        """Generate the import header with Rust function imports."""

        func_imports = []
        for func_name in rust_functions:
            func_imports.append(f"        {func_name} as _rust_{func_name},")

        func_list = ', '.join(rust_functions)

        header = f'''# ===== PyRust Optimizer Header =====
# Auto-generated optimization layer
# Original code preserved below

import os
_PYRUST_DEBUG = os.environ.get('PYRUST_DEBUG', '').lower() in ('1', 'true', 'yes')

try:
    # Try to import optimized Rust implementations
    from .{rust_module} import (
{chr(10).join(func_imports)}
    )
    _RUST_AVAILABLE = True
    if _PYRUST_DEBUG:
        print(f"✅ Loaded Rust optimizations: {func_list}")
except ImportError as e:
    _RUST_AVAILABLE = False
    if _PYRUST_DEBUG:
        print(f"⚠️ Rust optimizations not available: {{e}}")

# ===== PyRust Optimizer Optimized Function Wrappers =====
'''

        return header

    @staticmethod
    def _extract_param_names(params: str) -> str:
        """Extract parameter names from function signature."""
        if not params.strip():
            return ""

        param_names = []
        for param in params.split(','):
            param = param.strip()
            if '=' in param:
                param = param.split('=')[0].strip()
            if param and not param.startswith('*'):
                param_names.append(param)

        return ', '.join(param_names)

    @staticmethod
    def _indent_code(code: str) -> str:
        """Indent code block for use in function body."""
        lines = code.splitlines()
        indented_lines = []

        for line in lines:
            if line.strip():  # Only indent non-empty lines
                indented_lines.append(f"    {line}")
            else:
                indented_lines.append(line)

        return '\n'.join(indented_lines)


def demo_import_guards():
    """Demo the import guard functionality."""

    # Create a mock Python function
    def slow_python_function(data):
        """Slow Python implementation for testing."""
        result = []
        for item in data:
            if item > 0:
                result.append(item ** 2)
        return result

    # Create import guard
    guard = ImportGuard("nonexistent_rust_module")  # This will fail to import

    # Create guarded function
    guarded_func = guard.create_guarded_function("slow_function", slow_python_function)

    # Test the function
    test_data = [1, -2, 3, -4, 5]
    print("🧪 Testing guarded function...")

    result = guarded_func(test_data)
    print(f"Result: {result}")

    # Call it multiple times to test performance tracking
    for i in range(3):
        guarded_func([i, i+1, i+2])

    # Print performance report
    guard.print_performance_report()

    # Test code generation
    print("\n📝 Generated guarded function code:")
    print("-" * 40)

    generator = GuardedFunctionGenerator()
    header = generator.generate_import_header(["slow_function", "fast_function"])
    print(header)

    func_code = generator.generate_guarded_function_code(
        "slow_function",
        "data",
        "A slow function that needs optimization.",
        '''result = []
for item in data:
    if item > 0:
        result.append(item ** 2)
return result'''
    )
    print(func_code)


if __name__ == "__main__":
    demo_import_guards()
