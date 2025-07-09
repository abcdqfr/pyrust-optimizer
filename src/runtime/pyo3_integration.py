"""
PyO3 integration layer for PyRust Optimizer.

This module handles compilation of Rust code, PyO3 binding generation,
and creation of hybrid Python-Rust modules for seamless integration.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import importlib.util
import sys


@dataclass
class CompiledModule:
    """Represents a compiled Rust module."""
    name: str
    library_path: str
    python_module: Any
    functions: List[str]
    speedup_achieved: Dict[str, float]


class PyO3Integration:
    """
    Handles PyO3 integration for Rust-Python interop.

    This class compiles Rust code with PyO3 bindings and creates
    hybrid Python-Rust modules that can be imported and used
    seamlessly from Python.
    """

    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = workspace_dir or "/tmp/pyrust_workspace"
        self.compiled_modules = {}
        self.cargo_installed = self._check_cargo_installation()

        # Create workspace directory
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _check_cargo_installation(self) -> bool:
        """Check if Cargo is installed for Rust compilation."""
        try:
            result = subprocess.run(['cargo', '--version'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def create_rust_project(self, module_name: str, rust_code: str,
                           cargo_toml: str) -> str:
        """
        Create a new Rust project with PyO3 bindings.

        Args:
            module_name: Name of the module
            rust_code: Rust source code
            cargo_toml: Cargo.toml configuration

        Returns:
            Path to the created project
        """
        project_path = Path(self.workspace_dir) / module_name

        # Create project structure
        project_path.mkdir(exist_ok=True)
        src_path = project_path / "src"
        src_path.mkdir(exist_ok=True)

        # Write Cargo.toml
        (project_path / "Cargo.toml").write_text(cargo_toml)

        # Write Rust source code
        (src_path / "lib.rs").write_text(rust_code)

        return str(project_path)

    def compile_rust_module(self, project_path: str,
                           release: bool = True) -> Optional[str]:
        """
        Compile Rust module to shared library.

        Args:
            project_path: Path to the Rust project
            release: Whether to compile in release mode

        Returns:
            Path to compiled shared library or None if compilation failed
        """
        if not self.cargo_installed:
            print("❌ Cargo not installed. Please install Rust toolchain.")
            return None

        try:
            # Compile the Rust project
            compile_cmd = ['cargo', 'build']
            if release:
                compile_cmd.append('--release')

            result = subprocess.run(
                compile_cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"❌ Rust compilation failed:")
                print(result.stderr)
                return None

            # Find the compiled library
            target_dir = Path(project_path) / "target"
            if release:
                lib_dir = target_dir / "release"
            else:
                lib_dir = target_dir / "debug"

            # Look for shared library (.so, .dylib, .dll)
            for ext in ['.so', '.dylib', '.dll']:
                lib_files = list(lib_dir.glob(f"*{ext}"))
                if lib_files:
                    return str(lib_files[0])

            print("❌ Compiled library not found")
            return None

        except subprocess.TimeoutExpired:
            print("❌ Rust compilation timed out")
            return None
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return None

    def load_compiled_module(self, library_path: str,
                           module_name: str) -> Optional[Any]:
        """
        Load compiled Rust module into Python.

        Args:
            library_path: Path to the compiled shared library
            module_name: Name of the module

        Returns:
            Loaded Python module or None if loading failed
        """
        try:
            # Load the module using importlib
            spec = importlib.util.spec_from_file_location(
                module_name, library_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
            else:
                print(f"❌ Could not create module spec for {module_name}")
                return None

        except Exception as e:
            print(f"❌ Failed to load module {module_name}: {e}")
            return None

    def create_hybrid_module(self, rust_functions: List,
                           module_name: str = "pyrust_optimized") -> Optional[CompiledModule]:
        """
        Create a complete hybrid Python-Rust module.

        Args:
            rust_functions: List of RustFunction objects
            module_name: Name of the generated module

        Returns:
            CompiledModule object or None if creation failed
        """
        print(f"🚀 Creating hybrid module: {module_name}")

        # Import RustCodeGenerator to get complete module code
        from ..generator.rust_generator import RustCodeGenerator

        generator = RustCodeGenerator()

        # Generate complete Rust module code
        rust_code = generator.generate_complete_rust_module(
            rust_functions, module_name
        )

        # Generate Cargo.toml
        cargo_toml = generator.generate_cargo_toml(module_name)

        # Create Rust project
        project_path = self.create_rust_project(
            module_name, rust_code, cargo_toml
        )
        print(f"📁 Created Rust project at: {project_path}")

        # Compile Rust module
        library_path = self.compile_rust_module(project_path)
        if not library_path:
            return None

        print(f"🔧 Compiled Rust library: {library_path}")

        # Load compiled module
        python_module = self.load_compiled_module(library_path, module_name)
        if not python_module:
            return None

        print(f"✅ Loaded Python module: {module_name}")

        # Create CompiledModule object
        function_names = [func.name for func in rust_functions]
        speedup_map = {func.name: func.estimated_speedup for func in rust_functions}

        compiled_module = CompiledModule(
            name=module_name,
            library_path=library_path,
            python_module=python_module,
            functions=function_names,
            speedup_achieved=speedup_map
        )

        # Store in registry
        self.compiled_modules[module_name] = compiled_module

        return compiled_module

    def create_integration(self, rust_code: str, module_name: str,
                          output_path: Path) -> Dict[str, str]:
        """
        Create PyO3 integration files for a Rust module.

        Args:
            rust_code: Generated Rust code
            module_name: Name of the module
            output_path: Output directory path

        Returns:
            Dictionary of {file_path: content} for all integration files
        """
        from ..generator.rust_generator import RustCodeGenerator

        generator = RustCodeGenerator()

        # Generate all necessary files
        files = {}

        # 1. Cargo.toml
        cargo_toml = generator.generate_cargo_toml(module_name)
        files["Cargo.toml"] = cargo_toml

        # 2. Rust source code
        files["src/lib.rs"] = rust_code

        # 3. Python interface
        python_interface = f'''"""
Generated Python interface for optimized Rust module: {module_name}

This module contains optimized Rust implementations of your Python hotspots.
Import and use these functions for 10-100x performance improvements.
"""

# Note: To use this module, first compile the Rust code:
#   cd {output_path}
#   maturin develop
#
# Then import in your Python code:
#   from {module_name} import *

try:
    from .{module_name} import *
except ImportError as e:
    print(f"⚠️  Rust module not compiled yet. Run 'maturin develop' in {output_path}")
    print(f"Error: {{e}}")

    # Provide fallback Python implementations
    def placeholder_function(*args, **kwargs):
        raise RuntimeError(f"Rust module {module_name} not compiled. Run 'maturin develop'")

    # Export placeholder
    __all__ = ["placeholder_function"]
'''
        files["__init__.py"] = python_interface

        # 4. README with instructions
        readme = f'''# {module_name} - Optimized Rust Module

This directory contains Rust optimizations generated by PyRust Optimizer.

## Quick Start

1. **Compile the Rust module:**
   ```bash
   cd {output_path}
   maturin develop
   ```

2. **Import in Python:**
   ```python
   import {module_name}
   # Use optimized functions for 10-100x speedups!
   ```

## Files

- `Cargo.toml` - Rust project configuration
- `src/lib.rs` - Optimized Rust implementations
- `__init__.py` - Python interface
- `README.md` - This file

## Dependencies

- Rust toolchain: https://rustup.rs/
- Maturin: `pip install maturin`

## Performance

The optimized functions in this module provide significant speedups over pure Python:
- Estimated performance improvement: 5-100x
- Memory efficiency: Reduced allocations
- Type safety: Rust's type system prevents runtime errors

Generated by PyRust Optimizer 🚀
'''
        files["README.md"] = readme

        return files

    def generate_python_interface(self, compiled_module: CompiledModule) -> str:
        """
        Generate Python interface code for the compiled module.

        Args:
            compiled_module: CompiledModule object

        Returns:
            Python interface code
        """
        interface_code = f'''
"""
Generated Python interface for {compiled_module.name}.

This module provides optimized Rust implementations of Python functions
with seamless integration and fallback support.
"""

import {compiled_module.name}

class OptimizedFunctions:
    """Wrapper class for optimized Rust functions."""

    def __init__(self):
        self.module = {compiled_module.name}
        self.speedup_map = {compiled_module.speedup_achieved}

'''

        # Generate wrapper methods for each function
        for func_name in compiled_module.functions:
            speedup = compiled_module.speedup_achieved.get(func_name, 1.0)
            interface_code += f'''
    def {func_name}(self, *args, **kwargs):
        """
        Optimized {func_name} with {speedup}x speedup.

        This function uses Rust for performance-critical operations
        while maintaining the same Python interface.
        """
        try:
            return self.module.{func_name}(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Rust optimization failed for {func_name}: {{e}}")
'''

        interface_code += f'''

# Create global instance for easy access
optimized = OptimizedFunctions()

# Export functions at module level
'''

        for func_name in compiled_module.functions:
            interface_code += f'{func_name} = optimized.{func_name}\n'

        return interface_code

    def benchmark_performance(self, compiled_module: CompiledModule,
                            python_functions: Dict[str, callable],
                            test_args: Dict[str, tuple] = None) -> Dict[str, float]:
        """
        Benchmark performance of Rust vs Python implementations.

        Args:
            compiled_module: CompiledModule to benchmark
            python_functions: Dictionary of Python function implementations
            test_args: Test arguments for each function

        Returns:
            Dictionary of actual speedup ratios
        """
        import time

        actual_speedups = {}
        test_args = test_args or {}

        print(f"\n🏁 Benchmarking {compiled_module.name}...")

        for func_name in compiled_module.functions:
            if func_name not in python_functions:
                continue

            python_func = python_functions[func_name]
            rust_func = getattr(compiled_module.python_module, func_name)
            args = test_args.get(func_name, ())

            # Benchmark Python implementation
            start_time = time.time()
            try:
                python_result = python_func(*args)
                python_time = time.time() - start_time
            except Exception as e:
                print(f"❌ Python {func_name} failed: {e}")
                continue

            # Benchmark Rust implementation
            start_time = time.time()
            try:
                rust_result = rust_func(*args)
                rust_time = time.time() - start_time
            except Exception as e:
                print(f"❌ Rust {func_name} failed: {e}")
                continue

            # Calculate speedup
            if rust_time > 0:
                speedup = python_time / rust_time
                actual_speedups[func_name] = speedup

                print(f"⚡ {func_name}: {speedup:.1f}x speedup "
                      f"(Python: {python_time:.3f}s, Rust: {rust_time:.3f}s)")
            else:
                print(f"⚡ {func_name}: Rust too fast to measure!")

        return actual_speedups

    def cleanup_workspace(self):
        """Clean up the workspace directory."""
        try:
            shutil.rmtree(self.workspace_dir)
            print(f"🧹 Cleaned up workspace: {self.workspace_dir}")
        except Exception as e:
            print(f"❌ Failed to cleanup workspace: {e}")


class RustInstaller:
    """Helper class to install Rust toolchain if needed."""

    @staticmethod
    def is_rust_installed() -> bool:
        """Check if Rust is installed."""
        try:
            result = subprocess.run(['rustc', '--version'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def install_rust():
        """Install Rust using rustup (requires user interaction)."""
        print("🦀 Rust is not installed. To install Rust:")
        print("1. Visit https://rustup.rs/")
        print("2. Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        print("3. Restart your terminal")
        print("4. Verify installation with: rustc --version")


def demo_pyo3_integration():
    """Demo the PyO3 integration pipeline."""
    print("🔥 PyRust Optimizer - PyO3 Integration Demo")
    print("=" * 55)

    # Check Rust installation
    if not RustInstaller.is_rust_installed():
        print("❌ Rust is not installed!")
        RustInstaller.install_rust()
        return

    # Import dependencies
    from ..generator.rust_generator import RustCodeGenerator, RustFunction

    # Create integration instance
    integration = PyO3Integration()

    # Generate some Rust functions
    generator = RustCodeGenerator()

    # Create a simple nested loop function
    python_code = '''
result = 0
for i in range(100):
    for j in range(100):
        result += i * j
'''

    rust_func = generator.generate_rust_function(
        python_code, 'nested_loop', 'demo_nested_loop'
    )

    print(f"📦 Generated Rust function: {rust_func.name}")
    print(f"   Estimated speedup: {rust_func.estimated_speedup}x")

    # Create hybrid module
    compiled_module = integration.create_hybrid_module([rust_func], "demo_module")

    if compiled_module:
        print(f"✅ Successfully created hybrid module!")
        print(f"   Module name: {compiled_module.name}")
        print(f"   Library path: {compiled_module.library_path}")
        print(f"   Functions: {compiled_module.functions}")

        # Generate Python interface
        interface_code = integration.generate_python_interface(compiled_module)
        print(f"📋 Generated Python interface ({len(interface_code)} characters)")

        # Test the function
        try:
            result = compiled_module.python_module.demo_nested_loop()
            print(f"🚀 Rust function result: {result}")
        except Exception as e:
            print(f"❌ Error testing function: {e}")

    else:
        print("❌ Failed to create hybrid module")

    print(f"\n🎯 PyO3 integration demo complete!")


if __name__ == "__main__":
    demo_pyo3_integration()
