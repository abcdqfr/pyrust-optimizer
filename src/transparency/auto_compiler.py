"""
AutoCompiler for PyRust Optimizer Transparent Integration.

This module handles automatic compilation of Rust modules with smart
virtual environment detection and error handling.
"""

import os
import sys
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import shutil


class AutoCompiler:
    """
    Handles automatic compilation of Rust modules with smart environment management.

    This class detects virtual environments, creates them if needed, and compiles
    Rust modules using maturin with proper error handling and fallback mechanisms.
    """

    def __init__(self):
        self.temp_venv_dir = None
        self.original_env = None

    def compile_with_venv_detection(self, rust_project_path: str) -> Tuple[bool, str]:
        """
        Smart compilation with virtual environment handling.

        Args:
            rust_project_path: Path to the Rust project directory

        Returns:
            (success: bool, message: str) - compilation result and status message
        """
        project_path = Path(rust_project_path)
        if not project_path.exists():
            return False, f"Rust project path does not exist: {rust_project_path}"

        try:
            # Step 1: Detect or create virtual environment
            venv_info = self._detect_or_create_venv(project_path)
            if not venv_info['success']:
                return False, venv_info['message']

            # Step 2: Install maturin if needed
            maturin_result = self._ensure_maturin_installed(venv_info['python_executable'])
            if not maturin_result['success']:
                return False, maturin_result['message']

            # Step 3: Compile with maturin
            compile_result = self._compile_rust_module(
                project_path,
                venv_info['python_executable'],
                venv_info['environment']
            )

            return compile_result['success'], compile_result['message']

        except Exception as e:
            return False, f"Compilation failed with error: {e}"
        finally:
            # Clean up temporary environment if created
            self._cleanup_temp_venv()

    def _detect_or_create_venv(self, project_path: Path) -> Dict[str, any]:
        """Detect existing virtual environment or create a temporary one."""

        # Check for existing virtual environment
        venv_result = self._detect_existing_venv()
        if venv_result['found']:
            return {
                'success': True,
                'message': f"Using existing virtual environment: {venv_result['path']}",
                'python_executable': venv_result['python'],
                'environment': venv_result['env']
            }

        # Create temporary virtual environment
        temp_result = self._create_temp_venv(project_path.parent)
        return temp_result

    def _detect_existing_venv(self) -> Dict[str, any]:
        """Detect if we're already in a virtual environment."""

        # Check VIRTUAL_ENV environment variable
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            python_exe = os.path.join(virtual_env, 'bin', 'python')
            if os.path.exists(python_exe):
                return {
                    'found': True,
                    'path': virtual_env,
                    'python': python_exe,
                    'env': os.environ.copy()
                }

        # Check CONDA_PREFIX environment variable
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            python_exe = os.path.join(conda_prefix, 'bin', 'python')
            if os.path.exists(python_exe):
                return {
                    'found': True,
                    'path': conda_prefix,
                    'python': python_exe,
                    'env': os.environ.copy()
                }

        # Check for local .venv directory
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            venv_path = parent / '.venv'
            if venv_path.exists():
                python_exe = venv_path / 'bin' / 'python'
                if python_exe.exists():
                    env = os.environ.copy()
                    env['VIRTUAL_ENV'] = str(venv_path)
                    env['PATH'] = f"{venv_path / 'bin'}:{env.get('PATH', '')}"
                    return {
                        'found': True,
                        'path': str(venv_path),
                        'python': str(python_exe),
                        'env': env
                    }

        return {'found': False}

    def _create_temp_venv(self, base_path: Path) -> Dict[str, any]:
        """Create a temporary virtual environment."""
        try:
            # Create temporary directory for venv
            self.temp_venv_dir = tempfile.mkdtemp(prefix='pyrust_venv_')
            venv_path = Path(self.temp_venv_dir) / 'venv'

            # Create virtual environment
            venv.create(venv_path, with_pip=True)

            python_exe = venv_path / 'bin' / 'python'
            if not python_exe.exists():
                # Windows
                python_exe = venv_path / 'Scripts' / 'python.exe'

            if not python_exe.exists():
                return {
                    'success': False,
                    'message': "Failed to create virtual environment"
                }

            # Set up environment variables
            env = os.environ.copy()
            env['VIRTUAL_ENV'] = str(venv_path)
            bin_path = venv_path / 'bin' if (venv_path / 'bin').exists() else venv_path / 'Scripts'
            env['PATH'] = f"{bin_path}:{env.get('PATH', '')}"

            return {
                'success': True,
                'message': f"Created temporary virtual environment: {venv_path}",
                'python_executable': str(python_exe),
                'environment': env
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to create virtual environment: {e}"
            }

    def _ensure_maturin_installed(self, python_executable: str) -> Dict[str, any]:
        """Ensure maturin is installed in the virtual environment."""
        try:
            # Check if maturin is already installed
            result = subprocess.run(
                [python_executable, '-c', 'import maturin'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': "Maturin already available"
                }

            # Install maturin
            print("📦 Installing maturin in virtual environment...")
            result = subprocess.run(
                [python_executable, '-m', 'pip', 'install', 'maturin'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': "Successfully installed maturin"
                }
            else:
                return {
                    'success': False,
                    'message': f"Failed to install maturin: {result.stderr}"
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'message': "Timeout while installing maturin"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error installing maturin: {e}"
            }

    def _compile_rust_module(self, project_path: Path, python_executable: str, env: Dict[str, str]) -> Dict[str, any]:
        """Compile the Rust module using maturin."""
        try:
            print(f"🔥 Compiling Rust module in {project_path}...")

            # Run maturin develop
            result = subprocess.run(
                [python_executable, '-m', 'maturin', 'develop'],
                cwd=project_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': "Successfully compiled Rust module"
                }
            else:
                return {
                    'success': False,
                    'message': f"Compilation failed: {result.stderr}"
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'message': "Compilation timed out (5 minutes)"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Compilation error: {e}"
            }

    def _cleanup_temp_venv(self):
        """Clean up temporary virtual environment."""
        if self.temp_venv_dir and os.path.exists(self.temp_venv_dir):
            try:
                shutil.rmtree(self.temp_venv_dir)
                print(f"🧹 Cleaned up temporary venv: {self.temp_venv_dir}")
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temp venv: {e}")
            finally:
                self.temp_venv_dir = None

    def check_rust_toolchain(self) -> Dict[str, any]:
        """Check if Rust toolchain is available."""
        try:
            result = subprocess.run(
                ['cargo', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return {
                    'available': True,
                    'version': result.stdout.strip(),
                    'message': "Rust toolchain available"
                }
            else:
                return {
                    'available': False,
                    'message': "Cargo command failed"
                }

        except FileNotFoundError:
            return {
                'available': False,
                'message': "Rust toolchain not found. Install from: https://rustup.rs/"
            }
        except Exception as e:
            return {
                'available': False,
                'message': f"Error checking Rust toolchain: {e}"
            }

    def install_dependencies(self, project_path: Path, python_executable: str, env: Dict[str, str]) -> Dict[str, any]:
        """Install additional Python dependencies if needed."""
        requirements_file = project_path / 'requirements.txt'
        if not requirements_file.exists():
            return {'success': True, 'message': "No requirements.txt found"}

        try:
            result = subprocess.run(
                [python_executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': "Successfully installed dependencies"
                }
            else:
                return {
                    'success': False,
                    'message': f"Failed to install dependencies: {result.stderr}"
                }

        except Exception as e:
            return {
                'success': False,
                'message': f"Error installing dependencies: {e}"
            }


def demo_auto_compiler():
    """Demo the auto compiler functionality."""

    # Create a simple Rust project for testing
    test_project = Path("test_rust_project")
    test_project.mkdir(exist_ok=True)

    # Create Cargo.toml
    cargo_toml = test_project / "Cargo.toml"
    cargo_toml.write_text('''[package]
name = "test_module"
version = "0.1.0"
edition = "2021"

[lib]
name = "test_module"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
''')

    # Create src directory and lib.rs
    src_dir = test_project / "src"
    src_dir.mkdir(exist_ok=True)

    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text('''use pyo3::prelude::*;

#[pyfunction]
fn test_function() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

#[pymodule]
fn test_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    Ok(())
}
''')

    # Test auto compiler
    compiler = AutoCompiler()

    print("🔧 Testing Rust toolchain...")
    rust_check = compiler.check_rust_toolchain()
    print(f"Rust available: {rust_check['available']} - {rust_check['message']}")

    if rust_check['available']:
        print("\n🚀 Testing compilation...")
        success, message = compiler.compile_with_venv_detection(str(test_project))
        print(f"Compilation result: {success}")
        print(f"Message: {message}")
    else:
        print("⚠️ Skipping compilation test - Rust not available")

    # Cleanup
    try:
        shutil.rmtree(test_project)
        print("🧹 Cleaned up test project")
    except Exception as e:
        print(f"⚠️ Could not clean up test project: {e}")


if __name__ == "__main__":
    demo_auto_compiler()
