"""
Transparent CLI for PyRust Optimizer.

This module provides the main entry point for transparent in-place optimization
that modifies Python files to use Rust optimizations while preserving the
exact same import and function call interface.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import tempfile
import importlib.util

# Import our transparency components
from .auto_compiler import AutoCompiler
from .import_guards import GuardedFunctionGenerator
from ..profiler.hotspot_detector import HotspotDetector
from ..analyzer.ast_analyzer import ASTAnalyzer
from ..generator.rust_generator import RustCodeGenerator
from ..runtime.pyo3_integration import PyO3Integration


class TransparentOptimizer:
    """
    Main class for transparent Python optimization.

    This class orchestrates the entire workflow of detecting hotspots,
    generating Rust code, compiling it, and modifying the original
    Python file to use optimizations transparently.
    """

    def __init__(self):
        self.hotspot_detector = HotspotDetector()
        self.ast_analyzer = ASTAnalyzer()
        self.rust_generator = RustCodeGenerator()
        self.pyo3_integration = PyO3Integration()
        self.auto_compiler = AutoCompiler()
        self.function_generator = GuardedFunctionGenerator()

        self.backup_suffix = ".pyrust_backup"
        self.rust_module_name = "_pyrust_optimized"

    def optimize_in_place(self, python_file: str,
                         min_speedup: float = 2.0,
                         force: bool = False,
                         verbose: bool = False) -> Dict[str, Any]:
        """
        Main entry point for transparent optimization.

        Args:
            python_file: Path to Python file to optimize
            min_speedup: Minimum speedup threshold for optimization
            force: Force optimization even if already optimized
            verbose: Enable verbose output

        Returns:
            Dictionary with optimization results and status
        """

        result = {
            'success': False,
            'message': '',
            'optimized_functions': [],
            'estimated_speedup': 0.0,
            'backup_created': False,
            'rust_compiled': False,
            'file_modified': False
        }

        try:
            file_path = Path(python_file)

            # Step 1: Validate input
            validation_result = self._validate_input(file_path, force)
            if not validation_result['valid']:
                result['message'] = validation_result['message']
                return result

            if verbose:
                print(f"🎯 Starting transparent optimization of {python_file}")

            # Step 2: Create backup
            backup_result = self._create_backup(file_path)
            if not backup_result['success']:
                result['message'] = backup_result['message']
                return result
            result['backup_created'] = True

            if verbose:
                print(f"📁 Backup created: {backup_result['backup_path']}")

            # Step 3: Detect hotspots
            hotspots_result = self._detect_hotspots(file_path, min_speedup, verbose)
            if not hotspots_result['success']:
                result['message'] = hotspots_result['message']
                return result

            hotspots = hotspots_result['hotspots']
            if not hotspots:
                result['message'] = f"No optimization opportunities found (min speedup: {min_speedup}x)"
                return result

            if verbose:
                print(f"🔍 Found {len(hotspots)} optimization targets")

            # Step 4: Generate and compile Rust code
            rust_result = self._generate_and_compile_rust(file_path, hotspots, verbose)
            if not rust_result['success']:
                result['message'] = rust_result['message']
                return result
            result['rust_compiled'] = True

            if verbose:
                print("🦀 Rust module compiled successfully")

            # Step 5: Modify original file with transparent guards
            modify_result = self._modify_file_with_guards(
                file_path, hotspots, rust_result['module_path'], verbose
            )
            if not modify_result['success']:
                result['message'] = modify_result['message']
                return result
            result['file_modified'] = True

            # Step 6: Validate the modification works
            validation_result = self._validate_optimization(file_path, verbose)
            if not validation_result['success']:
                result['message'] = f"Optimization validation failed: {validation_result['message']}"
                return result

            # Success!
            result.update({
                'success': True,
                'message': 'Transparent optimization completed successfully',
                'optimized_functions': [h.function_name for h in hotspots],
                'estimated_speedup': max(h.optimization_potential for h in hotspots) * 10,
                'rust_module_path': rust_result['module_path']
            })

            if verbose:
                print("🎉 Transparent optimization completed!")
                print(f"✅ Functions optimized: {', '.join(result['optimized_functions'])}")
                print(f"⚡ Estimated speedup: {result['estimated_speedup']:.1f}x")

            return result

        except Exception as e:
            # Clean up on error
            self._cleanup_on_error(file_path, result)
            result['message'] = f"Optimization failed: {e}"
            if verbose:
                import traceback
                print(f"❌ Error during optimization:")
                traceback.print_exc()
            return result

    def _validate_input(self, file_path: Path, force: bool) -> Dict[str, Any]:
        """Validate the input file and optimization state."""

        if not file_path.exists():
            return {'valid': False, 'message': f"File not found: {file_path}"}

        if not file_path.suffix == '.py':
            return {'valid': False, 'message': f"Not a Python file: {file_path}"}

        # Check if already optimized
        if not force and self._is_already_optimized(file_path):
            return {
                'valid': False,
                'message': f"File already optimized. Use --force to re-optimize: {file_path}"
            }

        return {'valid': True, 'message': 'Input validation passed'}

    def _is_already_optimized(self, file_path: Path) -> bool:
        """Check if file has already been optimized."""
        try:
            content = file_path.read_text()
            return "PyRust Optimizer Header" in content
        except:
            return False

    def _create_backup(self, file_path: Path) -> Dict[str, Any]:
        """Create backup of original file."""
        try:
            backup_path = file_path.with_suffix(f"{file_path.suffix}{self.backup_suffix}")
            shutil.copy2(file_path, backup_path)
            return {
                'success': True,
                'backup_path': str(backup_path),
                'message': 'Backup created successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to create backup: {e}"
            }

    def _detect_hotspots(self, file_path: Path, min_speedup: float, verbose: bool) -> Dict[str, Any]:
        """Detect optimization hotspots in the Python file."""
        try:
            hotspots = self.hotspot_detector.detect_hotspots(str(file_path))

            # Filter by minimum speedup threshold
            filtered_hotspots = []
            for hotspot in hotspots:
                estimated_speedup = hotspot.optimization_potential * 10
                if estimated_speedup >= min_speedup:
                    filtered_hotspots.append(hotspot)

            if verbose and filtered_hotspots:
                print("\n🎯 Optimization targets:")
                for i, hotspot in enumerate(filtered_hotspots, 1):
                    speedup = hotspot.optimization_potential * 10
                    print(f"   {i}. {hotspot.function_name} (estimated {speedup:.1f}x speedup)")

            return {
                'success': True,
                'hotspots': filtered_hotspots,
                'message': f'Found {len(filtered_hotspots)} optimization targets'
            }

        except Exception as e:
            return {
                'success': False,
                'hotspots': [],
                'message': f'Hotspot detection failed: {e}'
            }

    def _generate_and_compile_rust(self, file_path: Path, hotspots: List, verbose: bool) -> Dict[str, Any]:
        """Generate Rust code and compile it."""
        try:
            # Create Rust module directory
            rust_dir = file_path.parent / self.rust_module_name
            rust_dir.mkdir(exist_ok=True)

            # Read original Python code for AST analysis
            original_code = file_path.read_text()
            ast_info = self.ast_analyzer.analyze_code(original_code)

            # Generate Rust code
            rust_code = self.rust_generator.generate_rust_module(
                hotspots, self.rust_module_name, ast_info
            )

            # Generate integration files
            integration_files = self.pyo3_integration.create_integration(
                rust_code, self.rust_module_name, rust_dir
            )

            # Write all files
            for file_path_rel, content in integration_files.items():
                full_path = rust_dir / file_path_rel
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            if verbose:
                print(f"📝 Generated Rust project in {rust_dir}")

            # Compile the Rust module
            success, message = self.auto_compiler.compile_with_venv_detection(str(rust_dir))

            if not success:
                return {
                    'success': False,
                    'message': f'Rust compilation failed: {message}'
                }

            return {
                'success': True,
                'module_path': str(rust_dir),
                'message': 'Rust module generated and compiled successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Rust generation/compilation failed: {e}'
            }

    def _modify_file_with_guards(self, file_path: Path, hotspots: List,
                               rust_module_path: str, verbose: bool) -> Dict[str, Any]:
        """Modify the original Python file with optimization guards."""
        try:
            # Read original content
            original_content = file_path.read_text()

            # Extract function information from hotspots
            optimized_functions = []
            for hotspot in hotspots:
                if hasattr(hotspot, 'function_name') and hotspot.function_name != 'global':
                    optimized_functions.append(hotspot.function_name)

            if not optimized_functions:
                return {
                    'success': False,
                    'message': 'No named functions found to optimize'
                }

            # Generate import header
            header = self.function_generator.generate_import_header(
                optimized_functions, self.rust_module_name
            )

            # Parse AST to find function definitions
            import ast
            tree = ast.parse(original_content)
            lines = original_content.splitlines()

            # Build modified content
            modified_lines = []
            processed_lines = set()

            # Add header at the beginning
            modified_lines.extend(header.splitlines())
            modified_lines.append("")

            # Process each function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in optimized_functions:
                    func_start = node.lineno - 1

                    # Find function end
                    func_end = len(lines)
                    func_indent = len(lines[func_start]) - len(lines[func_start].lstrip())

                    for i in range(func_start + 1, len(lines)):
                        line = lines[i]
                        if line.strip():
                            current_indent = len(line) - len(line.lstrip())
                            if current_indent <= func_indent and (line.strip().startswith('def ') or line.strip().startswith('class ')):
                                func_end = i
                                break

                    # Extract function info
                    signature = lines[func_start].strip()
                    params_match = signature.split('(', 1)[1].rsplit(')', 1)[0] if '(' in signature else ""

                    # Get docstring
                    docstring = f"Original implementation with Rust optimization for {node.name}"
                    if (node.body and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                        docstring = node.body[0].value.value

                    # Extract function body
                    body_lines = lines[func_start + 1:func_end]
                    original_body = '\n'.join(body_lines)

                    # Generate guarded function
                    guarded_func = self.function_generator.generate_guarded_function_code(
                        node.name, params_match, docstring, original_body, self.rust_module_name
                    )

                    modified_lines.extend(guarded_func.splitlines())
                    modified_lines.append("")

                    # Mark lines as processed
                    for i in range(func_start, func_end):
                        processed_lines.add(i)

            # Add remaining unprocessed lines
            for i, line in enumerate(lines):
                if i not in processed_lines:
                    modified_lines.append(line)

            # Add footer
            modified_lines.append("")
            modified_lines.append("# ===== End PyRust Optimizer =====")

            # Write modified content
            modified_content = '\n'.join(modified_lines)
            file_path.write_text(modified_content)

            if verbose:
                print(f"✏️ Modified {file_path} with optimization guards")

            return {
                'success': True,
                'message': 'File modified successfully with optimization guards'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'File modification failed: {e}'
            }

    def _validate_optimization(self, file_path: Path, verbose: bool) -> Dict[str, Any]:
        """Validate that the optimized file can still be imported and used."""
        try:
            # Try to import the modified module
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)

            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if verbose:
                    print("✅ Optimization validation passed - file can be imported")

                return {
                    'success': True,
                    'message': 'Optimization validation passed'
                }
            else:
                return {
                    'success': False,
                    'message': 'Could not create module spec for validation'
                }

        except Exception as e:
            return {
                'success': False,
                'message': f'Validation failed: {e}'
            }

    def _cleanup_on_error(self, file_path: Path, result: Dict[str, Any]):
        """Clean up files if optimization failed."""
        try:
            # Restore from backup if file was modified
            if result.get('file_modified') or result.get('backup_created'):
                backup_path = file_path.with_suffix(f"{file_path.suffix}{self.backup_suffix}")
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    print(f"🔄 Restored {file_path} from backup")

            # Clean up Rust module directory if created
            rust_dir = file_path.parent / self.rust_module_name
            if rust_dir.exists() and result.get('rust_compiled'):
                shutil.rmtree(rust_dir)
                print(f"🧹 Cleaned up Rust module directory")

        except Exception as e:
            print(f"⚠️ Warning: Cleanup failed: {e}")

    def rollback(self, python_file: str) -> Dict[str, Any]:
        """Rollback optimization to original state."""
        try:
            file_path = Path(python_file)
            backup_path = file_path.with_suffix(f"{file_path.suffix}{self.backup_suffix}")

            if not backup_path.exists():
                return {
                    'success': False,
                    'message': f'No backup found for {python_file}'
                }

            # Restore from backup
            shutil.copy2(backup_path, file_path)

            # Clean up Rust module
            rust_dir = file_path.parent / self.rust_module_name
            if rust_dir.exists():
                shutil.rmtree(rust_dir)

            return {
                'success': True,
                'message': f'Successfully rolled back {python_file}'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Rollback failed: {e}'
            }


def main():
    """Main CLI entry point for transparent optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PyRust Optimizer - Transparent Python→Rust Optimization"
    )
    parser.add_argument("file", help="Python file to optimize")
    parser.add_argument("--min-speedup", type=float, default=2.0,
                       help="Minimum speedup threshold (default: 2.0x)")
    parser.add_argument("--force", action="store_true",
                       help="Force optimization even if already optimized")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--rollback", action="store_true",
                       help="Rollback optimization to original state")

    args = parser.parse_args()

    optimizer = TransparentOptimizer()

    if args.rollback:
        result = optimizer.rollback(args.file)
        print(result['message'])
        return 0 if result['success'] else 1
    else:
        result = optimizer.optimize_in_place(
            args.file,
            min_speedup=args.min_speedup,
            force=args.force,
            verbose=args.verbose
        )

        print(result['message'])

        if result['success']:
            print(f"\n🎉 Transparent optimization completed!")
            print(f"📁 File: {args.file}")
            print(f"⚡ Estimated speedup: {result['estimated_speedup']:.1f}x")
            print(f"🎯 Functions optimized: {', '.join(result['optimized_functions'])}")
            print(f"\n💡 Your existing code now works exactly the same, but faster!")
            print(f"   No import changes needed - just use your functions normally.")

        return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
