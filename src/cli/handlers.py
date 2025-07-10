"""
PyRust Optimizer - Command Handlers

These handlers implement the actual CLI command logic that's automatically
wired up by the Infrastructure as Code configuration in commands.py.
"""

import click
from pathlib import Path
from typing import Optional

from ..profiler.hotspot_detector import HeuristicHotspotDetector
from ..analyzer.ast_analyzer import ASTAnalyzer
from ..generator.rust_generator import RustCodeGenerator
from ..runtime.pyo3_integration import PyO3Integration


class CommandHandlers:
    """Handles CLI command implementations."""

    def __init__(self):
        self.hotspot_detector = HeuristicHotspotDetector()
        self.ast_analyzer = ASTAnalyzer()
        self.rust_generator = RustCodeGenerator()
        self.pyo3_integration = PyO3Integration()

    def handle_optimize(self, python_file: str, output: Optional[str],
                       module_name: str, min_speedup: float,
                       dry_run: bool, verbose: bool):
        """🔥 Handle the optimize command."""

        click.echo(f"🚀 PyRust Optimizer - Optimizing {python_file}")
        click.echo("=" * 50)

        # Validate input file
        input_path = Path(python_file)
        if not input_path.exists():
            click.echo(f"❌ Error: File {python_file} not found!")
            return 1

        # Set up output directory
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / "optimized"

        output_path.mkdir(exist_ok=True)

        try:
            # Step 1: Read and parse Python code
            if verbose:
                click.echo(f"📖 Reading {input_path}")

            python_code = input_path.read_text()

            # Step 2: Detect hotspots
            if verbose:
                click.echo("🔍 Detecting performance hotspots...")

            hotspots = self.hotspot_detector.detect_hotspots(python_file)

            if not hotspots:
                click.echo("✅ No optimization opportunities found!")
                return 0

            # Filter by minimum speedup
            filtered_hotspots = [
                h for h in hotspots
                if h.optimization_potential >= min_speedup / 10  # Convert to 0-1 scale
            ]

            if not filtered_hotspots:
                click.echo(f"ℹ️ No hotspots meet minimum speedup threshold of {min_speedup}x")
                return 0

            click.echo(f"🎯 Found {len(filtered_hotspots)} optimization targets:")
            for i, hotspot in enumerate(filtered_hotspots, 1):
                click.echo(f"   {i}. {hotspot.function_name} "
                          f"(estimated {hotspot.optimization_potential * 10:.1f}x speedup)")

            # Calculate total potential speedup
            max_speedup = max(h.optimization_potential * 10 for h in filtered_hotspots)
            avg_speedup = sum(h.optimization_potential * 10 for h in filtered_hotspots) / len(filtered_hotspots)

            click.echo(f"\n📊 Performance Improvement Summary:")
            click.echo(f"   🚀 Maximum estimated speedup: {max_speedup:.1f}x")
            click.echo(f"   📈 Average estimated speedup: {avg_speedup:.1f}x")
            click.echo(f"   🎯 Total optimization targets: {len(filtered_hotspots)}")

            if dry_run:
                click.echo("\n🏃 Dry run - no code generated")
                return 0

            # 🎯 INTERACTIVE PROMPT - Let user decide if speedups are worth it
            click.echo(f"\n⚡ Ready to generate optimized Rust code!")
            click.echo(f"   📦 This will create a complete Rust module with PyO3 bindings")
            click.echo(f"   ⏱️  Compilation time: ~30-60 seconds")
            click.echo(f"   🎯 Expected performance gain: {max_speedup:.1f}x speedup")

            proceed = click.confirm(f"\nContinue with Rust optimization generation?", default=True)

            if not proceed:
                click.echo("🛑 Optimization cancelled by user")
                click.echo("💡 Use --dry-run flag to analyze without prompts")
                return 0

            click.echo(f"\n🔥 Proceeding with optimization generation...")

            # Step 3: Analyze AST
            if verbose:
                click.echo("🌳 Analyzing Python AST...")

            ast_info = self.ast_analyzer.analyze_code(python_code)

            # Step 4: Generate Rust code
            if verbose:
                click.echo("⚙️ Generating optimized Rust code...")

            rust_code = self.rust_generator.generate_rust_module(
                filtered_hotspots,
                module_name,
                ast_info
            )

            # Step 5: Create PyO3 integration
            if verbose:
                click.echo("🔗 Setting up Python-Rust integration...")

            integration_files = self.pyo3_integration.create_integration(
                rust_code,
                module_name,
                output_path
            )

            # Step 6: Write all files
            for file_path, content in integration_files.items():
                full_path = output_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

                if verbose:
                    click.echo(f"📝 Created {full_path}")

            # Auto-compile the Rust module
            click.echo(f"\n🔥 Auto-compiling Rust module...")

            try:
                import subprocess
                result = subprocess.run(
                    ['maturin', 'develop'],
                    cwd=output_path,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    click.echo("✅ Rust module compiled successfully!")
                    click.echo(f"🚀 Module '{module_name}' ready for immediate import!")

                    # Success summary with seamless integration
                    click.echo(f"\n🎉 Optimization complete!")
                    click.echo(f"📁 Output directory: {output_path}")
                    click.echo(f"🦀 Generated Rust module: {module_name}")
                    click.echo(f"⚡ Estimated speedup: {max(h.optimization_potential for h in filtered_hotspots) * 10:.1f}x")
                    click.echo(f"\n💡 Usage (works immediately):")
                    click.echo(f"   import sys; sys.path.append('{output_path}')")
                    click.echo(f"   import {module_name}")
                    click.echo(f"   # Use optimized functions with {max(h.optimization_potential for h in filtered_hotspots) * 10:.1f}x speedup!")

                else:
                    click.echo("⚠️  Auto-compilation failed. Manual steps required:")
                    click.echo(f"   cd {output_path}")
                    click.echo("   maturin develop")

            except FileNotFoundError:
                click.echo("⚠️  maturin not found. Install with: pip install maturin")
                click.echo("\n🛠️ Manual steps:")
                click.echo(f"   cd {output_path}")
                click.echo("   maturin develop")
            except Exception as e:
                click.echo(f"⚠️  Auto-compilation error: {e}")
                click.echo("\n🛠️ Manual steps:")
                click.echo(f"   cd {output_path}")
                click.echo("   maturin develop")

            return 0

        except Exception as e:
            click.echo(f"❌ Error during optimization: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1

    def handle_analyze(self, python_file: str, threshold: float, verbose: bool):
        """🔍 Handle the analyze command."""

        click.echo(f"🔍 PyRust Optimizer - Analyzing {python_file}")
        click.echo("=" * 50)

        # Validate input file
        input_path = Path(python_file)
        if not input_path.exists():
            click.echo(f"❌ Error: File {python_file} not found!")
            return 1

        try:
            # Read and analyze
            hotspots = self.hotspot_detector.detect_hotspots(python_file)

            # Filter by confidence threshold
            filtered_hotspots = [
                h for h in hotspots
                if h.confidence_score >= threshold
            ]

            if not filtered_hotspots:
                click.echo(f"✅ No hotspots found above confidence threshold of {threshold}")
                return 0

            # Display results
            click.echo(f"🎯 Found {len(filtered_hotspots)} potential optimization targets:\n")

            for i, hotspot in enumerate(filtered_hotspots, 1):
                click.echo(f"📊 Hotspot #{i}: {hotspot.function_name}")
                click.echo(f"   📍 Line {hotspot.line_number}")
                click.echo(f"   🎯 Type: {hotspot.hotspot_type}")
                click.echo(f"   🔥 Confidence: {hotspot.confidence_score:.2f}")
                click.echo(f"   ⚡ Est. Speedup: {hotspot.optimization_potential * 10:.1f}x")

                if verbose:
                    click.echo(f"   📝 Reason: {hotspot.description}")

                click.echo()

            # Summary
            total_speedup = sum(h.optimization_potential for h in filtered_hotspots) * 10
            avg_confidence = sum(h.confidence_score for h in filtered_hotspots) / len(filtered_hotspots)

            click.echo(f"📈 Analysis Summary:")
            click.echo(f"   🎯 Hotspots: {len(filtered_hotspots)}")
            click.echo(f"   ⚡ Total potential speedup: {total_speedup:.1f}x")
            click.echo(f"   🔥 Average confidence: {avg_confidence:.2f}")

            return 0

        except Exception as e:
            click.echo(f"❌ Error during analysis: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1

    def handle_setup(self, workspace: Optional[str]):
        """🛠️ Handle the setup command."""

        click.echo("🛠️ PyRust Optimizer - Environment Setup")
        click.echo("=" * 40)

        # Check dependencies
        dependencies = [
            ("Python 3.8+", self._check_python),
            ("Rust toolchain", self._check_rust),
            ("Maturin", self._check_maturin),
            ("Tree-sitter", self._check_tree_sitter)
        ]

        all_good = True
        for dep_name, check_func in dependencies:
            if check_func():
                click.echo(f"✅ {dep_name}")
            else:
                click.echo(f"❌ {dep_name}")
                all_good = False

        if workspace:
            workspace_path = Path(workspace)
            workspace_path.mkdir(parents=True, exist_ok=True)
            click.echo(f"📁 Workspace created: {workspace_path}")

        if all_good:
            click.echo("\n🎉 Environment setup complete!")
            click.echo("🚀 Ready to optimize Python with Rust!")
        else:
            click.echo("\n⚠️ Please install missing dependencies")
            click.echo("📖 See README.txt for installation instructions")

        return 0 if all_good else 1

    def handle_demo(self):
        """🎮 Handle the demo command."""

        click.echo("🎮 PyRust Optimizer - Live Demonstration")
        click.echo("=" * 45)

        # Create demo Python code
        demo_code = '''
def fibonacci_slow(n):
    """Slow recursive Fibonacci - perfect optimization target!"""
    if n <= 1:
        return n
    return fibonacci_slow(n-1) + fibonacci_slow(n-2)

def matrix_multiply(a, b):
    """Nested loop matrix multiplication - another great target!"""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]

    return result

def process_data(data):
    """I/O and processing - moderate optimization potential."""
    processed = []
    for item in data:
        if item > 0:
            processed.append(item ** 2)
    return processed
'''

        click.echo("📝 Demo Python code:")
        click.echo("-" * 20)
        click.echo(demo_code)

        click.echo("\n🔍 Analyzing for hotspots...")

        # Create a temporary file for analysis
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(demo_code)
            temp_file = f.name

        try:
            # Analyze the demo code
            hotspots = self.hotspot_detector.detect_hotspots(temp_file)
        finally:
            # Clean up temp file
            os.unlink(temp_file)

        click.echo(f"🎯 Found {len(hotspots)} optimization opportunities:")

        for i, hotspot in enumerate(hotspots, 1):
            click.echo(f"\n🔥 Hotspot #{i}: {hotspot.function_name}")
            click.echo(f"   📍 Type: {hotspot.hotspot_type}")
            click.echo(f"   ⚡ Estimated speedup: {hotspot.optimization_potential * 10:.1f}x")
            click.echo(f"   🔥 Confidence: {hotspot.confidence_score:.2f}")
            click.echo(f"   💡 {hotspot.description}")

        click.echo(f"\n🎉 Demo complete!")
        click.echo(f"💡 Try: pyrust optimize your_slow_script.py")

        return 0

    def _check_python(self) -> bool:
        """Check if Python 3.8+ is available."""
        import sys
        return sys.version_info >= (3, 8)

    def _check_rust(self) -> bool:
        """Check if Rust toolchain is available."""
        import subprocess
        try:
            result = subprocess.run(['rustc', '--version'],
                                   capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _check_maturin(self) -> bool:
        """Check if Maturin is available."""
        import subprocess
        try:
            result = subprocess.run(['maturin', '--version'],
                                   capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _check_tree_sitter(self) -> bool:
        """Check if Tree-sitter Python bindings are available."""
        try:
            import tree_sitter
            return True
        except ImportError:
            return False
