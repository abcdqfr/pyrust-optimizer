"""
PyRust Optimizer - Command Line Interface

A revolutionary CLI tool for selective Python→Rust optimization using Click.
Achieve 10-100x performance gains while preserving Python ecosystem compatibility.
"""

import click
import os
import sys
from pathlib import Path
from typing import List, Optional
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.hotspot_detector import HeuristicHotspotDetector, Hotspot
from generator.rust_generator import RustCodeGenerator, RustFunction
from runtime.pyo3_integration import PyO3Integration, CompiledModule


@click.group()
@click.version_option(version='0.1.0', prog_name='PyRust Optimizer')
def cli():
    """
    🚀 PyRust Optimizer - Revolutionary Python→Rust selective optimization.

    Transform your Python bottlenecks into blazing-fast Rust while keeping
    everything else in readable Python. Achieve 10-100x speedups with zero
    ecosystem disruption.
    """
    pass


@cli.command()
@click.argument('python_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', '-o', help='Output directory for optimized code')
@click.option('--module-name', '-m', default='optimized', help='Name for generated module')
@click.option('--min-speedup', '-s', default=5.0, type=float,
              help='Minimum estimated speedup to optimize (default: 5.0x)')
@click.option('--dry-run', '-d', is_flag=True,
              help='Show what would be optimized without generating code')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def optimize(python_file: str, output: Optional[str], module_name: str,
            min_speedup: float, dry_run: bool, verbose: bool):
    """
    🔥 Optimize a Python file by converting hotspots to Rust.

    Analyzes PYTHON_FILE for performance bottlenecks and generates
    optimized Rust code for the most critical paths.

    Example:
        pyrust optimize my_slow_script.py --output ./optimized/
    """
    # Import here to avoid circular imports
    from .handlers import CommandHandlers

    # Delegate to the working implementation
    handlers = CommandHandlers()
    exit_code = handlers.handle_optimize(
        python_file, output, module_name, min_speedup, dry_run, verbose
    )

    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.argument('python_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--threshold', '-t', default=0.5, type=float,
              help='Confidence threshold for hotspot detection (0.0-1.0)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed analysis')
def analyze(python_file: str, threshold: float, verbose: bool):
    """
    🔍 Analyze a Python file for performance hotspots.

    Identifies potential optimization targets without generating code.
    Useful for understanding where your bottlenecks are.

    Example:
        pyrust analyze my_script.py --threshold 0.7 --verbose
    """
    # Import here to avoid circular imports
    from .handlers import CommandHandlers

    # Delegate to the working implementation
    handlers = CommandHandlers()
    exit_code = handlers.handle_analyze(python_file, threshold, verbose)

    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.option('--workspace', '-w', help='Custom workspace directory')
def setup(workspace: Optional[str]):
    """
    🛠️ Set up PyRust Optimizer development environment.

    Checks dependencies, creates workspace, and verifies installation.
    """
    click.echo("🛠️ PyRust Optimizer - Environment Setup")
    click.echo("=" * 45)

    # Check Python
    click.echo("🐍 Checking Python installation...")
    python_version = sys.version_info
    if python_version >= (3, 8):
        click.echo(f"   ✅ Python {python_version.major}.{python_version.minor} detected")
    else:
        click.echo(f"   ❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return

    # Check Rust
    click.echo("🦀 Checking Rust installation...")
    from runtime.pyo3_integration import RustInstaller

    if RustInstaller.is_rust_installed():
        click.echo("   ✅ Rust toolchain detected")
    else:
        click.echo("   ❌ Rust not found")
        click.echo("   Install from: https://rustup.rs/")
        return

    # Create workspace
    workspace_dir = workspace or "/tmp/pyrust_workspace"
    click.echo(f"📁 Creating workspace at {workspace_dir}...")

    try:
        os.makedirs(workspace_dir, exist_ok=True)
        click.echo("   ✅ Workspace created")
    except Exception as e:
        click.echo(f"   ❌ Failed to create workspace: {e}")
        return

    # Test compilation
    click.echo("🔧 Testing Rust compilation...")

    integration = PyO3Integration(workspace_dir)
    generator = RustCodeGenerator()

    # Create a simple test function
    test_code = "for i in range(10): pass"
    test_func = generator.generate_rust_function(test_code, 'large_iteration', 'test_func')

    compiled_module = integration.create_hybrid_module([test_func], "test_module")

    if compiled_module:
        click.echo("   ✅ Rust compilation successful")
        integration.cleanup_workspace()
    else:
        click.echo("   ❌ Rust compilation failed")
        return

    click.echo("\n🎉 Setup complete! PyRust Optimizer is ready to use.")
    click.echo("\n🚀 Try: pyrust optimize your_script.py")


@cli.command()
def demo():
    """
    🎮 Run PyRust Optimizer demonstration.

    Shows the complete optimization pipeline with example code.
    """
    click.echo("🎮 PyRust Optimizer - Live Demo")
    click.echo("=" * 40)

    # Create demo Python file
    demo_code = '''
def slow_nested_loop():
    result = 0
    for i in range(1000):
        for j in range(1000):
            result += i * j
    return result

def mathematical_computation():
    import math
    result = 0
    for i in range(10000):
        result += math.sin(i) + math.cos(i)
    return result

if __name__ == "__main__":
    print("Running slow Python code...")
    result1 = slow_nested_loop()
    result2 = mathematical_computation()
    print(f"Results: {result1}, {result2}")
'''

    demo_file = Path("/tmp/demo_script.py")
    demo_file.write_text(demo_code)

    click.echo(f"📝 Created demo script: {demo_file}")
    click.echo("\n🔍 Analyzing for hotspots...")

    # Run analysis
    from click.testing import CliRunner
    runner = CliRunner()

    # Analyze
    result = runner.invoke(analyze, [str(demo_file), '--verbose'])
    click.echo(result.output)

    # Optimize
    click.echo("\n⚡ Running optimization...")
    result = runner.invoke(optimize, [str(demo_file), '--dry-run'])
    click.echo(result.output)

    click.echo("\n🎯 Demo complete!")
    click.echo("Try running: pyrust optimize /tmp/demo_script.py")


if __name__ == '__main__':
    cli()
