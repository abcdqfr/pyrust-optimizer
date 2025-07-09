"""
PyRust Optimizer - Click Command Definitions (Infrastructure as Code)

This module defines the CLI structure using Click's decorators for
automatic command generation and infrastructure as code principles.
"""

import click
from typing import Optional
from pathlib import Path


# Define command configuration as data (Infrastructure as Code)
CLI_CONFIG = {
    'name': 'pyrust',
    'version': '0.1.0',
    'description': '🚀 PyRust Optimizer - Revolutionary Python→Rust selective optimization',
    'help': '''
Transform your Python bottlenecks into blazing-fast Rust while keeping
everything else in readable Python. Achieve 10-100x speedups with zero
ecosystem disruption.
    ''',
    'commands': {
        'optimize': {
            'help': '🔥 Optimize a Python file by converting hotspots to Rust',
            'description': '''
Analyzes PYTHON_FILE for performance bottlenecks and generates
optimized Rust code for the most critical paths.

Example: pyrust optimize my_slow_script.py --output ./optimized/
            ''',
            'arguments': [
                {
                    'name': 'python_file',
                    'type': 'path',
                    'required': True,
                    'help': 'Python file to optimize'
                }
            ],
            'options': [
                {
                    'name': '--output',
                    'short': '-o',
                    'help': 'Output directory for optimized code',
                    'type': 'path'
                },
                {
                    'name': '--module-name',
                    'short': '-m',
                    'default': 'optimized',
                    'help': 'Name for generated module'
                },
                {
                    'name': '--min-speedup',
                    'short': '-s',
                    'default': 5.0,
                    'type': 'float',
                    'help': 'Minimum estimated speedup to optimize (default: 5.0x)'
                },
                {
                    'name': '--dry-run',
                    'short': '-d',
                    'is_flag': True,
                    'help': 'Show what would be optimized without generating code'
                },
                {
                    'name': '--verbose',
                    'short': '-v',
                    'is_flag': True,
                    'help': 'Verbose output'
                }
            ]
        },
        'analyze': {
            'help': '🔍 Analyze a Python file for performance hotspots',
            'description': '''
Identifies potential optimization targets without generating code.
Useful for understanding where your bottlenecks are.

Example: pyrust analyze my_script.py --threshold 0.7 --verbose
            ''',
            'arguments': [
                {
                    'name': 'python_file',
                    'type': 'path',
                    'required': True,
                    'help': 'Python file to analyze'
                }
            ],
            'options': [
                {
                    'name': '--threshold',
                    'short': '-t',
                    'default': 0.5,
                    'type': 'float',
                    'help': 'Confidence threshold for hotspot detection (0.0-1.0)'
                },
                {
                    'name': '--verbose',
                    'short': '-v',
                    'is_flag': True,
                    'help': 'Show detailed analysis'
                }
            ]
        },
        'setup': {
            'help': '🛠️ Set up PyRust Optimizer development environment',
            'description': 'Checks dependencies, creates workspace, and verifies installation.',
            'options': [
                {
                    'name': '--workspace',
                    'short': '-w',
                    'help': 'Custom workspace directory'
                }
            ]
        },
        'demo': {
            'help': '🎮 Run PyRust Optimizer demonstration',
            'description': 'Shows the complete optimization pipeline with example code.'
        }
    }
}


def create_click_option(option_config: dict):
    """Create a Click option from configuration."""
    option_args = []
    option_kwargs = {}

    # Handle short and long names
    if 'short' in option_config:
        option_args.append(option_config['short'])
    option_args.append(option_config['name'])

    # Handle option parameters
    if 'default' in option_config:
        option_kwargs['default'] = option_config['default']
    if 'help' in option_config:
        option_kwargs['help'] = option_config['help']
    if 'type' in option_config:
        if option_config['type'] == 'float':
            option_kwargs['type'] = float
        elif option_config['type'] == 'path':
            option_kwargs['type'] = click.Path()
    if option_config.get('is_flag'):
        option_kwargs['is_flag'] = True

    return click.option(*option_args, **option_kwargs)


def create_click_argument(arg_config: dict):
    """Create a Click argument from configuration."""
    arg_kwargs = {}

    if arg_config.get('required'):
        arg_kwargs['required'] = True
    if 'type' in arg_config:
        if arg_config['type'] == 'path':
            arg_kwargs['type'] = click.Path(exists=True, dir_okay=False)

    return click.argument(arg_config['name'], **arg_kwargs)


def generate_command_function(cmd_name: str, cmd_config: dict):
    """Generate a Click command function from configuration."""

    def command_func(**kwargs):
        """Dynamically generated command function."""
        # Import the actual implementation
        from .handlers import CommandHandlers

        handler = CommandHandlers()
        method = getattr(handler, f"handle_{cmd_name}")
        return method(**kwargs)

    # Set function metadata
    command_func.__name__ = cmd_name
    command_func.__doc__ = cmd_config.get('description', cmd_config.get('help'))

    # Apply Click decorators dynamically
    decorators = []

    # Add options
    for option_config in reversed(cmd_config.get('options', [])):
        decorators.append(create_click_option(option_config))

    # Add arguments
    for arg_config in reversed(cmd_config.get('arguments', [])):
        decorators.append(create_click_argument(arg_config))

    # Add command decorator
    decorators.append(click.command(help=cmd_config.get('help')))

    # Apply all decorators
    for decorator in decorators:
        command_func = decorator(command_func)

    return command_func


def generate_cli_from_config(config: dict):
    """Generate a complete Click CLI from configuration (Infrastructure as Code)."""

    @click.group()
    @click.version_option(version=config['version'], prog_name=config['name'])
    def cli():
        """Dynamically generated CLI from configuration."""
        pass

    # Set CLI documentation
    cli.__doc__ = config.get('description', config.get('help'))

    # Generate and add commands
    for cmd_name, cmd_config in config['commands'].items():
        command_func = generate_command_function(cmd_name, cmd_config)
        cli.add_command(command_func)

    return cli


# Generate the CLI automatically from configuration
cli = generate_cli_from_config(CLI_CONFIG)


if __name__ == '__main__':
    # Show the generated CLI structure
    print("🏗️ Infrastructure as Code - Generated CLI Structure:")
    print("=" * 55)
    print(f"CLI Name: {CLI_CONFIG['name']}")
    print(f"Version: {CLI_CONFIG['version']}")
    print(f"Commands: {list(CLI_CONFIG['commands'].keys())}")

    for cmd_name, cmd_config in CLI_CONFIG['commands'].items():
        print(f"\n📋 Command: {cmd_name}")
        print(f"   Help: {cmd_config['help']}")
        print(f"   Arguments: {len(cmd_config.get('arguments', []))}")
        print(f"   Options: {len(cmd_config.get('options', []))}")

    print(f"\n🚀 CLI generated automatically from configuration!")
    print(f"🎯 This is Infrastructure as Code in action!")
