"""
Heuristics-based hotspot detection for PyRust Optimizer.

This module provides deterministic, rule-based identification of performance
hotspots in Python code using AST analysis and statistical profiling.
"""

import ast
import cProfile
import pstats
import io
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Hotspot:
    """Represents a detected performance hotspot."""
    function_name: str
    line_number: int
    hotspot_type: str
    confidence_score: float
    optimization_potential: float
    description: str


class HeuristicHotspotDetector:
    """
    Heuristics-based hotspot detector using deterministic rules.

    This class identifies performance bottlenecks using simple, predictable
    heuristics rather than complex AI/ML approaches.
    """

    def __init__(self):
        # Heuristic thresholds
        self.loop_complexity_threshold = 2  # Nested loops
        self.iteration_threshold = 1000     # Large iterations
        self.memory_threshold = 1000000     # Large allocations (bytes)
        self.cpu_threshold = 0.1           # CPU time threshold (seconds)

        # Hotspot type weights for confidence scoring
        self.hotspot_weights = {
            'nested_loop': 0.9,
            'large_iteration': 0.8,
            'memory_intensive': 0.7,
            'mathematical': 0.6,
            'string_processing': 0.5,
            'io_operation': 0.4
        }

    def detect_hotspots(self, python_file: str) -> List[Hotspot]:
        """
        Detect performance hotspots in a Python file using heuristics.

        Args:
            python_file: Path to the Python file to analyze

        Returns:
            List of detected hotspots with confidence scores
        """
        hotspots = []

        # Parse the Python file
        with open(python_file, 'r') as f:
            source_code = f.read()

        tree = ast.parse(source_code)

        # Apply heuristic rules
        hotspots.extend(self._detect_nested_loops(tree))
        hotspots.extend(self._detect_large_iterations(tree))
        hotspots.extend(self._detect_memory_intensive_operations(tree))
        hotspots.extend(self._detect_mathematical_computations(tree))
        hotspots.extend(self._detect_string_processing(tree))
        hotspots.extend(self._detect_io_operations(tree))

        # Sort by optimization potential
        hotspots.sort(key=lambda x: x.optimization_potential, reverse=True)

        return hotspots

    def _detect_nested_loops(self, tree: ast.AST) -> List[Hotspot]:
        """Detect nested loops (O(n²) complexity)."""
        hotspots = []

        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if this loop contains another loop
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        hotspots.append(Hotspot(
                            function_name=self._get_function_name(node),
                            line_number=node.lineno,
                            hotspot_type='nested_loop',
                            confidence_score=0.9,
                            optimization_potential=0.95,
                            description=f"Nested loop detected at line {node.lineno}"
                        ))
                        break

        return hotspots

    def _detect_large_iterations(self, tree: ast.AST) -> List[Hotspot]:
        """Detect loops with large iteration counts."""
        hotspots = []

        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if iterating over a large range or list
                if isinstance(node.iter, ast.Call):
                    if (isinstance(node.iter.func, ast.Name) and
                        node.iter.func.id in ['range', 'enumerate']):
                        # Check for large range values
                        for arg in node.iter.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                                if arg.value > self.iteration_threshold:
                                    hotspots.append(Hotspot(
                                        function_name=self._get_function_name(node),
                                        line_number=node.lineno,
                                        hotspot_type='large_iteration',
                                        confidence_score=0.8,
                                        optimization_potential=0.85,
                                        description=f"Large iteration ({arg.value}) at line {node.lineno}"
                                    ))

        return hotspots

    def _detect_memory_intensive_operations(self, tree: ast.AST) -> List[Hotspot]:
        """Detect memory-intensive operations."""
        hotspots = []

        memory_ops = [
            'list', 'dict', 'set', 'tuple',
            'numpy.array', 'pandas.DataFrame',
            'append', 'extend', 'insert'
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in memory_ops:
                        hotspots.append(Hotspot(
                            function_name=self._get_function_name(node),
                            line_number=node.lineno,
                            hotspot_type='memory_intensive',
                            confidence_score=0.7,
                            optimization_potential=0.75,
                            description=f"Memory-intensive operation '{node.func.id}' at line {node.lineno}"
                        ))

        return hotspots

    def _detect_mathematical_computations(self, tree: ast.AST) -> List[Hotspot]:
        """Detect mathematical computations."""
        hotspots = []

        math_ops = ['*', '/', '+', '-', '**', '%']
        math_functions = ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp']

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, (ast.Mult, ast.Div, ast.Pow)):
                    hotspots.append(Hotspot(
                        function_name=self._get_function_name(node),
                        line_number=node.lineno,
                        hotspot_type='mathematical',
                        confidence_score=0.6,
                        optimization_potential=0.8,
                        description=f"Mathematical operation at line {node.lineno}"
                    ))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in math_functions:
                    hotspots.append(Hotspot(
                        function_name=self._get_function_name(node),
                        line_number=node.lineno,
                        hotspot_type='mathematical',
                        confidence_score=0.6,
                        optimization_potential=0.8,
                        description=f"Mathematical function '{node.func.id}' at line {node.lineno}"
                    ))

        return hotspots

    def _detect_string_processing(self, tree: ast.AST) -> List[Hotspot]:
        """Detect string processing operations."""
        hotspots = []

        string_ops = ['split', 'join', 'replace', 'strip', 'format']

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in string_ops:
                        hotspots.append(Hotspot(
                            function_name=self._get_function_name(node),
                            line_number=node.lineno,
                            hotspot_type='string_processing',
                            confidence_score=0.5,
                            optimization_potential=0.6,
                            description=f"String operation '{node.func.attr}' at line {node.lineno}"
                        ))

        return hotspots

    def _detect_io_operations(self, tree: ast.AST) -> List[Hotspot]:
        """Detect I/O operations."""
        hotspots = []

        io_ops = ['open', 'read', 'write', 'print', 'input']

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in io_ops:
                    hotspots.append(Hotspot(
                        function_name=self._get_function_name(node),
                        line_number=node.lineno,
                        hotspot_type='io_operation',
                        confidence_score=0.4,
                        optimization_potential=0.3,  # I/O is often not CPU-bound
                        description=f"I/O operation '{node.func.id}' at line {node.lineno}"
                    ))

        return hotspots

    def _get_function_name(self, node: ast.AST) -> str:
        """Get the name of the function containing the node."""
        for parent in ast.walk(node):
            if isinstance(parent, ast.FunctionDef):
                return parent.name
        return "global"

    def profile_and_validate(self, python_file: str) -> Dict[str, Any]:
        """
        Profile the Python file and validate hotspot predictions.

        Args:
            python_file: Path to the Python file to profile

        Returns:
            Dictionary with profiling results and validation metrics
        """
        # Run cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        # Import and run the module (simplified)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("module", python_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            return {"error": f"Could not profile file: {e}"}

        profiler.disable()

        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()

        return {
            "profile_stats": s.getvalue(),
            "total_calls": profiler.getstats(),
            "hotspots_detected": len(self.detect_hotspots(python_file))
        }


def demo_hotspot_detection():
    """Demo the hotspot detection on a sample Python file."""

    # Create a sample Python file with known hotspots
    sample_code = '''
def slow_nested_loop():
    result = 0
    for i in range(1000):  # Large iteration
        for j in range(1000):  # Nested loop
            result += i * j  # Mathematical operation
    return result

def memory_intensive():
    data = []
    for i in range(10000):
        data.append(i)  # Memory allocation
    return data

def mathematical_computation():
    import math
    result = 0
    for i in range(1000):
        result += math.sin(i) + math.cos(i)  # Math functions
    return result
'''

    # Write sample file
    sample_file = "/tmp/sample_hotspots.py"
    with open(sample_file, 'w') as f:
        f.write(sample_code)

    # Detect hotspots
    detector = HeuristicHotspotDetector()
    hotspots = detector.detect_hotspots(sample_file)

    print("🔥 PyRust Optimizer - Hotspot Detection Demo")
    print("=" * 50)

    for i, hotspot in enumerate(hotspots, 1):
        print(f"\n{i}. {hotspot.hotspot_type.upper()}")
        print(f"   Function: {hotspot.function_name}")
        print(f"   Line: {hotspot.line_number}")
        print(f"   Confidence: {hotspot.confidence_score:.2f}")
        print(f"   Optimization Potential: {hotspot.optimization_potential:.2f}")
        print(f"   Description: {hotspot.description}")

    print(f"\n🎯 Total hotspots detected: {len(hotspots)}")
    print("💡 These hotspots are candidates for Rust optimization!")


if __name__ == "__main__":
    demo_hotspot_detection()
