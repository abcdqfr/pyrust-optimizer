"""
ASTAnalyzer: Universal AST Analysis using Tree-sitter

This module provides comprehensive AST analysis capabilities for identifying
code patterns, complexity metrics, and optimization opportunities.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import ast
import re
from collections import defaultdict, namedtuple
from dataclasses import dataclass

# Since tree-sitter-python might not be installed, we'll provide a fallback
try:
    import tree_sitter_python as ts_python
    from tree_sitter import Language, Parser, Node, Tree
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Tree-sitter not available. Using fallback AST analysis.")

@dataclass
class CodeMetrics:
    """Comprehensive code complexity metrics."""
    cyclomatic_complexity: int
    nesting_depth: int
    function_count: int
    loop_count: int
    conditional_count: int
    line_count: int
    variable_count: int
    rust_conversion_score: float

@dataclass
class FunctionAnalysis:
    """Detailed analysis of a function."""
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    return_type: Optional[str]
    complexity_metrics: CodeMetrics
    optimization_opportunities: List[str]
    dependencies: List[str]
    rust_suitability: float

class ASTAnalyzer:
    """
    Universal AST analyzer supporting both Tree-sitter and Python AST
    for comprehensive code analysis and optimization identification.
    """

    def __init__(self, use_tree_sitter: bool = True):
        """
        Initialize the AST analyzer.

        Args:
            use_tree_sitter: Whether to use Tree-sitter (preferred) or fallback to Python AST
        """
        self.use_tree_sitter = use_tree_sitter and TREE_SITTER_AVAILABLE

        if self.use_tree_sitter:
            self._setup_tree_sitter()

        # Optimization pattern recognition
        self.optimization_patterns = {
            'numerical_loops': self._detect_numerical_loops,
            'list_comprehensions': self._detect_list_comprehensions,
            'string_operations': self._detect_string_operations,
            'file_operations': self._detect_file_operations,
            'algorithmic_patterns': self._detect_algorithmic_patterns,
        }

        # Rust suitability weights
        self.rust_suitability_weights = {
            'loops': 0.3,
            'arithmetic': 0.25,
            'memory_intensive': 0.2,
            'parallelizable': 0.15,
            'io_operations': 0.1
        }

    def _setup_tree_sitter(self):
        """Setup Tree-sitter parser for Python."""
        try:
            self.parser = Parser()
            self.parser.set_language(ts_python.language())
        except Exception as e:
            print(f"Tree-sitter setup failed: {e}. Falling back to Python AST.")
            self.use_tree_sitter = False

    def analyze_code(self, source_code: str) -> Dict[str, Any]:
        """
        Analyze source code for optimization opportunities.

        Args:
            source_code: Python source code to analyze

        Returns:
            Comprehensive analysis results
        """
        if self.use_tree_sitter:
            return self._analyze_with_tree_sitter(source_code)
        else:
            return self._analyze_with_python_ast(source_code)

    def _analyze_with_tree_sitter(self, source_code: str) -> Dict[str, Any]:
        """Analyze code using Tree-sitter parser."""
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node

        analysis = {
            'functions': [],
            'global_metrics': self._calculate_global_metrics_ts(root_node),
            'optimization_opportunities': [],
            'dependency_graph': {},
            'type_annotations': {},
            'performance_hotspots': []
        }

        # Extract and analyze functions
        functions = self._extract_functions_ts(root_node)
        for func_node in functions:
            func_analysis = self._analyze_function_ts(func_node, source_code)
            analysis['functions'].append(func_analysis)

        # Identify optimization opportunities
        analysis['optimization_opportunities'] = self._identify_optimizations_ts(root_node)

        # Build dependency graph
        analysis['dependency_graph'] = self._build_dependency_graph_ts(root_node)

        return analysis

    def _analyze_with_python_ast(self, source_code: str) -> Dict[str, Any]:
        """Fallback analysis using Python's built-in AST module."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return {'error': f'Syntax error in source code: {e}'}

        analysis = {
            'functions': [],
            'global_metrics': self._calculate_global_metrics_ast(tree),
            'optimization_opportunities': [],
            'dependency_graph': {},
            'type_annotations': {},
            'performance_hotspots': []
        }

        # Extract and analyze functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_analysis = self._analyze_function_ast(node, source_code)
                analysis['functions'].append(func_analysis)

        # Identify optimization opportunities
        analysis['optimization_opportunities'] = self._identify_optimizations_ast(tree)

        return analysis

    def _extract_functions_ts(self, root_node) -> List:
        """Extract function nodes using Tree-sitter."""
        functions = []

        def traverse(node):
            if node.type == 'function_definition':
                functions.append(node)
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return functions

    def _analyze_function_ts(self, func_node, source_code: str) -> FunctionAnalysis:
        """Analyze a function using Tree-sitter."""
        # Extract function name
        name_node = func_node.child_by_field_name('name')
        func_name = source_code[name_node.start_byte:name_node.end_byte] if name_node else 'unknown'

        # Extract parameters
        parameters = self._extract_parameters_ts(func_node, source_code)

        # Calculate complexity metrics
        metrics = self._calculate_function_metrics_ts(func_node)

        # Identify optimization opportunities
        opportunities = self._identify_function_optimizations_ts(func_node, source_code)

        # Calculate Rust suitability
        rust_suitability = self._calculate_rust_suitability(metrics, opportunities)

        return FunctionAnalysis(
            name=func_name,
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            parameters=parameters,
            return_type=None,  # TODO: Extract from type annotations
            complexity_metrics=metrics,
            optimization_opportunities=opportunities,
            dependencies=[],  # TODO: Extract dependencies
            rust_suitability=rust_suitability
        )

    def _analyze_function_ast(self, func_node: ast.FunctionDef, source_code: str) -> FunctionAnalysis:
        """Analyze a function using Python AST."""
        # Extract basic information
        func_name = func_node.name
        parameters = [arg.arg for arg in func_node.args.args]

        # Calculate complexity metrics
        metrics = self._calculate_function_metrics_ast(func_node)

        # Identify optimization opportunities
        opportunities = self._identify_function_optimizations_ast(func_node)

        # Calculate Rust suitability
        rust_suitability = self._calculate_rust_suitability(metrics, opportunities)

        return FunctionAnalysis(
            name=func_name,
            start_line=func_node.lineno,
            end_line=func_node.end_lineno or func_node.lineno,
            parameters=parameters,
            return_type=None,  # TODO: Extract from type annotations
            complexity_metrics=metrics,
            optimization_opportunities=opportunities,
            dependencies=[],  # TODO: Extract dependencies
            rust_suitability=rust_suitability
        )

    def _calculate_function_metrics_ts(self, func_node) -> CodeMetrics:
        """Calculate complexity metrics using Tree-sitter."""
        metrics = {
            'cyclomatic_complexity': 1,  # Base complexity
            'nesting_depth': 0,
            'function_count': 0,
            'loop_count': 0,
            'conditional_count': 0,
            'line_count': func_node.end_point[0] - func_node.start_point[0] + 1,
            'variable_count': 0,
            'max_depth': 0
        }

        def traverse(node, depth=0):
            metrics['max_depth'] = max(metrics['max_depth'], depth)

            if node.type in ['for_statement', 'while_statement']:
                metrics['loop_count'] += 1
                metrics['cyclomatic_complexity'] += 1
            elif node.type in ['if_statement', 'elif_clause']:
                metrics['conditional_count'] += 1
                metrics['cyclomatic_complexity'] += 1
            elif node.type == 'function_definition':
                metrics['function_count'] += 1
            elif node.type == 'identifier':
                metrics['variable_count'] += 1

            for child in node.children:
                traverse(child, depth + 1)

        traverse(func_node)
        metrics['nesting_depth'] = metrics['max_depth']

        # Calculate Rust conversion score based on metrics
        rust_score = self._calculate_metrics_rust_score(metrics)

        return CodeMetrics(
            cyclomatic_complexity=metrics['cyclomatic_complexity'],
            nesting_depth=metrics['nesting_depth'],
            function_count=metrics['function_count'],
            loop_count=metrics['loop_count'],
            conditional_count=metrics['conditional_count'],
            line_count=metrics['line_count'],
            variable_count=metrics['variable_count'],
            rust_conversion_score=rust_score
        )

    def _calculate_function_metrics_ast(self, func_node: ast.FunctionDef) -> CodeMetrics:
        """Calculate complexity metrics using Python AST."""
        metrics = {
            'cyclomatic_complexity': 1,  # Base complexity
            'nesting_depth': 0,
            'function_count': 0,
            'loop_count': 0,
            'conditional_count': 0,
            'line_count': (func_node.end_lineno or func_node.lineno) - func_node.lineno + 1,
            'variable_count': 0,
            'max_depth': 0
        }

        def traverse(node, depth=0):
            metrics['max_depth'] = max(metrics['max_depth'], depth)

            if isinstance(node, (ast.For, ast.While)):
                metrics['loop_count'] += 1
                metrics['cyclomatic_complexity'] += 1
            elif isinstance(node, (ast.If, )):
                metrics['conditional_count'] += 1
                metrics['cyclomatic_complexity'] += 1
            elif isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
            elif isinstance(node, ast.Name):
                metrics['variable_count'] += 1

            for child in ast.iter_child_nodes(node):
                traverse(child, depth + 1)

        traverse(func_node)
        metrics['nesting_depth'] = metrics['max_depth']

        # Calculate Rust conversion score
        rust_score = self._calculate_metrics_rust_score(metrics)

        return CodeMetrics(
            cyclomatic_complexity=metrics['cyclomatic_complexity'],
            nesting_depth=metrics['nesting_depth'],
            function_count=metrics['function_count'],
            loop_count=metrics['loop_count'],
            conditional_count=metrics['conditional_count'],
            line_count=metrics['line_count'],
            variable_count=metrics['variable_count'],
            rust_conversion_score=rust_score
        )

    def _calculate_metrics_rust_score(self, metrics: Dict[str, int]) -> float:
        """Calculate Rust conversion score based on complexity metrics."""
        # Higher scores for characteristics that benefit from Rust
        loop_score = min(metrics['loop_count'] / 5.0, 1.0) * 0.3
        complexity_score = min(metrics['cyclomatic_complexity'] / 10.0, 1.0) * 0.2
        size_score = min(metrics['line_count'] / 50.0, 1.0) * 0.1

        # Penalize excessive complexity (harder to convert)
        if metrics['cyclomatic_complexity'] > 20:
            complexity_penalty = 0.2
        else:
            complexity_penalty = 0.0

        rust_score = loop_score + complexity_score + size_score - complexity_penalty
        return max(0.0, min(1.0, rust_score))

    def _identify_function_optimizations_ts(self, func_node, source_code: str) -> List[str]:
        """Identify optimization opportunities in a function using Tree-sitter."""
        opportunities = []

        # Apply pattern recognition
        for pattern_name, detector in self.optimization_patterns.items():
            if detector(func_node, source_code):
                opportunities.append(pattern_name)

        return opportunities

    def _identify_function_optimizations_ast(self, func_node: ast.FunctionDef) -> List[str]:
        """Identify optimization opportunities using Python AST."""
        opportunities = []

        # Detect loops
        if any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(func_node)):
            opportunities.append('loop_optimization')

        # Detect list comprehensions
        if any(isinstance(node, ast.ListComp) for node in ast.walk(func_node)):
            opportunities.append('list_comprehension_optimization')

        # Detect arithmetic operations
        if any(isinstance(node, (ast.Add, ast.Mult, ast.Sub, ast.Div))
               for node in ast.walk(func_node)):
            opportunities.append('arithmetic_optimization')

        return opportunities

    def _calculate_rust_suitability(self, metrics: CodeMetrics,
                                  opportunities: List[str]) -> float:
        """Calculate overall Rust suitability score."""
        base_score = metrics.rust_conversion_score

        # Bonus for specific optimization opportunities
        opportunity_bonus = 0.0
        for opp in opportunities:
            if 'loop' in opp:
                opportunity_bonus += 0.2
            elif 'arithmetic' in opp:
                opportunity_bonus += 0.15
            elif 'numerical' in opp:
                opportunity_bonus += 0.25

        return min(1.0, base_score + opportunity_bonus)

    # Pattern detection methods
    def _detect_numerical_loops(self, node, source_code: str) -> bool:
        """Detect numerical computation loops."""
        # This would implement Tree-sitter queries for numerical patterns
        return 'for' in source_code and any(op in source_code for op in ['+', '*', '-', '/'])

    def _detect_list_comprehensions(self, node, source_code: str) -> bool:
        """Detect list comprehensions that could be optimized."""
        return '[' in source_code and 'for' in source_code and 'in' in source_code

    def _detect_string_operations(self, node, source_code: str) -> bool:
        """Detect string processing operations."""
        string_ops = ['join', 'split', 'replace', 'strip', 'format']
        return any(op in source_code for op in string_ops)

    def _detect_file_operations(self, node, source_code: str) -> bool:
        """Detect file I/O operations."""
        file_ops = ['open', 'read', 'write', 'close']
        return any(op in source_code for op in file_ops)

    def _detect_algorithmic_patterns(self, node, source_code: str) -> bool:
        """Detect algorithmic patterns suitable for optimization."""
        algo_patterns = ['sort', 'search', 'filter', 'map', 'reduce']
        return any(pattern in source_code for pattern in algo_patterns)

    def _extract_parameters_ts(self, func_node, source_code: str) -> List[str]:
        """Extract function parameters using Tree-sitter."""
        params = []
        parameters_node = func_node.child_by_field_name('parameters')

        if parameters_node:
            for child in parameters_node.children:
                if child.type == 'identifier':
                    param_name = source_code[child.start_byte:child.end_byte]
                    params.append(param_name)

        return params

    def _calculate_global_metrics_ts(self, root_node) -> Dict[str, Any]:
        """Calculate global code metrics using Tree-sitter."""
        return {
            'total_functions': len(self._extract_functions_ts(root_node)),
            'total_lines': root_node.end_point[0] + 1,
            'complexity_distribution': {},  # TODO: Implement
        }

    def _calculate_global_metrics_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate global code metrics using Python AST."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return {
            'total_functions': len(functions),
            'total_lines': 0,  # TODO: Calculate from AST
            'complexity_distribution': {},
        }

    def _identify_optimizations_ts(self, root_node) -> List[Dict[str, Any]]:
        """Identify global optimization opportunities using Tree-sitter."""
        return []  # TODO: Implement global optimization detection

    def _identify_optimizations_ast(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify global optimization opportunities using Python AST."""
        return []  # TODO: Implement global optimization detection

    def _build_dependency_graph_ts(self, root_node) -> Dict[str, List[str]]:
        """Build function dependency graph using Tree-sitter."""
        return {}  # TODO: Implement dependency analysis


def analyze_python_code(source_code: str) -> Dict[str, Any]:
    """
    Convenience function to analyze Python code.

    Args:
        source_code: Python source code to analyze

    Returns:
        Comprehensive analysis results
    """
    analyzer = ASTAnalyzer()
    return analyzer.analyze_code(source_code)


# Example usage
if __name__ == "__main__":
    # Example code to analyze
    example_code = '''
def process_data(items):
    """Example function with optimization opportunities."""
    results = []
    for item in items:
        if item > 0:
            result = item * 2 + 1
            results.append(result)
    return results

def compute_matrix(matrix):
    """Numerical computation example."""
    total = 0
    for row in matrix:
        for value in row:
            total += value * value
    return total
'''

    # Analyze the code
    analyzer = ASTAnalyzer()
    results = analyzer.analyze_code(example_code)

    print("🌳 PyRust Optimizer - AST Analysis Results")
    print("=" * 50)
    print(f"Total functions analyzed: {len(results['functions'])}")

    for func in results['functions']:
        print(f"\n📊 Function: {func.name}")
        print(f"   Lines: {func.start_line}-{func.end_line}")
        print(f"   Complexity: {func.complexity_metrics.cyclomatic_complexity}")
        print(f"   Rust suitability: {func.rust_suitability:.2f}")
        print(f"   Optimization opportunities: {', '.join(func.optimization_opportunities)}")
