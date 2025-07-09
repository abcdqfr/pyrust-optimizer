"""
Data collection pipeline for PyRust Optimizer training.

This module collects training data from awesome-python repositories
to train our hotspot detection model.
"""

import os
import ast
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import requests
import re


@dataclass
class TrainingSample:
    """Represents a training sample for hotspot detection."""
    library_name: str
    file_path: str
    function_name: str
    code_snippet: str
    features: Dict[str, Any]
    is_hotspot: bool
    confidence_score: float


class AwesomePythonDataCollector:
    """
    Collects training data from awesome-python repositories.

    This class extracts Python code from popular libraries and
    creates training samples for our hotspot detection model.
    """

    def __init__(self, awesome_python_path: str = "/home/brandon/Documents/pytools/awesome-python"):
        self.awesome_python_path = Path(awesome_python_path)
        self.training_data = []
        self.libraries = []

    def extract_libraries_from_awesome(self) -> List[Tuple[str, str]]:
        """
        Extract library names and URLs from awesome-python readme.

        Returns:
            List of (library_name, github_url) tuples
        """
        readme_path = self.awesome_python_path / "readme.md"

        if not readme_path.exists():
            print(f"❌ Awesome-python readme not found at {readme_path}")
            return []

        libraries = []

        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract library links from markdown
        # Pattern: [library_name](github_url)
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)

        for name, url in matches:
            if 'github.com' in url:
                # Extract repo name from URL
                repo_name = url.split('github.com/')[-1].split('/')[-1]
                libraries.append((repo_name, url))

        print(f"📚 Found {len(libraries)} libraries in awesome-python")
        return libraries[:50]  # Start with first 50 libraries

    def clone_repository(self, repo_url: str, repo_name: str) -> str:
        """
        Clone a repository to local storage.

        Args:
            repo_url: GitHub URL of the repository
            repo_name: Name of the repository

        Returns:
            Path to cloned repository
        """
        clone_path = Path(f"/tmp/pyrust_training/{repo_name}")

        if clone_path.exists():
            print(f"📁 Repository {repo_name} already exists, skipping clone")
            return str(clone_path)

        try:
            clone_path.parent.mkdir(parents=True, exist_ok=True)

            # Clone the repository
            subprocess.run([
                'git', 'clone', '--depth', '1',
                repo_url, str(clone_path)
            ], check=True, capture_output=True)

            print(f"✅ Cloned {repo_name}")
            return str(clone_path)

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone {repo_name}: {e}")
            return ""

    def find_python_files(self, repo_path: str) -> List[str]:
        """
        Find all Python files in a repository.

        Args:
            repo_path: Path to the repository

        Returns:
            List of Python file paths
        """
        python_files = []
        repo_path = Path(repo_path)

        for file_path in repo_path.rglob("*.py"):
            # Skip test files, docs, etc.
            if any(skip in str(file_path) for skip in ['test', 'tests', 'docs', 'examples']):
                continue

            # Skip files that are too large
            if file_path.stat().st_size > 100000:  # 100KB limit
                continue

            python_files.append(str(file_path))

        return python_files

    def extract_functions_from_file(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Extract function definitions from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of (function_name, function_code) tuples
        """
        functions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function code
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10

                    lines = content.split('\n')
                    function_code = '\n'.join(lines[start_line-1:end_line])

                    functions.append((node.name, function_code))

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")

        return functions

    def extract_features(self, code_snippet: str) -> Dict[str, Any]:
        """
        Extract features from a code snippet for ML training.

        Args:
            code_snippet: Python code to analyze

        Returns:
            Dictionary of features
        """
        try:
            tree = ast.parse(code_snippet)

            features = {
                # Loop complexity
                'nested_loop_count': self._count_nested_loops(tree),
                'for_loop_count': self._count_for_loops(tree),
                'while_loop_count': self._count_while_loops(tree),
                'max_loop_depth': self._get_max_loop_depth(tree),

                # Function complexity
                'function_call_count': self._count_function_calls(tree),
                'method_call_count': self._count_method_calls(tree),
                'recursive_call_count': self._count_recursive_calls(tree),

                # Data structure usage
                'list_creation_count': self._count_list_creations(tree),
                'dict_creation_count': self._count_dict_creations(tree),
                'set_creation_count': self._count_set_creations(tree),
                'tuple_creation_count': self._count_tuple_creations(tree),

                # Memory operations
                'append_operations': self._count_append_ops(tree),
                'extend_operations': self._count_extend_ops(tree),
                'insert_operations': self._count_insert_ops(tree),

                # Mathematical operations
                'arithmetic_ops': self._count_arithmetic_ops(tree),
                'comparison_ops': self._count_comparison_ops(tree),
                'logical_ops': self._count_logical_ops(tree),

                # Code structure
                'line_count': len(code_snippet.split('\n')),
                'variable_count': self._count_variables(tree),
                'import_count': self._count_imports(tree),
                'class_count': self._count_classes(tree),

                # Complexity metrics
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                'cognitive_complexity': self._calculate_cognitive_complexity(tree)
            }

            return features

        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return {}

    def _count_nested_loops(self, tree: ast.AST) -> int:
        """Count nested loops in the AST."""
        nested_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        nested_count += 1
        return nested_count

    def _count_for_loops(self, tree: ast.AST) -> int:
        """Count for loops in the AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.For)])

    def _count_while_loops(self, tree: ast.AST) -> int:
        """Count while loops in the AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.While)])

    def _get_max_loop_depth(self, tree: ast.AST) -> int:
        """Get maximum loop nesting depth."""
        max_depth = 0
        current_depth = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(node, ast.FunctionDef):
                current_depth = 0

        return max_depth

    def _count_function_calls(self, tree: ast.AST) -> int:
        """Count function calls in the AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)])

    def _count_method_calls(self, tree: ast.AST) -> int:
        """Count method calls in the AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)])

    def _count_recursive_calls(self, tree: ast.AST) -> int:
        """Count recursive function calls."""
        recursive_count = 0
        function_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in function_names:
                    recursive_count += 1

        return recursive_count

    def _count_list_creations(self, tree: ast.AST) -> int:
        """Count list creations."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.List)])

    def _count_dict_creations(self, tree: ast.AST) -> int:
        """Count dictionary creations."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Dict)])

    def _count_set_creations(self, tree: ast.AST) -> int:
        """Count set creations."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Set)])

    def _count_tuple_creations(self, tree: ast.AST) -> int:
        """Count tuple creations."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Tuple)])

    def _count_append_ops(self, tree: ast.AST) -> int:
        """Count append operations."""
        return len([node for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'append'])

    def _count_extend_ops(self, tree: ast.AST) -> int:
        """Count extend operations."""
        return len([node for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'extend'])

    def _count_insert_ops(self, tree: ast.AST) -> int:
        """Count insert operations."""
        return len([node for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'insert'])

    def _count_arithmetic_ops(self, tree: ast.AST) -> int:
        """Count arithmetic operations."""
        return len([node for node in ast.walk(tree)
                   if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))])

    def _count_comparison_ops(self, tree: ast.AST) -> int:
        """Count comparison operations."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Compare)])

    def _count_logical_ops(self, tree: ast.AST) -> int:
        """Count logical operations."""
        return len([node for node in ast.walk(tree)
                   if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or))])

    def _count_variables(self, tree: ast.AST) -> int:
        """Count variable assignments."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Assign)])

    def _count_imports(self, tree: ast.AST) -> int:
        """Count import statements."""
        return len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])

    def _count_classes(self, tree: ast.AST) -> int:
        """Count class definitions."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity."""
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                complexity += 1
            elif isinstance(node, ast.While):
                complexity += 1
            elif isinstance(node, ast.For):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def determine_hotspot_label(self, code_snippet: str) -> Tuple[bool, float]:
        """
        Determine if a code snippet is a performance hotspot.

        Args:
            code_snippet: Python code to analyze

        Returns:
            Tuple of (is_hotspot, confidence_score)
        """
        # Use simple heuristics to determine hotspots
        features = self.extract_features(code_snippet)

        # Heuristic rules for hotspot detection
        hotspot_score = 0.0

        # Nested loops are likely hotspots
        if features.get('nested_loop_count', 0) > 0:
            hotspot_score += 0.4

        # Large loops are likely hotspots
        if features.get('for_loop_count', 0) > 2:
            hotspot_score += 0.3

        # Memory operations are likely hotspots
        if features.get('append_operations', 0) > 5:
            hotspot_score += 0.2

        # Mathematical operations are likely hotspots
        if features.get('arithmetic_ops', 0) > 10:
            hotspot_score += 0.2

        # Complex functions are likely hotspots
        if features.get('cyclomatic_complexity', 0) > 5:
            hotspot_score += 0.3

        is_hotspot = hotspot_score > 0.5
        confidence = min(hotspot_score, 1.0)

        return is_hotspot, confidence

    def collect_training_data(self, max_libraries: int = 20) -> List[TrainingSample]:
        """
        Collect training data from awesome-python repositories.

        Args:
            max_libraries: Maximum number of libraries to process

        Returns:
            List of training samples
        """
        print("🚀 Starting training data collection from awesome-python...")

        # Extract libraries from awesome-python
        libraries = self.extract_libraries_from_awesome()[:max_libraries]

        training_samples = []

        for i, (repo_name, repo_url) in enumerate(libraries):
            print(f"\n📦 Processing library {i+1}/{len(libraries)}: {repo_name}")

            # Clone repository
            repo_path = self.clone_repository(repo_url, repo_name)
            if not repo_path:
                continue

            # Find Python files
            python_files = self.find_python_files(repo_path)
            print(f"   Found {len(python_files)} Python files")

            # Extract functions from each file
            for file_path in python_files[:10]:  # Limit to 10 files per repo
                functions = self.extract_functions_from_file(file_path)

                for func_name, func_code in functions:
                    # Extract features
                    features = self.extract_features(func_code)

                    # Determine if it's a hotspot
                    is_hotspot, confidence = self.determine_hotspot_label(func_code)

                    # Create training sample
                    sample = TrainingSample(
                        library_name=repo_name,
                        file_path=file_path,
                        function_name=func_name,
                        code_snippet=func_code,
                        features=features,
                        is_hotspot=is_hotspot,
                        confidence_score=confidence
                    )

                    training_samples.append(sample)

        print(f"\n✅ Collected {len(training_samples)} training samples")
        return training_samples

    def save_training_data(self, samples: List[TrainingSample], output_path: str):
        """
        Save training data to JSON file.

        Args:
            samples: List of training samples
            output_path: Path to save the data
        """
        data = []

        for sample in samples:
            data.append({
                'library_name': sample.library_name,
                'file_path': sample.file_path,
                'function_name': sample.function_name,
                'code_snippet': sample.code_snippet,
                'features': sample.features,
                'is_hotspot': sample.is_hotspot,
                'confidence_score': sample.confidence_score
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved {len(samples)} training samples to {output_path}")


def demo_data_collection():
    """Demo the data collection pipeline."""
    print("🔥 PyRust Optimizer - Training Data Collection Demo")
    print("=" * 60)

    # Initialize collector
    collector = AwesomePythonDataCollector()

    # Collect training data
    training_samples = collector.collect_training_data(max_libraries=5)

    # Save training data
    output_path = "/home/brandon/Documents/Cursor/pyrust-optimizer/data/training_samples.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    collector.save_training_data(training_samples, output_path)

    # Print statistics
    hotspot_count = sum(1 for sample in training_samples if sample.is_hotspot)
    total_count = len(training_samples)

    print(f"\n📊 Training Data Statistics:")
    print(f"   Total samples: {total_count}")
    print(f"   Hotspots: {hotspot_count}")
    print(f"   Non-hotspots: {total_count - hotspot_count}")
    if total_count > 0:
        print(f"   Hotspot ratio: {hotspot_count/total_count:.2%}")
    else:
        print(f"   Hotspot ratio: 0.00% (no samples collected)")

    print(f"\n🎯 Ready to train the hotspot detection model!")


if __name__ == "__main__":
    demo_data_collection()
