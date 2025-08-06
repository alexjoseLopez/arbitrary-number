"""
Arbitrary Numbers: Exact Symbolic Computation for Python
========================================================

A Python implementation of the Arbitrary Numbers system for exact symbolic computation
with GPU acceleration, designed for inference models and scientific computing.

License: Apache 2.0
Target Hardware: Consumer-grade Nvidia 32GB GPUs (RTX 4090, RTX 6000 Ada)
"""

__version__ = "0.1.0"
__author__ = "Arbitrary Numbers Development Team"
__license__ = "Apache 2.0"

from .core.rational_list import RationalListNumber, FractionTerm
from .core.equation_nodes import (
    EquationNode,
    ConstantNode, 
    BinaryOpNode,
    UnaryOpNode,
    VariableNode
)
from .core.evaluator import EquationEvaluator
from .gpu.cuda_kernels import GPUEvaluator
from .ml.pytorch_layers import SymbolicLayer, SymbolicLinear

__all__ = [
    'RationalListNumber',
    'FractionTerm', 
    'EquationNode',
    'ConstantNode',
    'BinaryOpNode', 
    'UnaryOpNode',
    'VariableNode',
    'EquationEvaluator',
    'GPUEvaluator',
    'SymbolicLayer',
    'SymbolicLinear'
]
