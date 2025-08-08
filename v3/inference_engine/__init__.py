"""
Arbitrary Numbers Inference Engine
=================================

High-performance inference engine for exact symbolic computation in machine learning.
Optimized for consumer-grade Nvidia 32GB GPUs (RTX 4090, RTX 6000 Ada).

License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Arbitrary Numbers Development Team"
__license__ = "Apache 2.0"

from .core.engine import InferenceEngine
from .core.model_loader import ModelLoader
from .core.batch_processor import BatchProcessor
from .optimization.symbolic_optimizer import SymbolicOptimizer
from .gpu.inference_kernels import GPUInferenceEngine
from .api.inference_server import InferenceServer

__all__ = [
    'InferenceEngine',
    'ModelLoader',
    'BatchProcessor',
    'SymbolicOptimizer',
    'GPUInferenceEngine',
    'InferenceServer'
]
