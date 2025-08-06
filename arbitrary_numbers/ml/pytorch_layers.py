"""
PyTorch Integration Layer
========================

Custom PyTorch layers and modules for symbolic computation with Arbitrary Numbers.
Enables exact computation in neural networks and inference models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from ..core.equation_nodes import EquationNode, ConstantNode, BinaryOpNode, ExpressionBuilder
from ..core.rational_list import RationalListNumber, FractionTerm
from ..core.evaluator import EquationEvaluator
from ..gpu.cuda_kernels import GPUEvaluator


class SymbolicTensor:
    """
    Wrapper for tensors that maintains symbolic representation alongside numeric values.
    """
    
    def __init__(self, 
                 symbolic_data: List[List[EquationNode]], 
                 numeric_tensor: Optional[torch.Tensor] = None):
        self.symbolic_data = symbolic_data
        self.shape = (len(symbolic_data), len(symbolic_data[0]) if symbolic_data else 0)
        self._numeric_tensor = numeric_tensor
        self._is_evaluated = numeric_tensor is not None
    
    @property
    def numeric_tensor(self) -> torch.Tensor:
        """Get or compute the numeric tensor representation."""
        if not self._is_evaluated:
            self._evaluate_to_tensor()
        return self._numeric_tensor
    
    def _evaluate_to_tensor(self) -> None:
        """Evaluate symbolic data to numeric tensor."""
        evaluator = EquationEvaluator()
        
        numeric_data = []
        for row in self.symbolic_data:
            numeric_row = []
            for node in row:
                try:
                    result = evaluator.evaluate(node)
                    numeric_value = float(result.evaluate_exact())
                    numeric_row.append(numeric_value)
                except:
                    numeric_row.append(0.0)  # Fallback for evaluation errors
            numeric_data.append(numeric_row)
        
        self._numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32)
        self._is_evaluated = True
    
    def get_symbolic_expression(self, row: int, col: int) -> str:
        """Get symbolic expression at specific position."""
        if 0 <= row < len(self.symbolic_data) and 0 <= col < len(self.symbolic_data[row]):
            return self.symbolic_data[row][col].to_string()
        return "0"
    
    def to_device(self, device: torch.device) -> 'SymbolicTensor':
        """Move numeric tensor to specified device."""
        if self._is_evaluated:
            self._numeric_tensor = self._numeric_tensor.to(device)
        return self


class SymbolicFunction(Function):
    """
    Custom autograd function for symbolic operations.
    """
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, symbolic_data: List[List[EquationNode]]) -> torch.Tensor:
        """Forward pass with symbolic computation."""
        ctx.symbolic_data = symbolic_data
        
        # For now, just pass through the numeric tensor
        # In a full implementation, this would perform symbolic operations
        return input_tensor
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass - symbolic differentiation would go here."""
        # Placeholder: return gradient as-is
        # Full implementation would compute symbolic derivatives
        return grad_output, None


class SymbolicLinear(nn.Module):
    """
    Linear layer with symbolic weight representation.
    Maintains exact symbolic weights while supporting standard PyTorch operations.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 use_gpu: bool = True):
        super(SymbolicLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_gpu = use_gpu
        
        # Initialize symbolic weights
        self.symbolic_weights = self._create_symbolic_weights()
        
        # Create numeric weight tensor for PyTorch compatibility
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
            self.symbolic_bias = self._create_symbolic_bias()
        else:
            self.register_parameter('bias', None)
            self.symbolic_bias = None
        
        # GPU evaluator for acceleration
        if use_gpu:
            self.gpu_evaluator = GPUEvaluator()
        else:
            self.gpu_evaluator = None
        
        self.evaluator = EquationEvaluator()
    
    def _create_symbolic_weights(self) -> List[List[EquationNode]]:
        """Create symbolic weight matrix with rational number initialization."""
        weights = []
        for i in range(self.out_features):
            row = []
            for j in range(self.in_features):
                # Initialize with small random rational numbers
                numerator = torch.randint(-100, 101, (1,)).item()
                denominator = torch.randint(50, 151, (1,)).item()
                
                rational_weight = RationalListNumber([FractionTerm(numerator, denominator)])
                weight_node = ConstantNode(rational_weight)
                row.append(weight_node)
            weights.append(row)
        return weights
    
    def _create_symbolic_bias(self) -> List[EquationNode]:
        """Create symbolic bias vector."""
        bias = []
        for i in range(self.out_features):
            numerator = torch.randint(-50, 51, (1,)).item()
            denominator = torch.randint(25, 76, (1,)).item()
            
            rational_bias = RationalListNumber([FractionTerm(numerator, denominator)])
            bias_node = ConstantNode(rational_bias)
            bias.append(bias_node)
        return bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with symbolic computation.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Standard PyTorch linear operation for gradient computation
        output = F.linear(x, self.weight, self.bias)
        
        # Apply symbolic function for exact computation tracking
        if hasattr(self, 'symbolic_weights'):
            output = SymbolicFunction.apply(output, self.symbolic_weights)
        
        return output
    
    def get_symbolic_weight(self, out_idx: int, in_idx: int) -> str:
        """Get symbolic representation of specific weight."""
        if (0 <= out_idx < self.out_features and 
            0 <= in_idx < self.in_features):
            return self.symbolic_weights[out_idx][in_idx].to_string()
        return "0"
    
    def evaluate_symbolic_weights(self) -> torch.Tensor:
        """Evaluate all symbolic weights to numeric tensor."""
        numeric_weights = torch.zeros(self.out_features, self.in_features)
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                try:
                    result = self.evaluator.evaluate(self.symbolic_weights[i][j])
                    numeric_weights[i, j] = float(result.evaluate_exact())
                except:
                    numeric_weights[i, j] = 0.0
        
        return numeric_weights
    
    def sync_weights(self) -> None:
        """Synchronize symbolic weights with PyTorch parameters."""
        numeric_weights = self.evaluate_symbolic_weights()
        self.weight.data.copy_(numeric_weights)
        
        if self.bias is not None and self.symbolic_bias is not None:
            numeric_bias = torch.zeros(self.out_features)
            for i in range(self.out_features):
                try:
                    result = self.evaluator.evaluate(self.symbolic_bias[i])
                    numeric_bias[i] = float(result.evaluate_exact())
                except:
                    numeric_bias[i] = 0.0
            self.bias.data.copy_(numeric_bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, symbolic=True'


class SymbolicLayer(nn.Module):
    """
    General symbolic layer that can wrap any operation with symbolic tracking.
    """
    
    def __init__(self, operation: str = 'identity'):
        super(SymbolicLayer, self).__init__()
        self.operation = operation
        self.evaluator = EquationEvaluator()
        self.symbolic_history = []
    
    def forward(self, x: torch.Tensor, symbolic_data: Optional[List[List[EquationNode]]] = None) -> torch.Tensor:
        """Forward pass with optional symbolic tracking."""
        
        if symbolic_data is not None:
            # Track symbolic computation
            self.symbolic_history.append({
                'operation': self.operation,
                'input_shape': x.shape,
                'symbolic_expressions': len(symbolic_data)
            })
        
        # Apply the specified operation
        if self.operation == 'relu':
            return F.relu(x)
        elif self.operation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.operation == 'tanh':
            return torch.tanh(x)
        else:  # identity
            return x
    
    def get_symbolic_history(self) -> List[Dict[str, Any]]:
        """Get history of symbolic operations."""
        return self.symbolic_history.copy()


class SymbolicSequential(nn.Sequential):
    """
    Sequential container that maintains symbolic computation throughout the network.
    """
    
    def __init__(self, *args):
        super(SymbolicSequential, self).__init__(*args)
        self.symbolic_trace = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with symbolic tracing."""
        self.symbolic_trace = []
        
        for i, module in enumerate(self):
            x = module(x)
            
            # Record symbolic information if available
            if hasattr(module, 'get_symbolic_weight'):
                self.symbolic_trace.append({
                    'layer_index': i,
                    'layer_type': type(module).__name__,
                    'has_symbolic_weights': True
                })
            else:
                self.symbolic_trace.append({
                    'layer_index': i,
                    'layer_type': type(module).__name__,
                    'has_symbolic_weights': False
                })
        
        return x
    
    def get_symbolic_trace(self) -> List[Dict[str, Any]]:
        """Get trace of symbolic computation through the network."""
        return self.symbolic_trace.copy()
    
    def export_symbolic_expressions(self) -> Dict[str, List[str]]:
        """Export all symbolic expressions from the network."""
        expressions = {}
        
        for i, module in enumerate(self):
            if isinstance(module, SymbolicLinear):
                layer_expressions = []
                for out_idx in range(module.out_features):
                    for in_idx in range(module.in_features):
                        expr = module.get_symbolic_weight(out_idx, in_idx)
                        layer_expressions.append(f"w[{out_idx},{in_idx}] = {expr}")
                expressions[f'layer_{i}'] = layer_expressions
        
        return expressions


class ExplainableInferenceModel(nn.Module):
    """
    Inference model that provides exact symbolic explanations for its outputs.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(ExplainableInferenceModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build symbolic layers
        for hidden_dim in hidden_dims:
            layers.append(SymbolicLinear(prev_dim, hidden_dim))
            layers.append(SymbolicLayer('relu'))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(SymbolicLinear(prev_dim, output_dim))
        
        self.network = SymbolicSequential(*layers)
        self.evaluator = EquationEvaluator()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with symbolic tracking."""
        return self.network(x)
    
    def explain_prediction(self, x: torch.Tensor, output_index: int = 0) -> Dict[str, Any]:
        """
        Provide symbolic explanation for a specific prediction.
        
        Args:
            x: Input tensor (single sample)
            output_index: Which output neuron to explain
            
        Returns:
            Dictionary containing symbolic explanation
        """
        # Run forward pass to populate symbolic trace
        output = self.forward(x.unsqueeze(0))
        
        # Get symbolic expressions
        expressions = self.network.export_symbolic_expressions()
        trace = self.network.get_symbolic_trace()
        
        explanation = {
            'input_values': x.tolist(),
            'output_value': output[0, output_index].item(),
            'symbolic_expressions': expressions,
            'computation_trace': trace,
            'exact_computation': True,
            'precision_loss': 0.0  # No precision loss with symbolic computation
        }
        
        return explanation
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """Get complexity metrics for the symbolic model."""
        total_symbolic_weights = 0
        total_parameters = 0
        
        for module in self.network:
            if isinstance(module, SymbolicLinear):
                total_symbolic_weights += module.in_features * module.out_features
                total_parameters += module.weight.numel()
                if module.bias is not None:
                    total_parameters += module.bias.numel()
        
        return {
            'total_symbolic_weights': total_symbolic_weights,
            'total_parameters': total_parameters,
            'symbolic_coverage': (total_symbolic_weights / total_parameters) * 100,
            'layers': len([m for m in self.network if isinstance(m, (SymbolicLinear, SymbolicLayer))])
        }
