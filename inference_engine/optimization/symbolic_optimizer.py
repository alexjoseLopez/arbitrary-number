"""
Symbolic Optimizer
==================

Advanced optimization engine for symbolic expressions and inference models.
"""

import torch
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from ...arbitrary_numbers.core.equation_nodes import (
    EquationNode, ConstantNode, BinaryOpNode, UnaryOpNode, VariableNode
)
from ...arbitrary_numbers.core.rational_list import RationalListNumber, FractionTerm
from ...arbitrary_numbers.core.evaluator import EquationEvaluator
from ...arbitrary_numbers.ml.pytorch_layers import ExplainableInferenceModel, SymbolicLinear


@dataclass
class OptimizationRule:
    """Single optimization rule for symbolic expressions."""
    name: str
    pattern_matcher: callable
    transformer: callable
    priority: int = 1
    enabled: bool = True


class OptimizationStats:
    """Statistics for optimization operations."""
    
    def __init__(self):
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.failed_optimizations = 0
        self.rules_applied = {}
        self.complexity_reductions = []
        self.optimization_time = 0.0
        self.expressions_processed = 0
    
    def record_optimization(self, rule_name: str, success: bool, 
                          complexity_before: int, complexity_after: int, 
                          time_taken: float):
        """Record optimization attempt."""
        self.total_optimizations += 1
        self.optimization_time += time_taken
        
        if success:
            self.successful_optimizations += 1
            reduction = complexity_before - complexity_after
            self.complexity_reductions.append(reduction)
            
            if rule_name not in self.rules_applied:
                self.rules_applied[rule_name] = 0
            self.rules_applied[rule_name] += 1
        else:
            self.failed_optimizations += 1
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        avg_reduction = (sum(self.complexity_reductions) / 
                        len(self.complexity_reductions)) if self.complexity_reductions else 0
        
        return {
            'total_optimizations': self.total_optimizations,
            'successful_optimizations': self.successful_optimizations,
            'failed_optimizations': self.failed_optimizations,
            'success_rate': (self.successful_optimizations / 
                           max(1, self.total_optimizations)) * 100,
            'rules_applied': self.rules_applied,
            'average_complexity_reduction': avg_reduction,
            'total_optimization_time': self.optimization_time,
            'expressions_processed': self.expressions_processed,
            'optimizations_per_second': (self.total_optimizations / 
                                       max(0.001, self.optimization_time))
        }


class SymbolicOptimizer:
    """
    Advanced symbolic optimizer for exact computation expressions.
    Optimized for consumer 32GB Nvidia GPUs.
    """
    
    def __init__(self, 
                 max_optimization_depth: int = 10,
                 enable_parallel_optimization: bool = True,
                 worker_threads: int = 4):
        
        self.max_optimization_depth = max_optimization_depth
        self.enable_parallel_optimization = enable_parallel_optimization
        self.worker_threads = worker_threads
        
        self.optimization_rules = []
        self.stats = OptimizationStats()
        self.evaluator = EquationEvaluator()
        
        self.expression_cache = {}
        self.cache_lock = threading.RLock()
        
        if enable_parallel_optimization:
            self.executor = ThreadPoolExecutor(max_workers=worker_threads)
        else:
            self.executor = None
        
        self._initialize_default_rules()
    
    def add_optimization_rule(self, rule: OptimizationRule) -> None:
        """Add custom optimization rule."""
        self.optimization_rules.append(rule)
        self.optimization_rules.sort(key=lambda r: r.priority, reverse=True)
        print(f"Added optimization rule: {rule.name}")
    
    def optimize_expression(self, expression: EquationNode, 
                          max_iterations: int = None) -> EquationNode:
        """Optimize a single symbolic expression."""
        if max_iterations is None:
            max_iterations = self.max_optimization_depth
        
        start_time = time.time()
        original_complexity = expression.complexity()
        
        optimized = self._optimize_recursive(expression, 0, max_iterations)
        
        final_complexity = optimized.complexity()
        optimization_time = time.time() - start_time
        
        self.stats.record_optimization(
            "expression_optimization", 
            final_complexity < original_complexity,
            original_complexity, 
            final_complexity, 
            optimization_time
        )
        
        return optimized
    
    def optimize_model(self, model: ExplainableInferenceModel) -> ExplainableInferenceModel:
        """Optimize all symbolic expressions in a model."""
        start_time = time.time()
        
        optimized_count = 0
        
        for module in model.network:
            if isinstance(module, SymbolicLinear):
                optimized_count += self._optimize_symbolic_layer(module)
        
        optimization_time = time.time() - start_time
        
        print(f"Optimized {optimized_count} symbolic expressions in {optimization_time:.3f}s")
        
        return model
    
    def optimize_batch_expressions(self, expressions: List[EquationNode]) -> List[EquationNode]:
        """Optimize multiple expressions in parallel."""
        if not self.enable_parallel_optimization or len(expressions) < 4:
            return [self.optimize_expression(expr) for expr in expressions]
        
        start_time = time.time()
        
        futures = []
        for expr in expressions:
            future = self.executor.submit(self.optimize_expression, expr)
            futures.append(future)
        
        optimized_expressions = []
        for future in futures:
            try:
                optimized_expr = future.result(timeout=30.0)
                optimized_expressions.append(optimized_expr)
            except Exception as e:
                print(f"Optimization failed: {e}")
                optimized_expressions.append(expressions[len(optimized_expressions)])
        
        batch_time = time.time() - start_time
        print(f"Batch optimized {len(expressions)} expressions in {batch_time:.3f}s")
        
        return optimized_expressions
    
    def analyze_expression_complexity(self, expression: EquationNode) -> Dict[str, Any]:
        """Analyze expression complexity and optimization potential."""
        complexity = expression.complexity()
        
        node_counts = self._count_node_types(expression)
        
        optimization_potential = self._estimate_optimization_potential(expression)
        
        return {
            'total_complexity': complexity,
            'node_counts': node_counts,
            'optimization_potential': optimization_potential,
            'estimated_reduction': optimization_potential * 0.3,
            'recommended_rules': self._recommend_optimization_rules(expression)
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return self.stats.get_stats_dict()
    
    def clear_cache(self) -> None:
        """Clear expression optimization cache."""
        with self.cache_lock:
            self.expression_cache.clear()
        print("Cleared optimization cache")
    
    def _optimize_recursive(self, expression: EquationNode, 
                          depth: int, max_depth: int) -> EquationNode:
        """Recursively optimize expression using rules."""
        if depth >= max_depth:
            return expression
        
        expr_hash = self._hash_expression(expression)
        
        with self.cache_lock:
            if expr_hash in self.expression_cache:
                return self.expression_cache[expr_hash]
        
        current_expr = expression
        
        for rule in self.optimization_rules:
            if not rule.enabled:
                continue
            
            try:
                if rule.pattern_matcher(current_expr):
                    start_time = time.time()
                    
                    optimized = rule.transformer(current_expr)
                    
                    if optimized and optimized.complexity() < current_expr.complexity():
                        rule_time = time.time() - start_time
                        
                        self.stats.record_optimization(
                            rule.name, True,
                            current_expr.complexity(),
                            optimized.complexity(),
                            rule_time
                        )
                        
                        current_expr = optimized
                        
                        current_expr = self._optimize_recursive(current_expr, depth + 1, max_depth)
                        break
                        
            except Exception as e:
                print(f"Rule {rule.name} failed: {e}")
                continue
        
        if isinstance(current_expr, BinaryOpNode):
            left_optimized = self._optimize_recursive(current_expr.left, depth + 1, max_depth)
            right_optimized = self._optimize_recursive(current_expr.right, depth + 1, max_depth)
            
            if (left_optimized != current_expr.left or 
                right_optimized != current_expr.right):
                current_expr = BinaryOpNode(left_optimized, right_optimized, current_expr.operation)
        
        elif isinstance(current_expr, UnaryOpNode):
            operand_optimized = self._optimize_recursive(current_expr.operand, depth + 1, max_depth)
            
            if operand_optimized != current_expr.operand:
                current_expr = UnaryOpNode(operand_optimized, current_expr.operation)
        
        with self.cache_lock:
            self.expression_cache[expr_hash] = current_expr
        
        return current_expr
    
    def _optimize_symbolic_layer(self, layer: SymbolicLinear) -> int:
        """Optimize symbolic weights in a layer."""
        optimized_count = 0
        
        for i in range(layer.out_features):
            for j in range(layer.in_features):
                if hasattr(layer, 'symbolic_weights'):
                    original_weight = layer.symbolic_weights[i][j]
                    optimized_weight = self.optimize_expression(original_weight)
                    
                    if optimized_weight.complexity() < original_weight.complexity():
                        layer.symbolic_weights[i][j] = optimized_weight
                        optimized_count += 1
        
        return optimized_count
    
    def _initialize_default_rules(self) -> None:
        """Initialize default optimization rules."""
        
        self.add_optimization_rule(OptimizationRule(
            name="zero_addition",
            pattern_matcher=self._is_zero_addition,
            transformer=self._remove_zero_addition,
            priority=10
        ))
        
        self.add_optimization_rule(OptimizationRule(
            name="zero_multiplication",
            pattern_matcher=self._is_zero_multiplication,
            transformer=self._simplify_zero_multiplication,
            priority=10
        ))
        
        self.add_optimization_rule(OptimizationRule(
            name="one_multiplication",
            pattern_matcher=self._is_one_multiplication,
            transformer=self._remove_one_multiplication,
            priority=9
        ))
        
        self.add_optimization_rule(OptimizationRule(
            name="constant_folding",
            pattern_matcher=self._is_constant_operation,
            transformer=self._fold_constants,
            priority=8
        ))
        
        self.add_optimization_rule(OptimizationRule(
            name="rational_simplification",
            pattern_matcher=self._is_rational_simplifiable,
            transformer=self._simplify_rationals,
            priority=7
        ))
        
        self.add_optimization_rule(OptimizationRule(
            name="associative_reordering",
            pattern_matcher=self._is_associative_reorderable,
            transformer=self._reorder_associative,
            priority=6
        ))
    
    def _is_zero_addition(self, expr: EquationNode) -> bool:
        """Check if expression is addition with zero."""
        if not isinstance(expr, BinaryOpNode) or expr.operation != 'add':
            return False
        
        return (self._is_zero_constant(expr.left) or 
                self._is_zero_constant(expr.right))
    
    def _remove_zero_addition(self, expr: BinaryOpNode) -> EquationNode:
        """Remove zero from addition."""
        if self._is_zero_constant(expr.left):
            return expr.right
        elif self._is_zero_constant(expr.right):
            return expr.left
        return expr
    
    def _is_zero_multiplication(self, expr: EquationNode) -> bool:
        """Check if expression is multiplication with zero."""
        if not isinstance(expr, BinaryOpNode) or expr.operation != 'mul':
            return False
        
        return (self._is_zero_constant(expr.left) or 
                self._is_zero_constant(expr.right))
    
    def _simplify_zero_multiplication(self, expr: BinaryOpNode) -> EquationNode:
        """Simplify multiplication by zero."""
        zero_rational = RationalListNumber.from_int(0)
        return ConstantNode(zero_rational)
    
    def _is_one_multiplication(self, expr: EquationNode) -> bool:
        """Check if expression is multiplication with one."""
        if not isinstance(expr, BinaryOpNode) or expr.operation != 'mul':
            return False
        
        return (self._is_one_constant(expr.left) or 
                self._is_one_constant(expr.right))
    
    def _remove_one_multiplication(self, expr: BinaryOpNode) -> EquationNode:
        """Remove one from multiplication."""
        if self._is_one_constant(expr.left):
            return expr.right
        elif self._is_one_constant(expr.right):
            return expr.left
        return expr
    
    def _is_constant_operation(self, expr: EquationNode) -> bool:
        """Check if expression is operation between constants."""
        if not isinstance(expr, BinaryOpNode):
            return False
        
        return (isinstance(expr.left, ConstantNode) and 
                isinstance(expr.right, ConstantNode))
    
    def _fold_constants(self, expr: BinaryOpNode) -> EquationNode:
        """Fold constant operations."""
        try:
            result = self.evaluator.evaluate(expr)
            return ConstantNode(result)
        except:
            return expr
    
    def _is_rational_simplifiable(self, expr: EquationNode) -> bool:
        """Check if rational expression can be simplified."""
        if isinstance(expr, ConstantNode):
            return len(expr.value.terms) > 1
        return False
    
    def _simplify_rationals(self, expr: ConstantNode) -> EquationNode:
        """Simplify rational number representation."""
        simplified = expr.value.simplify()
        return ConstantNode(simplified)
    
    def _is_associative_reorderable(self, expr: EquationNode) -> bool:
        """Check if associative operations can be reordered."""
        if not isinstance(expr, BinaryOpNode):
            return False
        
        if expr.operation not in ['add', 'mul']:
            return False
        
        return isinstance(expr.left, BinaryOpNode) and expr.left.operation == expr.operation
    
    def _reorder_associative(self, expr: BinaryOpNode) -> EquationNode:
        """Reorder associative operations for better optimization."""
        if isinstance(expr.left, BinaryOpNode) and expr.left.operation == expr.operation:
            if isinstance(expr.left.right, ConstantNode) and isinstance(expr.right, ConstantNode):
                combined_constant = BinaryOpNode(expr.left.right, expr.right, expr.operation)
                folded = self._fold_constants(combined_constant)
                return BinaryOpNode(expr.left.left, folded, expr.operation)
        
        return expr
    
    def _is_zero_constant(self, expr: EquationNode) -> bool:
        """Check if expression is zero constant."""
        if not isinstance(expr, ConstantNode):
            return False
        
        try:
            return expr.value.evaluate_exact() == 0
        except:
            return False
    
    def _is_one_constant(self, expr: EquationNode) -> bool:
        """Check if expression is one constant."""
        if not isinstance(expr, ConstantNode):
            return False
        
        try:
            return expr.value.evaluate_exact() == 1
        except:
            return False
    
    def _count_node_types(self, expr: EquationNode) -> Dict[str, int]:
        """Count different types of nodes in expression."""
        counts = {
            'constant': 0,
            'variable': 0,
            'binary_op': 0,
            'unary_op': 0
        }
        
        def count_recursive(node):
            if isinstance(node, ConstantNode):
                counts['constant'] += 1
            elif isinstance(node, VariableNode):
                counts['variable'] += 1
            elif isinstance(node, BinaryOpNode):
                counts['binary_op'] += 1
                count_recursive(node.left)
                count_recursive(node.right)
            elif isinstance(node, UnaryOpNode):
                counts['unary_op'] += 1
                count_recursive(node.operand)
        
        count_recursive(expr)
        return counts
    
    def _estimate_optimization_potential(self, expr: EquationNode) -> float:
        """Estimate optimization potential of expression."""
        node_counts = self._count_node_types(expr)
        
        potential = 0.0
        
        potential += node_counts['constant'] * 0.3
        potential += node_counts['binary_op'] * 0.2
        potential += node_counts['unary_op'] * 0.1
        
        if isinstance(expr, ConstantNode) and len(expr.value.terms) > 1:
            potential += 0.5
        
        return min(1.0, potential)
    
    def _recommend_optimization_rules(self, expr: EquationNode) -> List[str]:
        """Recommend optimization rules for expression."""
        recommendations = []
        
        if self._is_zero_addition(expr):
            recommendations.append("zero_addition")
        
        if self._is_zero_multiplication(expr):
            recommendations.append("zero_multiplication")
        
        if self._is_one_multiplication(expr):
            recommendations.append("one_multiplication")
        
        if self._is_constant_operation(expr):
            recommendations.append("constant_folding")
        
        if self._is_rational_simplifiable(expr):
            recommendations.append("rational_simplification")
        
        return recommendations
    
    def _hash_expression(self, expr: EquationNode) -> str:
        """Generate hash for expression caching."""
        return str(hash(expr.to_string()))
