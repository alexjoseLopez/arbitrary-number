"""
Equation Evaluator
=================

Core evaluation engine for processing equation trees with optimization
and caching capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple
import time
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from .equation_nodes import EquationNode, ConstantNode, BinaryOpNode, UnaryOpNode
from .rational_list import RationalListNumber


class EvaluationCache:
    """
    Thread-safe cache for storing evaluated expressions.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[RationalListNumber]:
        """Get cached result if available."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: RationalListNumber) -> None:
        """Store result in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_oldest(self) -> None:
        """Remove least recently used item."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def size(self) -> int:
        """Return current cache size."""
        with self.lock:
            return len(self.cache)


class EquationEvaluator:
    """
    Main evaluation engine for equation trees.
    Supports caching, optimization, and parallel evaluation.
    """
    
    def __init__(self, cache_size: int = 10000, enable_optimization: bool = True):
        self.cache = EvaluationCache(cache_size)
        self.enable_optimization = enable_optimization
        self.evaluation_stats = defaultdict(int)
        self.lock = threading.RLock()
    
    def evaluate(self, 
                 node: EquationNode, 
                 variables: Optional[Dict[str, Any]] = None,
                 use_cache: bool = True) -> RationalListNumber:
        """
        Evaluate an equation node to a RationalListNumber.
        
        Args:
            node: The equation node to evaluate
            variables: Variable bindings for evaluation
            use_cache: Whether to use caching for this evaluation
            
        Returns:
            RationalListNumber result of evaluation
        """
        variables = variables or {}
        
        if use_cache:
            cache_key = self._generate_cache_key(node, variables)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.evaluation_stats['cache_hits'] += 1
                return cached_result
        
        self.evaluation_stats['cache_misses'] += 1
        
        if self.enable_optimization:
            optimized_node = self._optimize_node(node)
        else:
            optimized_node = node
        
        result = self._evaluate_node(optimized_node, variables)
        
        if use_cache:
            self.cache.put(cache_key, result)
        
        return result
    
    def evaluate_batch(self, 
                      nodes: List[EquationNode],
                      variables: Optional[Dict[str, Any]] = None,
                      max_workers: int = 4) -> List[RationalListNumber]:
        """
        Evaluate multiple equation nodes in parallel.
        
        Args:
            nodes: List of equation nodes to evaluate
            variables: Variable bindings for evaluation
            max_workers: Maximum number of worker threads
            
        Returns:
            List of RationalListNumber results
        """
        variables = variables or {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.evaluate, node, variables)
                for node in nodes
            ]
            
            results = [future.result() for future in futures]
        
        return results
    
    def _evaluate_node(self, node: EquationNode, variables: Dict[str, Any]) -> RationalListNumber:
        """Internal node evaluation without caching."""
        return node.evaluate(variables)
    
    def _optimize_node(self, node: EquationNode) -> EquationNode:
        """
        Apply optimization transformations to the equation tree.
        """
        if isinstance(node, ConstantNode):
            return node
        
        elif isinstance(node, BinaryOpNode):
            left_opt = self._optimize_node(node.left)
            right_opt = self._optimize_node(node.right)
            
            return self._optimize_binary_op(left_opt, right_opt, node.operation)
        
        elif isinstance(node, UnaryOpNode):
            operand_opt = self._optimize_node(node.operand)
            return self._optimize_unary_op(operand_opt, node.operation)
        
        else:
            return node
    
    def _optimize_binary_op(self, left: EquationNode, right: EquationNode, op: str) -> EquationNode:
        """Optimize binary operations."""
        
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            temp_node = BinaryOpNode(left, right, op)
            result = temp_node.evaluate()
            return ConstantNode(result)
        
        if op == 'add':
            if isinstance(left, ConstantNode) and left.value.evaluate_exact() == 0:
                return right
            if isinstance(right, ConstantNode) and right.value.evaluate_exact() == 0:
                return left
        
        elif op == 'mul':
            if isinstance(left, ConstantNode):
                val = left.value.evaluate_exact()
                if val == 0:
                    return ConstantNode.from_int(0)
                elif val == 1:
                    return right
            
            if isinstance(right, ConstantNode):
                val = right.value.evaluate_exact()
                if val == 0:
                    return ConstantNode.from_int(0)
                elif val == 1:
                    return left
        
        elif op == 'pow':
            if isinstance(right, ConstantNode):
                val = right.value.evaluate_exact()
                if val == 0:
                    return ConstantNode.from_int(1)
                elif val == 1:
                    return left
        
        return BinaryOpNode(left, right, op)
    
    def _optimize_unary_op(self, operand: EquationNode, op: str) -> EquationNode:
        """Optimize unary operations."""
        
        if isinstance(operand, ConstantNode):
            temp_node = UnaryOpNode(operand, op)
            result = temp_node.evaluate()
            return ConstantNode(result)
        
        if op == 'neg':
            if isinstance(operand, UnaryOpNode) and operand.operation == 'neg':
                return operand.operand
        
        return UnaryOpNode(operand, op)
    
    def _generate_cache_key(self, node: EquationNode, variables: Dict[str, Any]) -> str:
        """Generate a unique cache key for the node and variables."""
        node_str = node.to_string()
        var_str = str(sorted(variables.items())) if variables else ""
        return f"{node_str}|{var_str}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        with self.lock:
            total_evaluations = self.evaluation_stats['cache_hits'] + self.evaluation_stats['cache_misses']
            hit_rate = (self.evaluation_stats['cache_hits'] / total_evaluations * 100) if total_evaluations > 0 else 0
            
            return {
                'total_evaluations': total_evaluations,
                'cache_hits': self.evaluation_stats['cache_hits'],
                'cache_misses': self.evaluation_stats['cache_misses'],
                'cache_hit_rate': f"{hit_rate:.2f}%",
                'cache_size': self.cache.size(),
                'optimization_enabled': self.enable_optimization
            }
    
    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self.cache.clear()
    
    def reset_stats(self) -> None:
        """Reset evaluation statistics."""
        with self.lock:
            self.evaluation_stats.clear()


class BackgroundEvaluator:
    """
    Background evaluation engine that pre-computes expressions
    during idle CPU time.
    """
    
    def __init__(self, evaluator: EquationEvaluator):
        self.evaluator = evaluator
        self.pending_evaluations = []
        self.is_running = False
        self.worker_thread = None
        self.lock = threading.RLock()
    
    def queue_evaluation(self, node: EquationNode, variables: Optional[Dict[str, Any]] = None) -> None:
        """Queue an expression for background evaluation."""
        with self.lock:
            self.pending_evaluations.append((node, variables or {}))
            
            if not self.is_running:
                self.start()
    
    def start(self) -> None:
        """Start the background evaluation thread."""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
    
    def stop(self) -> None:
        """Stop the background evaluation thread."""
        with self.lock:
            self.is_running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=1.0)
    
    def _worker_loop(self) -> None:
        """Main worker loop for background evaluation."""
        while self.is_running:
            try:
                with self.lock:
                    if not self.pending_evaluations:
                        time.sleep(0.1)
                        continue
                    
                    node, variables = self.pending_evaluations.pop(0)
                
                self.evaluator.evaluate(node, variables, use_cache=True)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Background evaluation error: {e}")
                time.sleep(0.1)
    
    def pending_count(self) -> int:
        """Return number of pending evaluations."""
        with self.lock:
            return len(self.pending_evaluations)
