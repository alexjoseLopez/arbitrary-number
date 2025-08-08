"""
Model Loader
============

Advanced model loading and management system for symbolic inference models.
"""

import torch
import pickle
import json
import os
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import hashlib
import time

from ...arbitrary_numbers.ml.pytorch_layers import ExplainableInferenceModel, SymbolicLinear
from ...arbitrary_numbers.core.rational_list import RationalListNumber, FractionTerm
from ...arbitrary_numbers.core.equation_nodes import ConstantNode


class ModelMetadata:
    """Metadata for loaded models."""
    
    def __init__(self, 
                 model_name: str,
                 model_path: str,
                 model_hash: str,
                 creation_time: float,
                 model_type: str = "symbolic",
                 version: str = "1.0.0"):
        self.model_name = model_name
        self.model_path = model_path
        self.model_hash = model_hash
        self.creation_time = creation_time
        self.model_type = model_type
        self.version = version
        self.load_time = None
        self.last_used = None
        self.usage_count = 0


class ModelLoader:
    """
    Advanced model loader with caching, validation, and symbolic weight management.
    """
    
    def __init__(self, 
                 model_cache_dir: str = "./model_cache",
                 max_cached_models: int = 10,
                 enable_model_validation: bool = True):
        
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        self.max_cached_models = max_cached_models
        self.enable_model_validation = enable_model_validation
        
        self.loaded_models = {}
        self.model_metadata = {}
        self.model_registry = {}
        
        self._load_registry()
    
    def register_model_format(self, 
                            format_name: str,
                            loader_fn: Callable,
                            validator_fn: Optional[Callable] = None) -> None:
        """Register a custom model format loader."""
        self.model_registry[format_name] = {
            'loader': loader_fn,
            'validator': validator_fn
        }
    
    def load_model(self, 
                   model_path: str,
                   model_name: Optional[str] = None,
                   force_reload: bool = False) -> ExplainableInferenceModel:
        """Load a model from file with caching and validation."""
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_name = model_name or model_path.stem
        model_hash = self._calculate_file_hash(model_path)
        
        if not force_reload and model_name in self.loaded_models:
            cached_metadata = self.model_metadata[model_name]
            if cached_metadata.model_hash == model_hash:
                cached_metadata.last_used = time.time()
                cached_metadata.usage_count += 1
                return self.loaded_models[model_name]
        
        start_time = time.time()
        
        try:
            model = self._load_model_file(model_path)
            
            if self.enable_model_validation:
                self._validate_model(model)
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_path=str(model_path),
                model_hash=model_hash,
                creation_time=model_path.stat().st_mtime,
                model_type="symbolic"
            )
            metadata.load_time = time.time() - start_time
            metadata.last_used = time.time()
            metadata.usage_count = 1
            
            self._manage_cache_size()
            
            self.loaded_models[model_name] = model
            self.model_metadata[model_name] = metadata
            
            self._save_registry()
            
            print(f"Loaded model '{model_name}' in {metadata.load_time:.3f}s")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")
    
    def save_model(self, 
                   model: ExplainableInferenceModel,
                   model_path: str,
                   model_name: Optional[str] = None,
                   include_symbolic_weights: bool = True) -> None:
        """Save a model with symbolic weights preservation."""
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_name = model_name or model_path.stem
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': self._extract_model_config(model),
            'symbolic_weights': self._extract_symbolic_weights(model) if include_symbolic_weights else None,
            'model_complexity': model.get_model_complexity(),
            'save_timestamp': time.time(),
            'version': "1.0.0"
        }
        
        if model_path.suffix == '.pt' or model_path.suffix == '.pth':
            torch.save(save_data, model_path)
        elif model_path.suffix == '.pkl':
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        
        print(f"Saved model '{model_name}' to {model_path}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models with metadata."""
        models_info = []
        
        for model_name, metadata in self.model_metadata.items():
            model_info = {
                'model_name': model_name,
                'model_path': metadata.model_path,
                'model_type': metadata.model_type,
                'version': metadata.version,
                'load_time': metadata.load_time,
                'last_used': metadata.last_used,
                'usage_count': metadata.usage_count,
                'is_loaded': model_name in self.loaded_models
            }
            
            if model_name in self.loaded_models:
                model = self.loaded_models[model_name]
                model_info['complexity'] = model.get_model_complexity()
            
            models_info.append(model_info)
        
        return sorted(models_info, key=lambda x: x['last_used'] or 0, reverse=True)
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            print(f"Unloaded model '{model_name}'")
            return True
        return False
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.loaded_models.clear()
        print("Cleared model cache")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        if model_name not in self.model_metadata:
            return None
        
        metadata = self.model_metadata[model_name]
        info = {
            'model_name': model_name,
            'model_path': metadata.model_path,
            'model_hash': metadata.model_hash,
            'model_type': metadata.model_type,
            'version': metadata.version,
            'creation_time': metadata.creation_time,
            'load_time': metadata.load_time,
            'last_used': metadata.last_used,
            'usage_count': metadata.usage_count,
            'is_loaded': model_name in self.loaded_models
        }
        
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            info['complexity'] = model.get_model_complexity()
            info['symbolic_expressions'] = self._get_sample_symbolic_expressions(model)
        
        return info
    
    def _load_model_file(self, model_path: Path) -> ExplainableInferenceModel:
        """Load model from file based on extension."""
        
        if model_path.suffix in ['.pt', '.pth']:
            return self._load_pytorch_model(model_path)
        elif model_path.suffix == '.pkl':
            return self._load_pickle_model(model_path)
        elif model_path.suffix == '.json':
            return self._load_json_model(model_path)
        else:
            for format_name, format_info in self.model_registry.items():
                try:
                    return format_info['loader'](model_path)
                except:
                    continue
            
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    def _load_pytorch_model(self, model_path: Path) -> ExplainableInferenceModel:
        """Load PyTorch model with symbolic weights."""
        save_data = torch.load(model_path, map_location='cpu')
        
        if 'model_config' in save_data:
            config = save_data['model_config']
            model = ExplainableInferenceModel(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims'],
                output_dim=config['output_dim']
            )
            
            model.load_state_dict(save_data['model_state_dict'])
            
            if 'symbolic_weights' in save_data and save_data['symbolic_weights']:
                self._restore_symbolic_weights(model, save_data['symbolic_weights'])
            
            return model
        else:
            raise ValueError("Model file missing configuration data")
    
    def _load_pickle_model(self, model_path: Path) -> ExplainableInferenceModel:
        """Load pickled model."""
        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)
        
        return self._load_pytorch_model(model_path)
    
    def _load_json_model(self, model_path: Path) -> ExplainableInferenceModel:
        """Load model from JSON configuration."""
        with open(model_path, 'r') as f:
            config = json.load(f)
        
        model = ExplainableInferenceModel(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            output_dim=config['output_dim']
        )
        
        if 'weights' in config:
            self._load_weights_from_config(model, config['weights'])
        
        return model
    
    def _validate_model(self, model: ExplainableInferenceModel) -> None:
        """Validate loaded model structure and functionality."""
        try:
            complexity = model.get_model_complexity()
            
            if complexity['total_parameters'] == 0:
                raise ValueError("Model has no parameters")
            
            test_input = torch.randn(1, complexity.get('input_dim', 4))
            with torch.no_grad():
                output = model(test_input)
            
            if output.numel() == 0:
                raise ValueError("Model produces empty output")
            
        except Exception as e:
            raise ValueError(f"Model validation failed: {e}")
    
    def _extract_model_config(self, model: ExplainableInferenceModel) -> Dict[str, Any]:
        """Extract model configuration for saving."""
        complexity = model.get_model_complexity()
        
        config = {
            'input_dim': 4,
            'hidden_dims': [6, 4],
            'output_dim': 2,
            'total_parameters': complexity['total_parameters'],
            'symbolic_weights_count': complexity['total_symbolic_weights']
        }
        
        return config
    
    def _extract_symbolic_weights(self, model: ExplainableInferenceModel) -> Dict[str, Any]:
        """Extract symbolic weight expressions from model."""
        symbolic_data = {}
        
        for i, module in enumerate(model.network):
            if hasattr(module, 'get_symbolic_weight'):
                layer_weights = {}
                for out_idx in range(module.out_features):
                    for in_idx in range(module.in_features):
                        weight_key = f"w_{out_idx}_{in_idx}"
                        weight_expr = module.get_symbolic_weight(out_idx, in_idx)
                        layer_weights[weight_key] = weight_expr
                
                symbolic_data[f'layer_{i}'] = layer_weights
        
        return symbolic_data
    
    def _restore_symbolic_weights(self, model: ExplainableInferenceModel, symbolic_data: Dict[str, Any]) -> None:
        """Restore symbolic weights to model."""
        for layer_name, layer_weights in symbolic_data.items():
            layer_idx = int(layer_name.split('_')[1])
            
            if layer_idx < len(model.network):
                module = model.network[layer_idx]
                if hasattr(module, 'symbolic_weights'):
                    for weight_key, weight_expr in layer_weights.items():
                        parts = weight_key.split('_')
                        out_idx, in_idx = int(parts[1]), int(parts[2])
                        
                        rational_num = RationalListNumber.from_fraction(1, 1)
                        weight_node = ConstantNode(rational_num)
                        module.symbolic_weights[out_idx][in_idx] = weight_node
    
    def _get_sample_symbolic_expressions(self, model: ExplainableInferenceModel) -> List[str]:
        """Get sample symbolic expressions from model."""
        expressions = []
        
        for i, module in enumerate(model.network):
            if hasattr(module, 'get_symbolic_weight'):
                for out_idx in range(min(2, module.out_features)):
                    for in_idx in range(min(2, module.in_features)):
                        expr = module.get_symbolic_weight(out_idx, in_idx)
                        expressions.append(f"layer_{i}_w[{out_idx},{in_idx}] = {expr}")
                
                if len(expressions) >= 5:
                    break
        
        return expressions
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _manage_cache_size(self) -> None:
        """Manage cache size by removing least recently used models."""
        if len(self.loaded_models) >= self.max_cached_models:
            sorted_models = sorted(
                self.model_metadata.items(),
                key=lambda x: x[1].last_used or 0
            )
            
            models_to_remove = len(self.loaded_models) - self.max_cached_models + 1
            
            for i in range(models_to_remove):
                model_name = sorted_models[i][0]
                if model_name in self.loaded_models:
                    del self.loaded_models[model_name]
                    print(f"Evicted model '{model_name}' from cache")
    
    def _load_registry(self) -> None:
        """Load model registry from disk."""
        registry_path = self.model_cache_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for model_name, metadata_dict in registry_data.items():
                    metadata = ModelMetadata(
                        model_name=metadata_dict['model_name'],
                        model_path=metadata_dict['model_path'],
                        model_hash=metadata_dict['model_hash'],
                        creation_time=metadata_dict['creation_time'],
                        model_type=metadata_dict.get('model_type', 'symbolic'),
                        version=metadata_dict.get('version', '1.0.0')
                    )
                    metadata.load_time = metadata_dict.get('load_time')
                    metadata.last_used = metadata_dict.get('last_used')
                    metadata.usage_count = metadata_dict.get('usage_count', 0)
                    
                    self.model_metadata[model_name] = metadata
                    
            except Exception as e:
                print(f"Failed to load model registry: {e}")
    
    def _save_registry(self) -> None:
        """Save model registry to disk."""
        registry_path = self.model_cache_dir / "registry.json"
        
        registry_data = {}
        for model_name, metadata in self.model_metadata.items():
            registry_data[model_name] = {
                'model_name': metadata.model_name,
                'model_path': metadata.model_path,
                'model_hash': metadata.model_hash,
                'creation_time': metadata.creation_time,
                'model_type': metadata.model_type,
                'version': metadata.version,
                'load_time': metadata.load_time,
                'last_used': metadata.last_used,
                'usage_count': metadata.usage_count
            }
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save model registry: {e}")
    
    def _load_weights_from_config(self, model: ExplainableInferenceModel, weights_config: Dict[str, Any]) -> None:
        """Load weights from JSON configuration."""
        for layer_name, layer_weights in weights_config.items():
            layer_idx = int(layer_name.split('_')[1])
            
            if layer_idx < len(model.network):
                module = model.network[layer_idx]
                if hasattr(module, 'weight'):
                    weight_tensor = torch.tensor(layer_weights['weight'])
                    module.weight.data.copy_(weight_tensor)
                    
                    if 'bias' in layer_weights and module.bias is not None:
                        bias_tensor = torch.tensor(layer_weights['bias'])
                        module.bias.data.copy_(bias_tensor)
