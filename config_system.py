#!/usr/bin/env python3
"""
Configuration system for LLM model management and experimentation.
Supports loading YAML configs, validation, and metadata tracking.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelConfig:
    """Structured representation of a model configuration."""
    name: str
    description: str
    provider: str
    model: str
    type: str  # "switches", "labels", "preds"
    version: str
    parameters: Dict[str, Any]
    prompt_template: Optional[str]
    output_format: Dict[str, Any]
    created_date: str
    author: str
    notes: str
    include_irt: bool = False  # Whether to include inter-item response times in prompts
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(
            name=config_dict['name'],
            description=config_dict['description'],
            provider=config_dict['provider'],
            model=config_dict['model'],
            type=config_dict['type'],
            version=config_dict['version'],
            parameters=config_dict.get('parameters', {}),
            prompt_template=config_dict.get('prompt_template'),
            output_format=config_dict.get('output_format', {}),
            created_date=config_dict.get('created_date', ''),
            author=config_dict.get('author', ''),
            notes=config_dict.get('notes', ''),
            include_irt=config_dict.get('include_irt', False)  # Backward compatible default
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'provider': self.provider,
            'model': self.model,
            'type': self.type,
            'version': self.version,
            'parameters': self.parameters,
            'prompt_template': self.prompt_template,
            'output_format': self.output_format,
            'created_date': self.created_date,
            'author': self.author,
            'notes': self.notes,
            'include_irt': self.include_irt
        }


class ConfigManager:
    """Manages model configurations and metadata tracking."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.configs: Dict[str, ModelConfig] = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all YAML configuration files from the config directory."""
        if not self.config_dir.exists():
            print(f"Warning: Config directory {self.config_dir} does not exist")
            return
        
        yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        
        for config_file in yaml_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                config = ModelConfig.from_dict(config_data)
                self.configs[config.name] = config
                print(f"Loaded config: {config.name}")
                
            except Exception as e:
                print(f"Failed to load {config_file}: {e}")
    
    def get_config(self, name: str) -> Optional[ModelConfig]:
        """Get a specific configuration by name."""
        return self.configs.get(name)
    
    def list_configs(self, config_type: Optional[str] = None) -> List[ModelConfig]:
        """List all configurations, optionally filtered by type."""
        configs = list(self.configs.values())
        
        if config_type:
            configs = [c for c in configs if c.type == config_type]
        
        return sorted(configs, key=lambda c: c.name)
    
    def list_by_provider(self, provider: str) -> List[ModelConfig]:
        """List configurations by provider."""
        return [c for c in self.configs.values() if c.provider == provider]
    
    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate a configuration and return list of issues."""
        issues = []
        
        # Required fields
        if not config.name:
            issues.append("Missing required field: name")
        if not config.provider:
            issues.append("Missing required field: provider")
        if not config.model:
            issues.append("Missing required field: model")
        if not config.type:
            issues.append("Missing required field: type")
        
        # Valid provider
        if config.provider not in ['anthropic', 'together']:
            issues.append(f"Unknown provider: {config.provider}")
        
        # Valid type
        if config.type not in ['switches', 'labels', 'preds', 'prediction']:
            issues.append(f"Unknown type: {config.type}. Must be 'switches', 'labels', 'preds', or 'prediction'")
        
        # Required parameters
        if not config.parameters.get('max_tokens'):
            issues.append("Missing required parameter: max_tokens")
        
        # Parameter validation
        temp = config.parameters.get('temperature')
        if temp is not None and (temp < 0 or temp > 2):
            issues.append("Temperature must be between 0 and 2")
        
        top_p = config.parameters.get('top_p')
        if top_p is not None and (top_p < 0 or top_p > 1):
            issues.append("top_p must be between 0 and 1")
        
        return issues
    
    def create_metadata_record(self, config: ModelConfig, batch_id: Optional[str] = None, 
                             additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a metadata record for tracking experiment runs."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config_name': config.name,
            'config': config.to_dict(),
            'batch_id': batch_id,
            'run_info': additional_info or {}
        }
        return metadata
    
    def save_run_metadata(self, config: ModelConfig, output_file: str, 
                         batch_id: Optional[str] = None, 
                         additional_info: Optional[Dict[str, Any]] = None):
        """Save metadata for a model run alongside the output file."""
        metadata = self.create_metadata_record(config, batch_id, additional_info)
        
        # Create metadata filename
        output_path = Path(output_file)
        metadata_file = output_path.parent / f"{output_path.stem}_metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_file}")
        except Exception as e:
            print(f"Failed to save metadata: {e}")
    
    def compare_configs(self, config_names: List[str]) -> Dict[str, Any]:
        """Compare multiple configurations side by side."""
        comparison = {
            'configs': {},
            'differences': {},
            'summary': {}
        }
        
        configs = [self.get_config(name) for name in config_names if self.get_config(name)]
        
        if not configs:
            return comparison
        
        # Collect all configs
        for config in configs:
            comparison['configs'][config.name] = config.to_dict()
        
        # Find differences in key parameters
        fields_to_compare = ['provider', 'model', 'type', 'parameters']
        
        for field in fields_to_compare:
            values = {}
            for config in configs:
                if field == 'parameters':
                    values[config.name] = config.parameters
                else:
                    values[config.name] = getattr(config, field)
            
            comparison['differences'][field] = values
        
        # Summary statistics
        comparison['summary'] = {
            'num_configs': len(configs),
            'providers': list(set(c.provider for c in configs)),
            'types': list(set(c.type for c in configs)),
            'models': list(set(c.model for c in configs))
        }
        
        return comparison
    
    def print_config_summary(self):
        """Print a summary of all loaded configurations."""
        print("\nConfiguration Summary")
        print("=" * 50)
        
        if not self.configs:
            print("No configurations loaded.")
            return
        
        # Group by type
        by_type = {}
        for config in self.configs.values():
            if config.type not in by_type:
                by_type[config.type] = []
            by_type[config.type].append(config)
        
        for config_type, configs in by_type.items():
            print(f"\n🔧 {config_type.upper()} Configurations:")
            for config in sorted(configs, key=lambda c: c.name):
                print(f"  • {config.name} ({config.provider}/{config.model})")
                print(f"    {config.description}")
        
        print(f"\nTotal configurations: {len(self.configs)}")
    
    def interactive_config_selector(self, config_type: Optional[str] = None) -> Optional[ModelConfig]:
        """Interactive configuration selector."""
        configs = self.list_configs(config_type)
        
        if not configs:
            print("No configurations available.")
            return None
        
        print(f"\n🤖 Available {'(' + config_type + ') ' if config_type else ''}Configurations:")
        for i, config in enumerate(configs):
            print(f"  {i+1}. {config.name}")
            print(f"     {config.provider}/{config.model} - {config.description}")
        
        while True:
            try:
                choice = input(f"\nSelect configuration (1-{len(configs)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    selected = configs[idx]
                    print(f"Selected: {selected.name}")
                    return selected
                else:
                    print("Invalid choice. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("Selection cancelled.")
                return None


def create_sample_config(name: str, provider: str, model: str, config_type: str) -> Dict[str, Any]:
    """Create a sample configuration dictionary."""
    
    default_params = {
        'anthropic': {
            'max_tokens': 1500 if config_type == 'switches' else 1000,
            'temperature': 0.1 if config_type == 'switches' else 0.3,
            'top_p': 0.9
        },
        'together': {
            'max_tokens': 1500 if config_type == 'switches' else 1000,
            'temperature': 0.2 if config_type == 'switches' else 0.4,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
    }
    
    return {
        'name': name,
        'description': f"Sample {provider} configuration for {config_type}",
        'provider': provider,
        'model': model,
        'type': config_type,
        'version': '1.0',
        'parameters': default_params.get(provider, {}),
        'prompt_template': None,
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'author': 'auto_generated',
        'notes': 'Auto-generated sample configuration',
        'output_format': {
            'type': 'json',
            'schema': 'standard'
        }
    }


if __name__ == "__main__":
    # Demo/test the configuration system
    manager = ConfigManager()
    manager.print_config_summary()
    
    # Test validation
    print("\n🧪 Testing configuration validation...")
    for config_name, config in manager.configs.items():
        issues = manager.validate_config(config)
        if issues:
            print(f"INVALID {config_name}: {', '.join(issues)}")
        else:
            print(f"VALID {config_name}: Valid")