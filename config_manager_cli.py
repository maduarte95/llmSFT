#!/usr/bin/env python3
"""
Command-line utility for managing model configurations.
Provides tools for listing, comparing, validating, and creating configurations.
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from config_system import ConfigManager, ModelConfig, create_sample_config


def print_separator(char="-", length=60):
    """Print a separator line."""
    print(char * length)


def list_configs_command(manager: ConfigManager, config_type: Optional[str] = None):
    """List all configurations with details."""
    configs = manager.list_configs(config_type)
    
    if not configs:
        print("No configurations found.")
        return
    
    print(f"\n📋 {'All' if not config_type else config_type.title()} Configurations")
    print_separator()
    
    for config in configs:
        print(f"🔧 {config.name}")
        print(f"   Provider: {config.provider}")
        print(f"   Model: {config.model}")
        print(f"   Type: {config.type}")
        print(f"   Description: {config.description}")
        print(f"   Parameters: {config.parameters}")
        print(f"   Version: {config.version}")
        if config.notes:
            print(f"   Notes: {config.notes}")
        print()


def validate_configs_command(manager: ConfigManager):
    """Validate all configurations and report issues."""
    print("\n🧪 Configuration Validation Report")
    print_separator()
    
    valid_count = 0
    invalid_count = 0
    
    for name, config in manager.configs.items():
        issues = manager.validate_config(config)
        if issues:
            print(f"❌ {name}:")
            for issue in issues:
                print(f"   - {issue}")
            invalid_count += 1
        else:
            print(f"✅ {name}: Valid")
            valid_count += 1
        print()
    
    print_separator()
    print(f"Summary: {valid_count} valid, {invalid_count} invalid")


def compare_configs_command(manager: ConfigManager, config_names: List[str]):
    """Compare multiple configurations side by side."""
    comparison = manager.compare_configs(config_names)
    
    if not comparison['configs']:
        print("No valid configurations found for comparison.")
        return
    
    print(f"\n⚖️ Configuration Comparison")
    print_separator()
    
    # Summary
    print(f"Comparing {len(config_names)} configurations:")
    for name in config_names:
        if name in comparison['configs']:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} (not found)")
    
    print()
    
    # Key differences
    print("🔍 Key Differences:")
    
    for field, values in comparison['differences'].items():
        print(f"\n{field.title()}:")
        for config_name, value in values.items():
            if field == 'parameters':
                param_str = ', '.join([f"{k}={v}" for k, v in value.items()])
                print(f"  {config_name}: {param_str}")
            else:
                print(f"  {config_name}: {value}")
    
    # Summary stats
    print(f"\n📊 Summary:")
    summary = comparison['summary']
    print(f"  Providers: {', '.join(summary['providers'])}")
    print(f"  Types: {', '.join(summary['types'])}")
    print(f"  Models: {summary['num_configs']} different models")


def create_config_command(manager: ConfigManager):
    """Interactive configuration creation."""
    print("\n🛠️ Create New Configuration")
    print_separator()
    
    try:
        # Collect basic info
        name = input("Configuration name: ").strip()
        if not name:
            print("❌ Name is required.")
            return
        
        if name in manager.configs:
            print(f"❌ Configuration '{name}' already exists.")
            return
        
        description = input("Description: ").strip() or f"Custom configuration: {name}"
        
        # Provider selection
        print("\nSelect provider:")
        print("1. anthropic")
        print("2. together")
        
        provider_choice = input("Choose provider (1-2): ").strip()
        if provider_choice == "1":
            provider = "anthropic"
            default_model = "claude-sonnet-4-20250514"
        elif provider_choice == "2":
            provider = "together" 
            default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        else:
            print("❌ Invalid provider choice.")
            return
        
        model = input(f"Model (default: {default_model}): ").strip() or default_model
        
        # Type selection
        print("\nSelect type:")
        print("1. switches")
        print("2. labels")
        print("3. preds")
        
        type_choice = input("Choose type (1-3): ").strip()
        if type_choice == "1":
            config_type = "switches"
        elif type_choice == "2":
            config_type = "labels"
        elif type_choice == "3":
            config_type = "preds"
        else:
            print("❌ Invalid type choice.")
            return
        
        # Create config dictionary
        config_dict = create_sample_config(name, provider, model, config_type)
        config_dict['description'] = description
        config_dict['author'] = input("Author (optional): ").strip() or "unknown"
        config_dict['notes'] = input("Notes (optional): ").strip() or ""
        
        # Parameter customization
        print(f"\nDefault parameters: {config_dict['parameters']}")
        customize = input("Customize parameters? (y/N): ").strip().lower()
        
        if customize == 'y':
            # Allow basic parameter editing
            temp = input(f"Temperature (current: {config_dict['parameters'].get('temperature', 0.1)}): ").strip()
            if temp:
                try:
                    config_dict['parameters']['temperature'] = float(temp)
                except ValueError:
                    print("⚠️ Invalid temperature, keeping current value")
            
            max_tokens = input(f"Max tokens (current: {config_dict['parameters'].get('max_tokens', 1500)}): ").strip()
            if max_tokens:
                try:
                    config_dict['parameters']['max_tokens'] = int(max_tokens)
                except ValueError:
                    print("⚠️ Invalid max_tokens, keeping current value")
        
        # Save configuration
        config_file = Path("configs") / f"{name.replace(' ', '_').replace('-', '_')}.yaml"
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ Configuration saved to: {config_file}")
            print("🔄 Reload configurations to use the new config.")
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    except KeyboardInterrupt:
        print("\n❌ Configuration creation cancelled.")


def analyze_results_command():
    """Analyze results from different configurations."""
    print("\n📈 Results Analysis")
    print_separator()
    
    # Find result files
    current_dir = Path(".")
    result_files = list(current_dir.glob("*_with_switches_*.csv"))
    metadata_files = list(current_dir.glob("*_metadata.json"))
    
    print(f"Found {len(result_files)} result files and {len(metadata_files)} metadata files")
    
    if not result_files:
        print("No result files found. Run some experiments first!")
        return
    
    # Group by config name
    results_by_config = {}
    
    for result_file in result_files:
        # Try to extract config name from filename
        parts = result_file.stem.split('_')
        if len(parts) >= 3:
            config_name = '_'.join(parts[3:-1])  # Remove 'data_with_switches' and timestamp
            if config_name not in results_by_config:
                results_by_config[config_name] = []
            results_by_config[config_name].append(result_file)
    
    print(f"\n🔍 Results by configuration:")
    for config_name, files in results_by_config.items():
        print(f"  {config_name}: {len(files)} runs")
    
    # Show metadata for recent runs
    print(f"\n📊 Recent metadata:")
    metadata_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    for metadata_file in metadata_files[:3]:  # Show last 3
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            config_name = metadata.get('config_name', 'unknown')
            timestamp = metadata.get('timestamp', 'unknown')
            batch_id = metadata.get('batch_id', 'unknown')
            
            print(f"  {config_name} - {timestamp[:19]} (batch: {batch_id[:12]}...)")
            
        except Exception as e:
            print(f"  ❌ Error reading {metadata_file}: {e}")


def main():
    """Main CLI function."""
    if len(sys.argv) < 2:
        print("🔧 Configuration Manager CLI")
        print("Usage: python config_manager_cli.py <command> [args]")
        print("\nCommands:")
        print("  list [type]         - List all configurations (optionally filter by type)")
        print("  validate            - Validate all configurations")
        print("  compare <names...>  - Compare multiple configurations")
        print("  create              - Create a new configuration interactively")
        print("  analyze             - Analyze experimental results")
        print("\nExamples:")
        print("  python config_manager_cli.py list")
        print("  python config_manager_cli.py list switches")
        print("  python config_manager_cli.py validate")
        print("  python config_manager_cli.py compare claude-sonnet-4-switches llama-3.1-70b-switches")
        print("  python config_manager_cli.py create")
        print("  python config_manager_cli.py analyze")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Load configurations
    manager = ConfigManager()
    
    if command == "list":
        config_type = sys.argv[2] if len(sys.argv) > 2 else None
        list_configs_command(manager, config_type)
    
    elif command == "validate":
        validate_configs_command(manager)
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("❌ Compare command requires at least 2 configuration names")
            sys.exit(1)
        config_names = sys.argv[2:]
        compare_configs_command(manager, config_names)
    
    elif command == "create":
        create_config_command(manager)
    
    elif command == "analyze":
        analyze_results_command()
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)