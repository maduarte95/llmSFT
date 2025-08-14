#!/usr/bin/env python3
"""
Quick setup script for the LLM SFT project.
Run this after installing UV to get started quickly.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False


def check_uv_installed():
    """Check if UV is installed."""
    try:
        result = subprocess.run("uv --version", shell=True, check=True, capture_output=True)
        print(f"✅ UV is installed: {result.stdout.decode().strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ UV is not installed or not in PATH")
        print("Install UV from: https://docs.astral.sh/uv/getting-started/installation/")
        return False


def setup_environment():
    """Set up the .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔄 Setting up .env file...")
        try:
            env_file.write_text(env_example.read_text())
            print("✅ .env file created from template")
            print("📝 Please edit .env and add your API keys:")
            print("   - ANTHROPIC_API_KEY=your_key_here")
            print("   - TOGETHER_API_KEY=your_key_here")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️ No .env.example found, you'll need to create .env manually")
    
    return True


def main():
    """Main setup function."""
    print("🚀 LLM SFT Project Setup")
    print("=" * 50)
    
    # Check if UV is installed
    if not check_uv_installed():
        sys.exit(1)
    
    # Run UV sync to install dependencies
    if not run_command("uv sync", "Installing dependencies with UV"):
        sys.exit(1)
    
    # Set up environment file
    if not setup_environment():
        sys.exit(1)
    
    # Run a quick test to ensure everything works
    print("\n🧪 Running quick test...")
    if not run_command("uv run python -c 'import pandas, anthropic, requests; print(\"All imports successful!\")'", 
                       "Testing imports"):
        print("⚠️ Some imports failed, but the setup is mostly complete")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env and add your API keys")
    print("2. Test TogetherAI integration:")
    print("   uv run python test_together_api.py")
    print("3. Run switch classification:")
    print("   uv run python run_llm_classification_unified.py")
    print("4. Run group labeling:")
    print("   uv run python run_llm_labeling_unified.py")
    print("\nFor more info, see README_UNIFIED.md")
    print("=" * 50)


if __name__ == "__main__":
    main()