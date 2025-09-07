#!/usr/bin/env python3
"""
Setup script for Agricultural ML API
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.logging import setup_logging


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"Python version: {sys.version}")
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "trained_models", 
        "logs",
        "app",
        "app/api",
        "app/core",
        "app/models", 
        "app/schemas",
        "app/services",
        "scripts",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def create_init_files():
    """Create __init__.py files for packages"""
    init_files = [
        "app/__init__.py",
        "app/api/__init__.py",
        "app/core/__init__.py", 
        "app/models/__init__.py",
        "app/schemas/__init__.py",
        "app/services/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Package initialization file"""\n')
            print(f"Created: {init_file}")


def create_env_file():
    """Create .env file with default settings"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        env_content = """# Agricultural ML API Configuration

# API Settings
APP_NAME=Agricultural ML API
APP_VERSION=1.0.0
DEBUG=false

# Server Settings  
HOST=0.0.0.0
PORT=8000

# Model Settings
RANDOM_STATE=42
TEST_SIZE=0.2
N_ESTIMATORS=100
MAX_DEPTH=10

# Logging
LOG_LEVEL=INFO
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"Created: {env_file}")


def create_gitkeep_files():
    """Create .gitkeep files for empty directories"""
    gitkeep_dirs = ["data", "trained_models", "logs"]
    
    for directory in gitkeep_dirs:
        gitkeep_file = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_file):
            with open(gitkeep_file, 'w') as f:
                f.write("# Keep this directory in git\n")
            print(f"Created: {gitkeep_file}")


def install_dependencies():
    """Install Python dependencies"""
    try:
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def check_data_file():
    """Check if training data exists"""
    data_file = "data/agricultural_data.csv"
    
    if os.path.exists(data_file):
        print(f"✓ Training data found: {data_file}")
        return True
    else:
        print(f"⚠ Training data not found: {data_file}")
        print("Please place your agricultural_data.csv in the data/ directory")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("Agricultural ML API - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Create __init__.py files
    print("\nCreating package files...")
    create_init_files()
    
    # Create .env file
    print("\nCreating configuration files...")
    create_env_file()
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not install_dependencies():
        print("Setup incomplete - dependency installation failed")
        return False
    
    # Check for training data
    print("\nChecking for training data...")
    has_data = check_data_file()
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("=" * 60)
    print("✓ Directories created")
    print("✓ Package files created") 
    print("✓ Configuration files created")
    print("✓ Dependencies installed")
    
    if has_data:
        print("✓ Training data available")
        print("\nNext steps:")
        print("1. Run: python scripts/train_models.py")
        print("2. Run: uvicorn app.main:app --reload")
    else:
        print("⚠ Training data missing")
        print("\nNext steps:")
        print("1. Place agricultural_data.csv in the data/ directory")
        print("2. Run: python scripts/train_models.py") 
        print("3. Run: uvicorn app.main:app --reload")
    
    print("\nSetup completed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)