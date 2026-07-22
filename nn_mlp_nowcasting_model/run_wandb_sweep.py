#!/usr/bin/env python3
"""
Script to set up and run wandb sweeps for the nowcasting model.
"""

import os
import subprocess
import sys
import wandb
import yaml

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'torch', 'wandb', 'sklearn', 'xgboost', 
        'statsmodels', 'seaborn', 'matplotlib', 'tqdm', 'shap'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("All required packages are installed.")
    return True

def setup_wandb():
    """Set up wandb login and configuration."""
    print("Setting up Weights & Biases...")
    
    # Check if already logged in
    try:
        wandb.login()
        print("Already logged in to wandb.")
    except:
        print("Please log in to wandb:")
        wandb.login()
    
    # Set up environment variables
    if 'WANDB_ENTITY' not in os.environ:
        entity = input("Enter your wandb entity/username (press enter to skip): ").strip()
        if entity:
            os.environ['WANDB_ENTITY'] = entity
            print(f"Set WANDB_ENTITY to: {entity}")
        else:
            print("No entity set. Using default.")

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'data/country_code.csv',
        'data/GERD/DP_LIVE_08052023154811337.csv',
        'data/OECD/PATS_IPC_11062023234902217.csv',
        'data/IMF/WEOApr2023all.csv',
        'data/GT/trends_data_by_topic_resampled_filtered.csv',
        'data/GT/trends_data_by_topic_filtered.csv'
    ]
    
    missing_files = []
    base_path = '/Users/atin/Nowcasting/Nowcasting_github/'
    
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing data files: {missing_files}")
        print("Please ensure all required data files are in place.")
        return False
    
    print("All required data files found.")
    return True

def create_sweep():
    """Create a wandb sweep."""
    print("Creating wandb sweep...")
    
    # Load sweep configuration
    with open('wandb_config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="nowcasting-rd-mlp")
    print(f"Created sweep with ID: {sweep_id}")
    
    return sweep_id

def run_sweep(sweep_id, count=10):
    """Run the wandb sweep."""
    print(f"Running sweep {sweep_id} with {count} runs...")
    
    # Change to the correct directory
    os.chdir('/Users/atin/Nowcasting/Nowcasting_github/nn_mlp_nowcasting_model/')
    
    # Run the sweep
    try:
        wandb.agent(sweep_id, count=count)
    except KeyboardInterrupt:
        print("Sweep interrupted by user.")
    except Exception as e:
        print(f"Error during sweep: {e}")

def run_single_experiment():
    """Run a single experiment without sweep."""
    print("Running single experiment...")
    
    # Change to the correct directory
    os.chdir('/Users/atin/Nowcasting/Nowcasting_github/nn_mlp_nowcasting_model/')
    
    # Run the script
    try:
        subprocess.run([sys.executable, 'MLP_AGT_temporalsplit_wandb.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
    except KeyboardInterrupt:
        print("Experiment interrupted by user.")

def main():
    """Main function."""
    print("=== Nowcasting Model - Wandb Setup ===\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup wandb
    setup_wandb()
    
    # Check data files
    if not check_data_files():
        response = input("Some data files are missing. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run a single experiment")
    print("2. Create and run a hyperparameter sweep")
    print("3. Run an existing sweep")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        run_single_experiment()
    elif choice == "2":
        sweep_id = create_sweep()
        count = input("How many runs? (default: 10): ").strip()
        count = int(count) if count.isdigit() else 10
        run_sweep(sweep_id, count)
    elif choice == "3":
        sweep_id = input("Enter sweep ID: ").strip()
        count = input("How many runs? (default: 10): ").strip()
        count = int(count) if count.isdigit() else 10
        run_sweep(sweep_id, count)
    else:
        print("Invalid choice.")
        sys.exit(1)

if __name__ == "__main__":
    main() 