#!/usr/bin/env python3
"""
Script to systematically test and compare different neural network architectures
for the nowcasting R&D expenditures problem.

This script will:
1. Run separate sweeps for 1, 2, and 3 layer architectures
2. Track and compare results
3. Generate summary reports
"""

import os
import subprocess
import sys
import json
import pandas as pd
import wandb
from datetime import datetime

class ArchitectureComparison:
    def __init__(self):
        self.sweep_configs = [
            ("1-layer", "wandb_config_1layer.yaml"),
            ("2-layer", "wandb_config_2layer.yaml"), 
            ("3-layer", "wandb_config_3layer.yaml"),
            ("all-architectures", "wandb_config_all_architectures.yaml")
        ]
        self.sweep_ids = {}
        self.results_summary = []
        
    def check_wandb_login(self):
        """Check if wandb is properly configured"""
        try:
            wandb.login()
            print("✅ Wandb login successful")
            return True
        except Exception as e:
            print(f"❌ Wandb login failed: {e}")
            return False
    
    def create_sweeps(self):
        """Create wandb sweeps for each architecture"""
        if not self.check_wandb_login():
            return False
            
        print("\n🚀 Creating wandb sweeps for each architecture...")
        
        for arch_name, config_file in self.sweep_configs:
            if not os.path.exists(config_file):
                print(f"❌ Config file {config_file} not found!")
                continue
                
            try:
                # Create sweep
                result = subprocess.run(
                    ["wandb", "sweep", config_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Extract sweep ID
                sweep_id = result.stdout.strip().split()[-1]
                self.sweep_ids[arch_name] = sweep_id
                
                print(f"✅ Created {arch_name} sweep: {sweep_id}")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to create sweep for {arch_name}: {e}")
                
        return len(self.sweep_ids) > 0
    
    def run_sweeps(self, runs_per_sweep=20):
        """Run sweeps with specified number of runs each"""
        if not self.sweep_ids:
            print("❌ No sweeps created!")
            return False
            
        print(f"\n🏃‍♂️ Running {runs_per_sweep} runs for each sweep...")
        
        for arch_name, sweep_id in self.sweep_ids.items():
            print(f"\n📊 Starting {arch_name} sweep ({sweep_id})...")
            print(f"Command to run: wandb agent {sweep_id}")
            print(f"This will run {runs_per_sweep} experiments for {arch_name} architecture")
            
            # Ask user if they want to run automatically or manually
            response = input(f"Run {arch_name} sweep automatically? (y/n/skip): ").lower()
            
            if response == 'y':
                try:
                    # Run sweep with count limit
                    subprocess.run(
                        ["wandb", "agent", "--count", str(runs_per_sweep), sweep_id],
                        check=True
                    )
                    print(f"✅ Completed {arch_name} sweep")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to run {arch_name} sweep: {e}")
            elif response == 'skip':
                print(f"⏭️ Skipped {arch_name} sweep")
            else:
                print(f"📝 Manual command: wandb agent --count {runs_per_sweep} {sweep_id}")
    
    def analyze_results(self):
        """Analyze and compare results from all sweeps"""
        print("\n📈 Analyzing results from all sweeps...")
        
        api = wandb.Api()
        
        for arch_name, sweep_id in self.sweep_ids.items():
            try:
                sweep = api.sweep(f"epfl_stip/nowcasting-rd-mlp/{sweep_id}")
                runs = list(sweep.runs)
                
                if not runs:
                    print(f"⚠️ No completed runs found for {arch_name}")
                    continue
                
                # Get best run
                best_run = min(runs, key=lambda r: r.summary.get('final_ensemble_val_rmse', float('inf')))
                
                # Extract architecture info
                config = best_run.config
                architecture = []
                if config.get('hidden1_dim', 0) > 0:
                    architecture.append(config['hidden1_dim'])
                if config.get('hidden2_dim', 0) > 0:
                    architecture.append(config['hidden2_dim'])
                if config.get('hidden3_dim', 0) > 0:
                    architecture.append(config['hidden3_dim'])
                
                result = {
                    'architecture_type': arch_name,
                    'num_layers': len(architecture),
                    'layer_sizes': architecture,
                    'best_val_rmse': best_run.summary.get('final_ensemble_val_rmse', 'N/A'),
                    'best_val_r2': best_run.summary.get('final_ensemble_val_r2', 'N/A'),
                    'learning_rate': config.get('learning_rate', 'N/A'),
                    'batch_size': config.get('batch_size', 'N/A'),
                    'dropout_rate': config.get('dropout_rate', 'N/A'),
                    'weight_decay': config.get('weight_decay', 'N/A'),
                    'ensemble_size': config.get('size_ensemble', 'N/A'),
                    'num_epochs': config.get('num_epochs', 'N/A'),
                    'run_id': best_run.id,
                    'total_runs': len(runs)
                }
                
                self.results_summary.append(result)
                print(f"✅ {arch_name}: Best RMSE = {result['best_val_rmse']:.4f}, Architecture = {architecture}")
                
            except Exception as e:
                print(f"❌ Failed to analyze {arch_name}: {e}")
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        if not self.results_summary:
            print("❌ No results to report!")
            return
            
        print("\n📋 ARCHITECTURE COMPARISON REPORT")
        print("=" * 60)
        
        # Sort by performance
        sorted_results = sorted(self.results_summary, key=lambda x: x['best_val_rmse'] if x['best_val_rmse'] != 'N/A' else float('inf'))
        
        print(f"{'Rank':<4} {'Architecture':<15} {'Layers':<8} {'RMSE':<10} {'R²':<8} {'Structure'}")
        print("-" * 60)
        
        for i, result in enumerate(sorted_results, 1):
            rmse = f"{result['best_val_rmse']:.3f}" if result['best_val_rmse'] != 'N/A' else 'N/A'
            r2 = f"{result['best_val_r2']:.3f}" if result['best_val_r2'] != 'N/A' else 'N/A'
            structure = "-".join(map(str, result['layer_sizes'])) if result['layer_sizes'] else 'N/A'
            
            print(f"{i:<4} {result['architecture_type']:<15} {result['num_layers']:<8} {rmse:<10} {r2:<8} {structure}")
        
        # Save detailed results
        df = pd.DataFrame(self.results_summary)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"architecture_comparison_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        # Print recommendations
        if sorted_results:
            best = sorted_results[0]
            print(f"\n🏆 BEST ARCHITECTURE:")
            print(f"   Type: {best['architecture_type']}")
            print(f"   Structure: {'-'.join(map(str, best['layer_sizes'])) if best['layer_sizes'] else 'N/A'}")
            print(f"   Validation RMSE: {best['best_val_rmse']}")
            print(f"   Validation R²: {best['best_val_r2']}")
            print(f"   Learning Rate: {best['learning_rate']}")
            print(f"   Batch Size: {best['batch_size']}")
            print(f"   Dropout Rate: {best['dropout_rate']}")
    
    def run_full_comparison(self, runs_per_sweep=15):
        """Run the complete architecture comparison pipeline"""
        print("🔬 NEURAL NETWORK ARCHITECTURE COMPARISON")
        print("=" * 50)
        print("This will systematically test 1, 2, and 3 layer architectures")
        print(f"with {runs_per_sweep} experiments each to find the optimal network design.")
        print()
        
        # Step 1: Create sweeps
        if not self.create_sweeps():
            print("❌ Failed to create sweeps!")
            return False
        
        # Step 2: Run sweeps
        self.run_sweeps(runs_per_sweep)
        
        # Step 3: Analyze results
        input("\nPress Enter when all sweeps are complete to analyze results...")
        self.analyze_results()
        
        # Step 4: Generate report
        self.generate_report()
        
        return True

def main():
    """Main function"""
    comparison = ArchitectureComparison()
    
    print("Choose an option:")
    print("1. Run full architecture comparison")
    print("2. Create sweeps only")
    print("3. Analyze existing results")
    print("4. Show sweep commands")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        runs_per_sweep = int(input("Runs per sweep (default 15): ") or "15")
        comparison.run_full_comparison(runs_per_sweep)
    elif choice == "2":
        comparison.create_sweeps()
        print("\n📝 Run these commands manually:")
        for arch_name, sweep_id in comparison.sweep_ids.items():
            print(f"   wandb agent {sweep_id}  # {arch_name}")
    elif choice == "3":
        comparison.analyze_results()
        comparison.generate_report()
    elif choice == "4":
        comparison.create_sweeps()
        print("\n📝 Sweep commands:")
        for arch_name, sweep_id in comparison.sweep_ids.items():
            print(f"   {arch_name}: wandb agent {sweep_id}")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main() 