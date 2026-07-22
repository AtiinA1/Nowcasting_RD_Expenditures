"""
Targeted sweep for AGT, MGT, and AllVar temporal-split variants.

These three variants have not found configs that clearly beat the degenerate
default (MAPE≈72.81). This script runs a focused grid of hyperparameters
drawn from configs that worked well for Macros, LagRD, and AGTwRD.
"""

import subprocess
import sys
import os
import json
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VARIANT_SCRIPTS = {
    'AGT': os.path.join(SCRIPT_DIR, 'MLP_AGT_temporalsplit_wandb.py'),
    'MGT': os.path.join(SCRIPT_DIR, 'MLP_MGT_temporalsplit_wandb.py'),
    'AllVar': os.path.join(SCRIPT_DIR, 'MLP_AllVar_temporalsplit_wandb.py'),
}

# Configs inspired by what worked for Macros (MAPE=52.47), LagRD (MAPE=52.51),
# AGTwRD (MAPE=53.12), and MGTwRD (MAPE=55.24).
TARGETED_CONFIGS = [
    # Config that gave Macros 52.47 (already tried but let's verify with longer training)
    {
        'learning_rate': 0.018, 'batch_size': 64,
        'hidden1_dim': 256, 'hidden2_dim': 32, 'hidden3_dim': 32,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 1200, 'lr_gamma': 0.1,
        'optimizer': 'adamw', 'weight_decay': 0.001, 'dropout_rate': 0.0
    },
    # Config B (Macros=74.19, LagRD=74.24, AGTwRD=75.93) with more epochs
    {
        'learning_rate': 0.015, 'batch_size': 32,
        'hidden1_dim': 256, 'hidden2_dim': 20, 'hidden3_dim': 20,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 1200, 'lr_gamma': 0.5,
        'optimizer': 'adamw', 'weight_decay': 0.0001, 'dropout_rate': 0.1
    },
    # Lower LR variant (many successful configs use lr around 0.005-0.01)
    {
        'learning_rate': 0.008, 'batch_size': 64,
        'hidden1_dim': 256, 'hidden2_dim': 32, 'hidden3_dim': 32,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 2000, 'lr_gamma': 0.3,
        'optimizer': 'adamw', 'weight_decay': 0.001, 'dropout_rate': 0.0
    },
    # AGTwRD-inspired config (gave 53.12 for AGTwRD)
    {
        'learning_rate': 0.019, 'batch_size': 64,
        'hidden1_dim': 256, 'hidden2_dim': 16, 'hidden3_dim': 20,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 12000,
        'lr_milestone': 1200, 'lr_gamma': 0.1,
        'optimizer': 'adamw', 'weight_decay': 0.001, 'dropout_rate': 0.3
    },
    # Larger ensemble, moderate LR, no dropout
    {
        'learning_rate': 0.01, 'batch_size': 64,
        'hidden1_dim': 256, 'hidden2_dim': 32, 'hidden3_dim': 32,
        'embedding_dim': 4, 'size_ensemble': 10,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 2000, 'lr_gamma': 0.1,
        'optimizer': 'adamw', 'weight_decay': 0.001, 'dropout_rate': 0.0
    },
    # Wider first layer, lower LR, larger embedding
    {
        'learning_rate': 0.005, 'batch_size': 64,
        'hidden1_dim': 512, 'hidden2_dim': 64, 'hidden3_dim': 32,
        'embedding_dim': 8, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 3000, 'lr_gamma': 0.1,
        'optimizer': 'adamw', 'weight_decay': 0.0001, 'dropout_rate': 0.1
    },
    # Simpler 2-layer network (h2=0 means skip), similar to AGTwRD best
    {
        'learning_rate': 0.015, 'batch_size': 64,
        'hidden1_dim': 256, 'hidden2_dim': 0, 'hidden3_dim': 32,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 12000,
        'lr_milestone': 1500, 'lr_gamma': 0.1,
        'optimizer': 'adamw', 'weight_decay': 0.001, 'dropout_rate': 0.2
    },
    # Very low LR with long training (to avoid overshooting)
    {
        'learning_rate': 0.003, 'batch_size': 32,
        'hidden1_dim': 200, 'hidden2_dim': 32, 'hidden3_dim': 32,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 5000, 'lr_gamma': 0.3,
        'optimizer': 'adamw', 'weight_decay': 0.0001, 'dropout_rate': 0.1
    },
    # Higher weight decay for regularization
    {
        'learning_rate': 0.012, 'batch_size': 64,
        'hidden1_dim': 256, 'hidden2_dim': 32, 'hidden3_dim': 32,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 1500, 'lr_gamma': 0.1,
        'optimizer': 'adamw', 'weight_decay': 0.01, 'dropout_rate': 0.0
    },
    # LagRD-inspired (gave 52.51 for LagRD)
    {
        'learning_rate': 0.015, 'batch_size': 32,
        'hidden1_dim': 256, 'hidden2_dim': 16, 'hidden3_dim': 32,
        'embedding_dim': 4, 'size_ensemble': 5,
        'num_epochs': 30000, 'patience': 15000,
        'lr_milestone': 1200, 'lr_gamma': 0.5,
        'optimizer': 'adamw', 'weight_decay': 0.0001, 'dropout_rate': 0.1
    },
]


def run_single_config(variant, config, run_idx, total):
    """Run a single training with a given config for a given variant."""
    script = VARIANT_SCRIPTS[variant]
    config_json = json.dumps(config)
    
    env = os.environ.copy()
    env['FIXED_CONFIG_JSON'] = config_json
    
    print(f"\n{'='*70}")
    print(f"[{run_idx}/{total}] {variant} | "
          f"lr={config['learning_rate']}, batch={config['batch_size']}, "
          f"h=[{config['hidden1_dim']},{config['hidden2_dim']},{config['hidden3_dim']}], "
          f"ens={config['size_ensemble']}, dropout={config['dropout_rate']}, "
          f"wd={config['weight_decay']}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, script],
        env=env,
        capture_output=False,
        timeout=7200  # 2 hour timeout per run
    )
    
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run targeted sweeps for AGT, MGT, AllVar')
    parser.add_argument('--variants', nargs='+', default=['AGT', 'MGT', 'AllVar'],
                        choices=['AGT', 'MGT', 'AllVar'],
                        help='Which variants to run')
    parser.add_argument('--configs', type=str, default='all',
                        help='Comma-separated config indices (0-based) or "all"')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without running')
    args = parser.parse_args()
    
    if args.configs == 'all':
        config_indices = list(range(len(TARGETED_CONFIGS)))
    else:
        config_indices = [int(x) for x in args.configs.split(',')]
    
    configs_to_run = [TARGETED_CONFIGS[i] for i in config_indices]
    
    total_runs = len(args.variants) * len(configs_to_run)
    print(f"Targeted sweep: {len(args.variants)} variants x {len(configs_to_run)} configs = {total_runs} runs")
    
    if args.dry_run:
        for variant in args.variants:
            for i, cfg in enumerate(configs_to_run):
                print(f"  {variant} config {config_indices[i]}: "
                      f"lr={cfg['learning_rate']}, batch={cfg['batch_size']}, "
                      f"h=[{cfg['hidden1_dim']},{cfg['hidden2_dim']},{cfg['hidden3_dim']}], "
                      f"ens={cfg['size_ensemble']}")
        return
    
    run_idx = 0
    results = []
    for variant in args.variants:
        for cfg in configs_to_run:
            run_idx += 1
            try:
                rc = run_single_config(variant, cfg, run_idx, total_runs)
                results.append((variant, cfg, rc))
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT for {variant}")
                results.append((variant, cfg, -1))
            except Exception as e:
                print(f"  ERROR for {variant}: {e}")
                results.append((variant, cfg, -2))
    
    print(f"\n{'='*70}")
    print("SWEEP COMPLETE")
    print(f"{'='*70}")
    succeeded = sum(1 for _, _, rc in results if rc == 0)
    print(f"Succeeded: {succeeded}/{total_runs}")
    for variant, cfg, rc in results:
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {variant} lr={cfg['learning_rate']} h=[{cfg['hidden1_dim']},{cfg['hidden2_dim']},{cfg['hidden3_dim']}]: {status}")


if __name__ == '__main__':
    main()
