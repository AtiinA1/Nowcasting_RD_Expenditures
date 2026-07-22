# Weights & Biases (wandb) Hyperparameter Optimization Setup Guide

This guide explains how to set up and run hyperparameter sweeps using Weights & Biases for your MLP nowcasting models.

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements_wandb.txt
```

### 2. Create wandb Account & Setup
1. Go to [wandb.ai](https://wandb.ai) and create an account
2. Install wandb CLI: `pip install wandb`
3. Login: `wandb login`
4. Follow the prompts to authenticate

### 3. Update Entity Name
Edit the following files and replace `"your-entity-name"` with your actual wandb entity (username or team name):
- `MLP_AGT_temporalsplit_wandb.py`
- `run_wandb_sweep.py`

## 🚀 Running Hyperparameter Sweeps

### Option 1: Using the Automated Script (Recommended)

#### Create and Run a Sweep (50 experiments):
```bash
cd nn_mlp_nowcasting_model
python run_wandb_sweep.py --sweep_config wandb_sweep_config.yaml --count 50
```

#### Run More Experiments on Existing Sweep:
```bash
python run_wandb_sweep.py --sweep_id YOUR_SWEEP_ID --count 25
```

#### Just Create Sweep (Don't Run):
```bash
python run_wandb_sweep.py --sweep_config wandb_sweep_config.yaml --create_only
```

### Option 2: Manual wandb Commands

#### Create Sweep:
```bash
wandb sweep wandb_sweep_config.yaml
```

#### Run Sweep Agent:
```bash
wandb agent YOUR_SWEEP_ID
```

## ⚙️ Hyperparameters Being Tuned

The sweep configuration includes the following hyperparameters:

| Parameter | Type | Range/Values | Description |
|-----------|------|--------------|-------------|
| `learning_rate` | Log-uniform | 0.0001 - 0.1 | Learning rate for optimizer |
| `batch_size` | Categorical | [16, 32, 64, 128, 256] | Training batch size |
| `num_epochs` | Categorical | [100, 500, 1000, 2000, 5000] | Maximum training epochs |
| `hidden1_dim` | Categorical | [50, 100, 200, 300, 400, 500] | First hidden layer size |
| `hidden2_dim` | Categorical | [10, 20, 50, 100, 200] | Second hidden layer size |
| `hidden3_dim` | Categorical | [5, 10, 20, 50, 100] | Third hidden layer size |
| `embedding_dim` | Categorical | [2, 4, 8, 16, 32] | Country embedding dimension |
| `size_ensemble` | Categorical | [3, 5, 10, 15] | Number of models in ensemble |
| `patience` | Categorical | [500, 1000, 2000, 5000, 10000] | Early stopping patience |
| `optimizer` | Categorical | ["adam", "adamw", "sgd"] | Optimizer type |
| `lr_milestone` | Categorical | [10, 20, 50, 100] | LR scheduler milestone |
| `lr_gamma` | Uniform | 0.1 - 0.9 | LR decay factor |

## 📊 Metrics Tracked

The following metrics are automatically logged to wandb:

### Training Metrics (per epoch):
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `epoch`: Current epoch number

### Final Model Metrics:
- `final_ensemble_val_rmse`: Ensemble validation RMSE ⭐ (optimization target)
- `final_ensemble_val_mae`: Ensemble validation MAE
- `final_ensemble_val_mape`: Ensemble validation MAPE
- `final_avg_train_rmse`: Average training RMSE across models
- `final_avg_val_rmse`: Average validation RMSE across models

### Test Set Metrics:
- `test_rmse`: Test set RMSE
- `test_mae`: Test set MAE
- `test_mape`: Test set MAPE
- `test_r2`: Test set R-squared

## 🔧 Customizing the Sweep

### Modify Hyperparameters
Edit `wandb_sweep_config.yaml` to:
- Add/remove hyperparameters
- Change value ranges
- Modify search strategy (`bayes`, `random`, `grid`)
- Adjust early termination settings

### Example: Add New Hyperparameter
```yaml
parameters:
  dropout_rate:
    distribution: uniform
    min: 0.0
    max: 0.5
```

### Example: Change Search Method
```yaml
method: random  # instead of bayes
```

## 📈 Analyzing Results

### View Results in wandb Dashboard:
1. Go to your wandb project dashboard
2. Navigate to "Sweeps" tab
3. Click on your sweep to see:
   - Parallel coordinates plot
   - Hyperparameter importance
   - Best performing runs
   - Real-time training curves

### Export Best Configuration:
```python
import wandb
api = wandb.Api()
sweep = api.sweep("your-entity/nowcasting-rd-mlp/SWEEP_ID")
best_run = sweep.best_run()
print("Best hyperparameters:")
print(best_run.config)
```

## 🎯 Optimization Strategies

### For Fast Iteration:
- Use fewer epochs (100-500)
- Smaller ensemble sizes (3-5)
- Less aggressive patience (500-1000)

### For Best Performance:
- More epochs (2000-5000)
- Larger ensemble sizes (10-15)
- Higher patience (5000-10000)

### For Resource Constraints:
- Smaller batch sizes (16-32)
- Smaller hidden dimensions
- Reduce number of sweep runs

## 🚨 Important Notes

1. **Update Entity Name**: Replace `"your-entity-name"` with your wandb username/team
2. **Project Name**: Change `"nowcasting-rd-mlp"` if desired
3. **Resource Management**: Each run can take hours - plan accordingly
4. **Early Stopping**: The sweep uses Hyperband for early termination of poor runs
5. **Model Saving**: Best models are automatically saved to wandb

## 💡 Tips

- Start with a small sweep (10-20 runs) to test setup
- Monitor resource usage during sweeps
- Use wandb alerts for completion notifications
- Compare different model architectures by creating separate sweeps

## 🔗 Additional Resources

- [wandb Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Hyperparameter Optimization Guide](https://docs.wandb.ai/guides/sweeps/configuration)
- [wandb Python API](https://docs.wandb.ai/ref/python/)

## 🚀 Getting Started Checklist

- [ ] Install dependencies: `pip install -r requirements_wandb.txt`
- [ ] Create wandb account and login: `wandb login`
- [ ] Update entity name in scripts
- [ ] Test with small sweep: `python run_wandb_sweep.py --count 3`
- [ ] Launch full sweep: `python run_wandb_sweep.py --count 50`
- [ ] Monitor results in wandb dashboard 