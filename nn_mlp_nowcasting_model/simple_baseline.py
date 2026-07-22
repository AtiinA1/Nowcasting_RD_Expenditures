import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import the data loading functions from your main script
from MLP_AGT_temporalsplit_wandb import (
    load_and_preprocess_data, 
    preprocess_merged_data, 
    create_train_test_splits
)

def test_baseline_models():
    """Test simple baseline models to see if the data has predictive power"""
    
    print("Loading and preprocessing data...")
    merged_df, df_trends, rd_expenditure_df_rev, oecd_df_rev_4lag = load_and_preprocess_data()
    merged_df, max_lag = preprocess_merged_data(merged_df, df_trends, rd_expenditure_df_rev, oecd_df_rev_4lag)
    
    # Create train/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_splits(
        merged_df, df_trends, rd_expenditure_df_rev, max_lag
    )
    
    # Drop non-numeric columns
    X_train_numeric = X_train.select_dtypes(include=[np.number])
    X_val_numeric = X_val.select_dtypes(include=[np.number])  
    X_test_numeric = X_test.select_dtypes(include=[np.number])
    
    # Fill any remaining NaN values
    X_train_numeric = X_train_numeric.fillna(0)
    X_val_numeric = X_val_numeric.fillna(0)
    X_test_numeric = X_test_numeric.fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)
    
    # Convert y to 1D arrays
    y_train_vals = y_train.values.ravel()
    y_val_vals = y_val.values.ravel()
    y_test_vals = y_test.values.ravel()
    
    print(f"Dataset info:")
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Validation samples: {len(X_val_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    print(f"Target range: {y_train_vals.min():.2f} to {y_train_vals.max():.2f}")
    print(f"Target mean: {y_train_vals.mean():.2f}, std: {y_train_vals.std():.2f}")
    
    # Test different models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0), 
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train_vals)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_vals, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_vals, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_vals, test_pred))
        
        train_r2 = r2_score(y_train_vals, train_pred)
        val_r2 = r2_score(y_val_vals, val_pred)
        test_r2 = r2_score(y_test_vals, test_pred)
        
        results[name] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse, 
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2
        }
        
        print(f"  Train RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")
        print(f"  Val RMSE: {val_rmse:.2f}, R²: {val_r2:.3f}")
        print(f"  Test RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}")
    
    # Compare with naive baseline (always predict the mean)
    baseline_pred = np.full_like(y_test_vals, y_train_vals.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test_vals, baseline_pred))
    baseline_r2 = r2_score(y_test_vals, baseline_pred)
    
    print(f"\nBaseline (predict mean):")
    print(f"  Test RMSE: {baseline_rmse:.2f}, R²: {baseline_r2:.3f}")
    
    # Find best model
    best_model = min(results.keys(), key=lambda x: results[x]['test_rmse'])
    print(f"\nBest model: {best_model}")
    print(f"Best test RMSE: {results[best_model]['test_rmse']:.2f}")
    
    return results

if __name__ == "__main__":
    test_baseline_models() 