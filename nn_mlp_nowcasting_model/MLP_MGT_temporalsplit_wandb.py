import os
import wandb
import pandas as pd

# Reuse the full training pipeline from AGT wandb version
import MLP_AGT_temporalsplit_wandb as base

_orig_init_wandb = base.init_wandb

def init_wandb(config=None):
    """
    Variant wrapper for the MGT temporalsplit run.
    Delegates to base init to ensure defaults, then appends an 'MGT' tag.
    """
    cfg_override = {} if config is None else dict(config)
    # Optional: allow fixed config via env
    fixed_path = os.getenv("FIXED_CONFIG_PATH")
    fixed_json = os.getenv("FIXED_CONFIG_JSON")
    try:
        if fixed_path and os.path.exists(fixed_path):
            import json
            with open(fixed_path, "r") as f:
                cfg_override.update(json.load(f))
        elif fixed_json:
            import json
            cfg_override.update(json.loads(fixed_json))
    except Exception as e:
        print(f"[WARN] Failed to load fixed config override: {e}")
    cfg_override['dataset_variant'] = 'MGT'
    cfg = _orig_init_wandb(cfg_override)
    run = wandb.run
    current_tags = list(run.tags) if getattr(run, "tags", None) else []
    run.tags = current_tags + ["MGT"]
    return cfg

# Monkey-patch the base init with our tagged version
base.init_wandb = init_wandb

# Override create_train_test_splits to select MGT columns (GT YTD + GT yearly_avg_lag)
def create_train_test_splits(merged_df, df_trends, rd_expenditure_df_rev, max_lag):
    X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or (col == 'Month')  or ('_lag' in col) or ('_mean_YTD' in col)]].copy()
    X_all.fillna(0, inplace=True)
    y = merged_df[['Country', 'Year', 'Month', 'rd_expenditure']].copy()

    countries = X_all['Country'].unique()
    X_train_combined = pd.DataFrame()
    X_val_combined = pd.DataFrame()
    X_test_combined = pd.DataFrame()
    y_train_combined = pd.Series(dtype=float)
    y_val_combined = pd.Series(dtype=float)
    y_test_combined = pd.Series(dtype=float)

    train_ratio, val_ratio = 0.64, 0.16

    for country in countries:
        mask = X_all['Country'] == country
        X_country = X_all[mask].copy()
        y_country = y[mask].copy()
        sort_idx = X_country.sort_values(['Year', 'Month']).index
        X_country = X_country.loc[sort_idx].reset_index(drop=True)
        y_country = y_country.loc[sort_idx].reset_index(drop=True)

        n = len(X_country)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X_country.iloc[:train_end]
        X_val = X_country.iloc[train_end:val_end]
        X_test = X_country.iloc[val_end:]
        y_train = y_country.iloc[:train_end]
        y_val = y_country.iloc[train_end:val_end]
        y_test = y_country.iloc[val_end:]

        X_train_combined = pd.concat([X_train_combined, X_train])
        X_val_combined = pd.concat([X_val_combined, X_val])
        X_test_combined = pd.concat([X_test_combined, X_test])
        y_train_combined = pd.concat([y_train_combined, y_train['rd_expenditure']])
        y_val_combined = pd.concat([y_val_combined, y_val['rd_expenditure']])
        y_test_combined = pd.concat([y_test_combined, y_test['rd_expenditure']])

    X_train = X_train_combined.reset_index(drop=True)
    X_val = X_val_combined.reset_index(drop=True)
    X_test = X_test_combined.reset_index(drop=True)
    y_train = y_train_combined.reset_index(drop=True).to_frame('rd_expenditure')
    y_val = y_val_combined.reset_index(drop=True).to_frame('rd_expenditure')
    y_test = y_test_combined.reset_index(drop=True).to_frame('rd_expenditure')

    # Build MGT relevant columns from original MGT code
    base_keys = ['Year', 'Month', 'Country']
    gt_ytd = [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]
    gt_lags = []
    for lag in range(1, max_lag+1):
        gt_lags += [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]

    desired = [c for c in base_keys + gt_ytd + gt_lags if c in X_all.columns]

    X_train = X_train[desired].drop_duplicates()
    X_val = X_val[desired].drop_duplicates()
    X_test = X_test[desired].drop_duplicates()
    y_train = y_train.loc[X_train.index]
    y_val = y_val.loc[X_val.index]
    y_test = y_test.loc[X_test.index]

    return X_train, X_val, X_test, y_train, y_val, y_test

base.create_train_test_splits = create_train_test_splits

if __name__ == "__main__":
    base.main()


