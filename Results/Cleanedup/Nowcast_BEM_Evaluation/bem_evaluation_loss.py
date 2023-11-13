import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_estimate.csv")

# Calculate loss for each row
df['loss'] = (df['monthly_rd_expenditure'] - df['estimated_rd_expenditure']).abs()

# Calculate RMSE, MAE, MAPE for each country in general
results_general = df.groupby('Country').apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(((group['monthly_rd_expenditure'] - group['estimated_rd_expenditure'])**2).mean()),
        'MAE': (group['monthly_rd_expenditure'] - group['estimated_rd_expenditure']).abs().mean(),
        'MAPE': (group.loc[group['monthly_rd_expenditure'] != 0, 'loss'] / group.loc[group['monthly_rd_expenditure'] != 0, 'monthly_rd_expenditure']).mean() * 100
    })
)

# Calculate RMSE, MAE, MAPE for each month and each year
#df['Year'] = df['date'].dt.year
#df['Month'] = df['date'].dt.month

results_monthly = df.groupby(['Month']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(((group['monthly_rd_expenditure'] - group['estimated_rd_expenditure'])**2).mean()),
        'MAE': (group['monthly_rd_expenditure'] - group['estimated_rd_expenditure']).abs().mean(),
        'MAPE': (group.loc[group['monthly_rd_expenditure'] != 0, 'loss'] / group.loc[group['monthly_rd_expenditure'] != 0, 'monthly_rd_expenditure']).mean() * 100
    })
)

results_yearly = df.groupby(['Year']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(((group['monthly_rd_expenditure'] - group['estimated_rd_expenditure'])**2).mean()),
        'MAE': (group['monthly_rd_expenditure'] - group['estimated_rd_expenditure']).abs().mean(),
        'MAPE': (group.loc[group['monthly_rd_expenditure'] != 0, 'loss'] / group.loc[group['monthly_rd_expenditure'] != 0, 'monthly_rd_expenditure']).mean() * 100
    })
)

print(results_general)
print(results_monthly)
print(results_yearly)

# Save the results to CSV files
results_general.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/results_general.csv")
results_monthly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/results_monthly.csv")
results_yearly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/results_yearly.csv")

##########

# Aggregate data by country and year
aggregated_df = df.groupby(['Country', 'Year']).agg({
    'monthly_rd_expenditure': 'sum',
    'estimated_rd_expenditure': 'sum'
}).reset_index()

# Calculate the loss for each row
aggregated_df['loss'] = aggregated_df['estimated_rd_expenditure'] - aggregated_df['monthly_rd_expenditure']

# Calculate the RMSE, MAE, and MAPE for each country
results_aggregated = aggregated_df.groupby('Country').apply(lambda group: pd.Series({
    'RMSE': np.sqrt((group['loss']**2).mean()),
    'MAE': group['loss'].abs().mean(),
    'MAPE': (group['loss'].abs() / group['monthly_rd_expenditure']).mean() * 100
})).reset_index()

# Save the results to a CSV file
results_aggregated.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/results_aggregated.csv", index=False)

print(results_aggregated)


##########################################
##########################################

# Read the data
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/df_trends_monthly_estimate.csv")

# Calculate loss for each row
df['loss'] = (df['monthly_rd_expenditure'] - df['estimated_rd_expenditure']).abs()

# Calculate RMSE, MAE, MAPE for each country in general
results_general = df.groupby('Country').apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(((group['monthly_rd_expenditure'] - group['estimated_rd_expenditure'])**2).mean()),
        'MAE': (group['monthly_rd_expenditure'] - group['estimated_rd_expenditure']).abs().mean(),
        'MAPE': (group.loc[group['monthly_rd_expenditure'] != 0, 'loss'] / group.loc[group['monthly_rd_expenditure'] != 0, 'monthly_rd_expenditure']).mean() * 100
    })
)

# Calculate RMSE, MAE, MAPE for each month and each year
#df['Year'] = df['date'].dt.year
#df['Month'] = df['date'].dt.month

results_monthly = df.groupby(['Month']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(((group['monthly_rd_expenditure'] - group['estimated_rd_expenditure'])**2).mean()),
        'MAE': (group['monthly_rd_expenditure'] - group['estimated_rd_expenditure']).abs().mean(),
        'MAPE': (group.loc[group['monthly_rd_expenditure'] != 0, 'loss'] / group.loc[group['monthly_rd_expenditure'] != 0, 'monthly_rd_expenditure']).mean() * 100
    })
)

results_yearly = df.groupby(['Year']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(((group['monthly_rd_expenditure'] - group['estimated_rd_expenditure'])**2).mean()),
        'MAE': (group['monthly_rd_expenditure'] - group['estimated_rd_expenditure']).abs().mean(),
        'MAPE': (group.loc[group['monthly_rd_expenditure'] != 0, 'loss'] / group.loc[group['monthly_rd_expenditure'] != 0, 'monthly_rd_expenditure']).mean() * 100
    })
)

print(results_general)
print(results_monthly)
print(results_yearly)

# Save the results to CSV files
results_general.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/results_general.csv")
results_monthly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/results_monthly.csv")
results_yearly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/results_yearly.csv")


##########

# Aggregate data by country and year
aggregated_df = df.groupby(['Country', 'Year']).agg({
    'monthly_rd_expenditure': 'sum',
    'estimated_rd_expenditure': 'sum'
}).reset_index()

# Calculate the loss for each row
aggregated_df['loss'] = aggregated_df['estimated_rd_expenditure'] - aggregated_df['monthly_rd_expenditure']

# Calculate the RMSE, MAE, and MAPE for each country
results_aggregated = aggregated_df.groupby('Country').apply(lambda group: pd.Series({
    'RMSE': np.sqrt((group['loss']**2).mean()),
    'MAE': group['loss'].abs().mean(),
    'MAPE': (group['loss'].abs() / group['monthly_rd_expenditure']).mean() * 100
})).reset_index()

# Save the results to a CSV file
results_aggregated.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/results_aggregated.csv", index=False)

print(results_aggregated)


