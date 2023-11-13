import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Base directory where all model variation folders are located
base_dir = '/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/'

# Extract model variations based on directory names
model_variations = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Placeholder to collect dataframes
all_dfs = []

# Iterate through each model variation directory 
for variation in model_variations:
    test2_dir = os.path.join(base_dir, variation, 'test3')
    
    # Load ensemble MLP-Pytorch results
    mlp_filepath = os.path.join(test2_dir, 'metrics_df_minibatchGD.csv')
    if os.path.exists(mlp_filepath):
        mlp_df = pd.read_csv(mlp_filepath)
        mlp_df['Algorithm'] = 'MLP-Pytorch'
        mlp_df['Model_Variation'] = variation
        mlp_df['Lag'] = 3
        all_dfs.append(mlp_df)

    # Load results from other models
    for model_name in ['ElasticNet', 'MLPSci', 'RF', 'SVR', 'XGBoost']:
        model_dir = os.path.join(test2_dir, model_name)
        matching_files = [f for f in os.listdir(model_dir) if f.startswith('evaluation_test_') and f.endswith('.csv')]
        
        if matching_files:
            model_filepath = os.path.join(model_dir, matching_files[0])  # Assuming there's only one matching file per directory
            print(f"Reading data from: {model_filepath}")  # Diagnostic print
            model_df = pd.read_csv(model_filepath)
            model_df['Algorithm'] = model_name if model_name != "MLPSci" else "MLP-Scikit"
            model_df['Model_Variation'] = variation
            model_df['Lag'] = 3
            model_df.rename(columns={
                'MAE': 'Test_MAE',
                'MAPE': 'Test_MAPE',
                'RMSE': 'Test_RMSE'
            }, inplace=True)            
            all_dfs.append(model_df)
        else:
            print(f"No matching file found in: {os.path.join(test2_dir, model_name)}")  # Diagnostic print


# Combine all dataframes
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Save to CSV
    combined_df.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/final_evaluation.csv', index=False)
else:
    print("No dataframes were loaded.")

# combined_df = pd.concat(all_dfs, ignore_index=True)
# combined_df.to_csv("/path/to/save/combined_results.csv", index=False)

print(combined_df.columns)
breakpoint()

def plot_bar_metrics(metric):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=combined_df, x='Algorithm', y=metric, hue='Model_Variation')
    plt.title(f'Barplot of {metric} by Model Variation')
    plt.ylabel(metric)
    plt.xlabel('Model Variation')
    plt.xticks(rotation=45)
    plt.legend(title='Model_Variation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/barplot_{metric}.png', bbox_inches='tight')
    plt.show()

# Plot for each metric
for metric in ['Test_MAPE', 'Test_RMSE', 'Test_MAE']:
    plot_bar_metrics(metric)


duplicates = combined_df[combined_df.duplicated(subset=['Algorithm', 'Model_Variation'], keep=False)]
print(duplicates)
breakpoint()

# Pivot the dataframe to format it suitably for a heatmap
#heatmap_data = combined_df.pivot('Algorithm', 'Model_Variation', 'Test_MAPE')

# Keep only rows where 'Model' is 'Ensemble' or 'Model' is NaN
filtered_df = combined_df[(combined_df['Algorithm']!= 'MLP-Scikit')]
filtered_df = filtered_df[(filtered_df['Algorithm']!= 'ElasticNet')]
filtered_df = filtered_df[(filtered_df['Algorithm']!= 'SVR')]
filtered_df = filtered_df[(filtered_df['Model'] == 'Ensemble') | (filtered_df['Model'].isna())]

# Then, pivot the filtered dataframe for the heatmap
heatmap_data = filtered_df.pivot('Algorithm', 'Model_Variation', 'Test_MAPE')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Test_MAPE'})
plt.title('MAPE Heatmap across Model Variations and Algorithms for Ensemble Models')
plt.savefig(f'/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/heatmap_mape.png', bbox_inches='tight')
plt.show()

breakpoint()


# Pivot the dataframe to format it suitably for a heatmap
heatmap_data = filtered_df.pivot('Algorithm', 'Model_Variation', 'Test_RMSE')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Test_RMSE'})
plt.title('MAPE Heatmap across Model Variations and Algorithms')
plt.savefig(f'/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/heatmap_RMSE.png', bbox_inches='tight')
plt.show()


# Pivot the dataframe to format it suitably for a heatmap
heatmap_data = filtered_df.pivot('Algorithm', 'Model_Variation', 'Test_MAE')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Test_MAE'})
plt.title('MAPE Heatmap across Model Variations and Algorithms')
plt.savefig(f'/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/heatmap_maE.png', bbox_inches='tight')
plt.show()

