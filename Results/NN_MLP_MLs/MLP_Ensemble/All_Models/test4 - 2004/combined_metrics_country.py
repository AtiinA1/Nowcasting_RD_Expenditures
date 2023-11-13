import os
import pandas as pd

# Base directory containing all model directories
base_dir = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/"

# Extract model variations based on directory names
model_variations = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

all_dfs = []

def explode_mlp_df(df):
    # Create separate DataFrames for each model
    model_1_df = df.copy()
    model_1_df['Algorithm'] = 'MLP-Pytorch'
    model_1_df['RMSE'] = df['RMSE'].apply(lambda x: eval(x)[0])
    model_1_df['MAE'] = df['MAE'].apply(lambda x: eval(x)[0])
    model_1_df['MAPE'] = df['MAPE'].apply(lambda x: eval(x)[0])
    
    model_2_df = df.copy()
    model_2_df['Algorithm'] = 'MLP-Pytorch'
    model_2_df['RMSE'] = df['RMSE'].apply(lambda x: eval(x)[1])
    model_2_df['MAE'] = df['MAE'].apply(lambda x: eval(x)[1])
    model_2_df['MAPE'] = df['MAPE'].apply(lambda x: eval(x)[1])
    
    ensemble_df = df.copy()
    ensemble_df['Algorithm'] = 'MLP-Pytorch'
    ensemble_df['RMSE'] = df['RMSE'].apply(lambda x: eval(x)[2])
    ensemble_df['MAE'] = df['MAE'].apply(lambda x: eval(x)[2])
    ensemble_df['MAPE'] = df['MAPE'].apply(lambda x: eval(x)[2])
    
    return pd.concat([model_1_df, model_2_df, ensemble_df])

# Iterate through each model variation directory 
for variation in model_variations:
    test2_dir = os.path.join(base_dir, variation, "test4")

    # Load country metrics from MLP-PyTorch
    mlp_filepath = os.path.join(test2_dir, 'country_metrics.csv')

    # For the MLP-PyTorch part:
    if os.path.exists(mlp_filepath):
        mlp_df = pd.read_csv(mlp_filepath)
        mlp_df = explode_mlp_df(mlp_df)
        mlp_df['Model_Variation'] = variation
        mlp_df = mlp_df[['Country', 'Algorithm', 'Model_Variation', 'RMSE', 'MAE', 'MAPE']]
        all_dfs.append(mlp_df)

    # if os.path.exists(mlp_filepath):
    #     mlp_df = pd.read_csv(mlp_filepath)
    #     mlp_df = mlp_df.explode('Models').explode('RMSE').explode('MAE').explode('MAPE')
    #     mlp_df['Algorithm'] = mlp_df['Models']
    #     mlp_df['Model_Variation'] = variation
    #     mlp_df = mlp_df[['Country', 'Algorithm', 'Model_Variation', 'RMSE', 'MAE', 'MAPE']]
    #     all_dfs.append(mlp_df)

    # Load country evaluation results from other models
    for model_name in ['ElasticNet', 'MLPSci', 'RF', 'SVR', 'XGBoost']:
        model_dir = os.path.join(test2_dir, model_name)
        country_filepath = os.path.join(model_dir, f'country_evaluation.csv')
        
        if os.path.exists(country_filepath):
            model_df = pd.read_csv(country_filepath)
            model_df['Algorithm'] = model_name if model_name != "MLPSci" else "MLP-Scikit"
            model_df['Model_Variation'] = variation
            all_dfs.append(model_df)

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/combined_country_results.csv", index=False)


import seaborn as sns
import matplotlib.pyplot as plt

combined_df_rev = combined_df[(combined_df['Algorithm']!= 'MLP-Scikit')]

# Assuming combined_df is the DataFrame with all your results
def plot_metrics(metric):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df_rev, x='Algorithm', y=metric, hue='Model_Variation')
    plt.title(f'Boxplot of {metric} by Model Variation and Algorithm')
    plt.ylabel(metric)
    plt.xlabel('Algorithm')
    plt.xticks(rotation=45)
    plt.legend(title='Model_Variation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/boxplot_{metric}.png', bbox_inches='tight')
    plt.show()

# Plot for each metric
for metric in ['MAPE', 'RMSE', 'MAE']:
    plot_metrics(metric)

# List of selected algorithms to focus on
selected_algorithms = ['MLP-Pytorch', 'XGBoost', 'RF']

# Filter the DataFrame
filtered_df = combined_df[combined_df['Algorithm'].isin(selected_algorithms)]

def plot_zoomed_metrics(metric):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x='Algorithm', y=metric, hue='Model_Variation')
    plt.title(f'Boxplot of {metric} by Model Variation (Zoomed)')
    plt.ylabel(metric)
    plt.xlabel('Algorithm')
    plt.xticks(rotation=45)
    plt.legend(title='Model_Variation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the zoomed figure
    plt.savefig(f'/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/zoomed_boxplot_{metric}.png', bbox_inches='tight')
    plt.show()

# Plot for each metric
for metric in ['MAPE', 'RMSE', 'MAE']:
    plot_zoomed_metrics(metric)

