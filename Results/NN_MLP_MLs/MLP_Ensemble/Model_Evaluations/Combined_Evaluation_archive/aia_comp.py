import pandas as pd
import os
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# List of filenames
files = [
    "combined_results_Related.csv", 
    "combined_results_KWO.csv", 
    "combined_results_KWO_CrossCountry.csv",
    "combined_results_KWO_CrossCountry_XGBoost_variations.csv",
    "combined_results_KWO_CrossCountry_NN_MLP.csv"
]

path_to_files = "/Users/atin/Nowcasting/data/Results/Combined_Evaluation/"  # Replace with your path
all_dfs = []

# Iterate through each file
for file in files:
    full_path = os.path.join(path_to_files, file)
    df = pd.read_csv(full_path)
    
    # Extract model variation from filename and assign it as a new column
    model_variation = file.replace("combined_results_", "").replace(".csv", "")
    df['Variation'] = model_variation

    all_dfs.append(df)

# Combine all dataframes into one
combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)

# Save the combined dataframe if needed
combined_df.to_csv(os.path.join(path_to_files, "/Users/atin/Nowcasting/data/Results/Combined_Evaluation/combined_all_results.csv"), index=False)

# Comparisons can now be made on the combined_df DataFrame. 
# For example, for an overall comparison:
print(combined_df.groupby('Variation').mean())

# For models with the same lag time (e.g., lag-3):
lag_3_df = combined_df[combined_df['Lag'] == 3]
print(lag_3_df.groupby('Variation').mean())


# Set a style for the plots
sns.set_style("whitegrid")

# Overall results
overall_means = combined_df.groupby('Variation').mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='Variation', y='RMSE', data=overall_means)
plt.title('Overall RMSE by Model Variation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(path_to_files, "/Users/atin/Nowcasting/data/Results/Combined_Evaluation/overall_rmse_comparison.png"))

# Models with lag=3
lag_3_means = combined_df[combined_df['Lag'] == 3].groupby('Variation').mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='Variation', y='RMSE', data=lag_3_means)
plt.title('RMSE by Model Variation for lag=3')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(path_to_files, "/Users/atin/Nowcasting/data/Results/Combined_Evaluation/lag3_rmse_comparison.png"))

# Models with lag=3 and Sub-Model as AllVar
lag_3_allvar_means = combined_df[(combined_df['Lag'] == 3) & (combined_df['Sub-Model'] == 'AllVar')].groupby('Variation').mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='Variation', y='RMSE', data=lag_3_allvar_means)
plt.title('RMSE by Model Variation for lag=3 with Sub-Model as AllVar')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(path_to_files, "/Users/atin/Nowcasting/data/Results/Combined_Evaluation/lag3_allvar_rmse_comparison.png"))

plt.show()
