import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in your data
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/Combined_Evaluation/combined_all_results_org.csv")
df = df[df['Model'] != 'LinReg']

## -------------------------------------------------------------------------
### Evaluation of ML Algorithms ###

# Filter the data to include only specific variations and ML algorithms
df_filtered = df[df['Variation'].isin(['KWO', 'Related', 'KWO_CrossCountry'])]
# Create a new column to represent the unique combination of Model, Sub-Model, and Variation
df_filtered['SubModel_Variation'] = df_filtered['Sub-Model'] + '-' + df_filtered['Variation']

# Plotting for MAE
plt.figure(figsize=(18, 8))
sns.boxplot(x='Model', y='MAE', data=df_filtered, hue="SubModel_Variation")
plt.title('MAE by Model, Sub-Model, and Variation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
plt.savefig("/Users/atin/Nowcasting/data/Results/Combined_Evaluation/ML_Overview_Model_SubModel_Variation_MAEOnly.png", dpi=300)

# Plotting for MAPE
plt.figure(figsize=(18, 8))
sns.boxplot(x='Model', y='MAPE', data=df_filtered, hue="SubModel_Variation")
plt.title('MAPE by Model, Sub-Model, and Variation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
plt.savefig("/Users/atin/Nowcasting/data/Results/Combined_Evaluation/ML_Overview_Model_SubModel_Variation_MAPEOnly.png", dpi=300)

breakpoint()

## -------------------------------------------------------------------------

df_filtered['SubModel_lag'] = df_filtered['Sub-Model'].astype(str) + "_" + df_filtered['Lag'].astype(str)

# MAE grouped bar plot
plt.figure(figsize=(15, 8))
ax1 = sns.barplot(data=df_filtered, x="Model", y="MAE", hue="SubModel_lag", ci=None) 
ax1.set_title('MAE for each Model, Sub-Model and lag')
plt.ylabel('MAE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/Combined_Evaluation/ML_Overview_Model_SubModel_Lag_MAEOnly.png', dpi=300)
plt.show()

plt.figure(figsize=(15, 8))
ax1 = sns.barplot(data=df_filtered, x="Model", y="MAPE", hue="SubModel_lag", ci=None) 
ax1.set_title('MAPE for each Model, Sub-Model and lag')
plt.ylabel('MAPE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/Combined_Evaluation/ML_Overview_Model_SubModel_Lag_MAPEOnly.png', dpi=300)

breakpoint()

## -------------------------------------------------------------------------
# Create a figure with two subplots, one for MAE and one for RMSE
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Create the MAE subplot
sns.boxplot(data=df_filtered, x='Model', y='MAE', hue='Sub-Model',
            ax=axes[0], palette='Set3')
axes[0].set_title('MAE Comparison')
axes[0].set_yscale('log')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Mean Absolute Error (log scale)')

# Create the RMSE subplot
sns.boxplot(data=df_filtered, x='Model', y='MAPE', hue='Sub-Model',
            ax=axes[1], palette='Set3')
axes[1].set_title('MAPE Comparison')
axes[1].set_yscale('log')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Mean Absolute Percentage Error (log scale)')

# Final layout adjustments
plt.tight_layout()

# Save the figure
plt.savefig("/Users/atin/Nowcasting/data/Results/Combined_Evaluation/ML_comparison_figure_alllags.png", dpi=300)

breakpoint()
#df.dropna()
df_lag3 = df_filtered[df_filtered['Lag']==3]

# Create a figure with two subplots, one for MAE and one for RMSE
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Create the MAE subplot
sns.boxplot(data=df_lag3, x='Model', y='MAE', hue='Sub-Model',
            ax=axes[0], palette='Set3')
axes[0].set_title('MAE Comparison')
axes[0].set_yscale('log')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Mean Absolute Error (log scale)')

# Create the RMSE subplot
sns.boxplot(data=df_lag3, x='Model', y='MAPE', hue='Sub-Model',
            ax=axes[1], palette='Set3')
axes[1].set_title('MAPE Comparison')
axes[1].set_yscale('log')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Mean Absolute Percentage Error (log scale)')

# Final layout adjustments
plt.tight_layout()

# Save the figure
plt.savefig("/Users/atin/Nowcasting/data/Results/Combined_Evaluation/ML_comparison_figure_lag3.png", dpi=300)

breakpoint()

## -------------------------------------------------------------------------
# Figure 1: Boxplot Comparing RMSE Across Different Models (Overall)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Model', y='RMSE')
plt.title("Overall RMSE Across Different Models")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.yscale('log') # Optional: log scale for better visualization
plt.xticks(rotation=90)
plt.savefig('/Users/atin/Nowcasting/data/Results/Combined_Evaluation/highlighted_figure1.png')
plt.show()

# Figure 2: Boxplot Comparing RMSE Across Different Models and Sub-Models (Lag=3)
plt.figure(figsize=(12, 6))
df_lag3 = df[df['Lag']==3]
df_lag3.dropna()

print(df.isna().sum())
print(df['Lag'].dtypes)
print(df[df['Lag']=='3'].head())

breakpoint()

sns.boxplot(data=df_lag3, x='Model', y='RMSE', hue='Sub-Model')
plt.title("RMSE Across Different Models and Sub-Models (Lag=3)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.yscale('log') # Optional: log scale for better visualization
plt.xticks(rotation=90)
plt.savefig('/Users/atin/Nowcasting/data/Results/Combined_Evaluation/highlighted_figure2.png')
plt.show()

# Figure 3: Heatmap of MAE Across Countries for Different Models
plt.figure(figsize=(12, 6))
heatmap_data = df.pivot_table(values='MAE', index='Country', columns='Model')
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True)
plt.title("Heatmap of MAE Across Countries for Different Models")
plt.xticks(rotation=90)
plt.savefig('/Users/atin/Nowcasting/data/Results/Combined_Evaluation/highlighted_figure3.png')
plt.show()

# Figure 4: Line Plot Showing MAPE for Different Models Over Different Lags
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Lag', y='MAPE', hue='Model')
plt.title("MAPE for Different Models Over Different Lags")
plt.xlabel("Lag")
plt.ylabel("MAPE")
plt.xticks(rotation=90)
plt.savefig('/Users/atin/Nowcasting/data/Results/Combined_Evaluation/highlighted_figure4.png')
plt.show()

## -------------------------------------------------------------------------
# XGBoost Only
df_filtered_XGBoost = df[(df['Model'] == 'XGBoost') | (df['Model'] == 'XGBRegressor_woInteractionTerms') | (df['Model'] == 'XGBRegressor_wInteractionTerms') | (df['Model'] == 'XGBRegressor_wdrop_winteract') | (df['Model'] == 'XGBRegressor_wdrop_wointeract')]
df_filtered_XGBoost = df_filtered_XGBoost[df_filtered_XGBoost['Sub-Model'] == 'AllVar']
df_filtered_XGBoost = df_filtered_XGBoost[df_filtered_XGBoost['Variation'].isin(['KWO_CrossCountry', 'KWO_CrossCountry_XGBoost_variations'])]
df_filtered_XGBoost['Model_Lag'] = df_filtered_XGBoost['Model'].astype(str) + "_" + df_filtered_XGBoost['Lag'].astype(str)

# Plotting Combined only for XGBoost (wo variations)
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
sns.barplot(x='Variation', y='MAE', data=df_filtered_XGBoost, hue='Model_Lag', ax=axs[0])
sns.barplot(x='Variation', y='MAPE', data=df_filtered_XGBoost, hue='Model_Lag', ax=axs[1])

axs[0].set_title('MAE for XGBoost Model by Sub-Model, and Variation')
axs[0].set_xticklabels(axs[0].get_xticklabels())

axs[1].set_title('MAPE for XGBoost Model by Sub-Model, and Variation')
axs[1].set_xticklabels(axs[1].get_xticklabels())


plt.tight_layout()
plt.show()
plt.savefig("/Users/atin/Nowcasting/data/Results/Combined_Evaluation/XGBoost_Overview_Model_SubModel_Variation_wovars.png", dpi=300)

breakpoint()

## -------------------------------------------------------------------------
# Filter the data to include only specific variations and ML algorithms
df_AllVar_Lag3 = df[df['Sub-Model'] == 'AllVar']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Lag']==3]
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Country'] == 'CrossCountry']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'LinSVR']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'RF']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'LinReg']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'NN_MLPPytorch_Model_1']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'NN_MLPPytorch_Model_2']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'NN_MLPPytorch_Model_3']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'NN_MLPPytorch_Model_4']
df_AllVar_Lag3 = df_AllVar_Lag3[df_AllVar_Lag3['Model'] != 'NN_MLPPytorch_Model_5']


# Plotting for MAE & MAPE
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
sns.barplot(x='Model', y='MAE', data=df_AllVar_Lag3, ax=axs[0])
sns.barplot(x='Model', y='MAPE', data=df_AllVar_Lag3, ax=axs[1])

axs[0].set_title('MAE for different versions of XGBoosts and NNs')
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

axs[1].set_title('MAPE for different versions of XGBoosts and NNs')
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()
plt.savefig("/Users/atin/Nowcasting/data/Results/Combined_Evaluation/XGBvNN_Overview_AllVar_Lag3.png", dpi=300)

