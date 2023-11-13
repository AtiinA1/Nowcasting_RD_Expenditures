import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/combined_country_results.csv")

# List of selected algorithms to focus on
selected_algorithms = ['MLP-Pytorch', 'MLP-Scikit', 'XGBoost', 'RF']

# Filter the DataFrame
filtered_df = df[df['Algorithm'].isin(selected_algorithms)]

df = df[df['Algorithm'] != 'LinReg']
df = df[df['Algorithm'] != 'SVR']
df = df[df['Algorithm'] != 'ElasticNet']
df = df[df['Algorithm'] != 'MLP-Scikit']

# Create a figure with three subplots, one for each metric
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot MAE
sns.boxplot(data=df, x='Algorithm', y='MAE', hue='Model_Variation', ax=ax[0])
ax[0].set_title('Mean Absolute Error Comparison')
ax[0].set_xlabel('')
ax[0].set_yscale('log') # Use logarithmic scale for better visualization

# Plot RMSE
sns.boxplot(data=df, x='Algorithm', y='RMSE', hue='Model_Variation', ax=ax[1])
ax[1].set_title('Root Mean Square Error Comparison')
ax[0].set_xlabel('')
ax[1].set_yscale('log') # Use logarithmic scale for better visualization

# Plot MAPE
sns.boxplot(data=df, x='Algorithm', y='MAPE', hue='Model_Variation', ax=ax[2])
ax[2].set_title('Mean Absolute Percentage Error Comparison')
ax[2].set_yscale('log') # Use logarithmic scale for better visualization

plt.tight_layout()

# Save the figure
plt.savefig('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/combined_results_V3_KWO_boxplot1.png', dpi=300)

#plt.show()


#----------------------------------------------------------------------------------

# Create a figure and axis
fig, ax = plt.subplots(3, 1, figsize=(12, 8))

# Create a boxplot of MAE values
sns.boxplot(x='Algorithm', y='MAE', hue='Model_Variation', data=df, ax=ax[0])
ax[0].set_title('MAE Boxplot for each Model and Model_Variation')
ax[0].set_ylabel('MAE')

# Create a boxplot of RMSE values
sns.boxplot(x='Algorithm', y='RMSE', hue='Model_Variation', data=df, ax=ax[1])
ax[1].set_title('RMSE Boxplot for each Model and Model_Variation')
ax[1].set_ylabel('RMSE')

# Create a boxplot of MAPE values
sns.boxplot(x='Algorithm', y='MAPE', hue='Model_Variation', data=df, ax=ax[2])
ax[2].set_title('MAPE Boxplot for each Model and Model_Variation')
ax[2].set_ylabel('MAPE')

# Show the plot
plt.tight_layout()

# Save the figure
plt.savefig('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/combined_results_V3_KWO_boxplot2.png', dpi=300)

# Show the plot
#plt.show()


#-------------------------------------------------------------------------------------

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='Algorithm', y='MAE', hue='Model_Variation')
plt.title('Comparison of MAE values across Models and Model_Variations for Common Countries')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/MAE_boxplot_common_V3_KWO_model_submodel_MAE.png")
#plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='Algorithm', y='RMSE', hue='Model_Variation')
plt.title('Comparison of MAE values across Models and Model_Variations for Common Countries')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/MAE_boxplot_common_V3_KWO_model_submodel_RMSE.png")
#plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='Algorithm', y='MAPE', hue='Model_Variation')
plt.title('Comparison of MAE values across Models and Model_Variations for Common Countries')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/MAE_boxplot_common_V3_KWO_model_submodel_MAPE.png")
#plt.show()

######################


df['SubModel_lag'] = df['Model_Variation'].astype(str) + "_" + df['Lag'].astype(str)


# MAE grouped bar plot
plt.figure(figsize=(15, 8))
ax1 = sns.barplot(data=df, x="Model", y="MAE", hue="SubModel_lag", ci=None) 
ax1.set_title('MAE for each Model, Model_Variation and lag')
plt.ylabel('MAE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/combined_results_V3_barplot2_MAE_model_subm_lag.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Model", y="RMSE", hue="SubModel_lag", ci=None) # concatenate Model and Model_Variation
ax2.set_title('RMSE for each Model, Model_Variation and lag')
plt.ylabel('RMSE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/combined_results_V3_barplot2_RMSE_model_subm_lag.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Model", y="MAPE", hue="SubModel_lag", ci=None) # concatenate Model and Model_Variation
ax2.set_title('MAPE for each Model, Model_Variation and lag')
plt.ylabel('MAPE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model_Evaluations/test10/combined_results_V3_barplot2_MAPE_model_subm_lag.png', dpi=300)
#plt.show()


