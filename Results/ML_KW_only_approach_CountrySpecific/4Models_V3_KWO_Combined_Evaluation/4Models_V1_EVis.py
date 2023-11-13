import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
df_raw = pd.read_csv("/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_KWO.csv")


def get_common_countries(df):
    model_submodel_pairs = df[['Model', 'Sub-Model']].drop_duplicates().values.tolist()
    country_sets = [set(df[(df['Model'] == pair[0]) & (df['Sub-Model'] == pair[1])]['Country']) for pair in model_submodel_pairs]
    common_countries = set.intersection(*country_sets)
    return common_countries

common_countries = get_common_countries(df_raw)
df = df_raw[df_raw['Country'].isin(common_countries)]

df = df[df['Model'] != 'LinReg']
df = df[df['Model'] != 'LinSVR']
df = df[df['Model'] != 'ElasticNet']

# Create a figure with three subplots, one for each metric
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot MAE
sns.boxplot(data=df, x='Model', y='MAE', hue='Sub-Model', ax=ax[0])
ax[0].set_title('Mean Absolute Error Comparison')
ax[0].set_xlabel('')
ax[0].set_yscale('log') # Use logarithmic scale for better visualization

# Plot RMSE
sns.boxplot(data=df, x='Model', y='RMSE', hue='Sub-Model', ax=ax[1])
ax[1].set_title('Root Mean Square Error Comparison')
ax[0].set_xlabel('')
ax[1].set_yscale('log') # Use logarithmic scale for better visualization

# Plot MAPE
sns.boxplot(data=df, x='Model', y='MAPE', hue='Sub-Model', ax=ax[2])
ax[2].set_title('Mean Absolute Percentage Error Comparison')
ax[2].set_yscale('log') # Use logarithmic scale for better visualization

plt.tight_layout()

# Save the figure
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_KWO_boxplot1.png', dpi=300)

#plt.show()


#----------------------------------------------------------------------------------

# Create a figure and axis
fig, ax = plt.subplots(3, 1, figsize=(12, 8))

# Create a boxplot of MAE values
sns.boxplot(x='Model', y='MAE', hue='Sub-Model', data=df, ax=ax[0])
ax[0].set_title('MAE Boxplot for each Model and Sub-Model')
ax[0].set_ylabel('MAE')

# Create a boxplot of RMSE values
sns.boxplot(x='Model', y='RMSE', hue='Sub-Model', data=df, ax=ax[1])
ax[1].set_title('RMSE Boxplot for each Model and Sub-Model')
ax[1].set_ylabel('RMSE')

# Create a boxplot of MAPE values
sns.boxplot(x='Model', y='MAPE', hue='Sub-Model', data=df, ax=ax[2])
ax[2].set_title('MAPE Boxplot for each Model and Sub-Model')
ax[2].set_ylabel('MAPE')

# Show the plot
plt.tight_layout()

# Save the figure
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_KWO_boxplot2.png', dpi=300)

# Show the plot
#plt.show()


#------------------------------------------------------------------------------------


# MAE grouped bar plot
plt.figure(figsize=(15, 8))
ax1 = sns.barplot(data=df, x="Country", y="MAE", hue="Model", ci=None)
ax1.set_title('MAE for each Model and Country')
plt.ylabel('MAE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_KWO_barplot2_mae.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Country", y="RMSE", hue="Model", ci=None)
ax2.set_title('RMSE for each Model and Country')
plt.ylabel('RMSE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_KWO_barplot2_rmse.png', dpi=300)
#plt.show()

# MAPE grouped bar plot
plt.figure(figsize=(15, 8))
ax3 = sns.barplot(data=df, x="Country", y="MAPE", hue="Model", ci=None)
ax3.set_title('MAPE for each Model and Country')
plt.ylabel('MAPE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_KWO_barplot2_mape.png', dpi=300)
#plt.show()

#-------------------------------------------------------------------------------------

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='Model', y='MAE', hue='Sub-Model')
plt.title('Comparison of MAE values across Models and Sub-Models for Common Countries')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/MAE_boxplot_common_V3_KWO_model_submodel_mae.png")
#plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='Model', y='RMSE', hue='Sub-Model')
plt.title('Comparison of MAE values across Models and Sub-Models for Common Countries')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/MAE_boxplot_common_V3_KWO_model_submodel_rmse.png")
#plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='Model', y='MAPE', hue='Sub-Model')
plt.title('Comparison of MAE values across Models and Sub-Models for Common Countries')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/MAE_boxplot_common_V3_KWO_model_submodel_mape.png")
#plt.show()

######################

g = sns.FacetGrid(df, col="Lag", row="Sub-Model", height=4, aspect=1)
g.map_dataframe(sns.barplot, x="Country", y="MAE", hue="Model", ci=None)
g.set_axis_labels("Country", "MAE")
g.add_legend()
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/mae.png", dpi=300)
plt.show()

g = sns.FacetGrid(df, col="Lag", row="Sub-Model", height=4, aspect=1)
g.map_dataframe(sns.barplot, x="Country", y="RMSE", hue="Model", ci=None)
g.set_axis_labels("Country", "RMSE")
g.add_legend()
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/rmse.png", dpi=300)
#plt.show()

g = sns.FacetGrid(df, col="Lag", row="Sub-Model", height=4, aspect=1)
g.map_dataframe(sns.barplot, x="Country", y="MAPE", hue="Model", ci=None)
g.set_axis_labels("Country", "MAPE")
g.add_legend()
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/MAPE.png", dpi=300)
#plt.show()

######################

df['Model_Lag'] = df['Model'].astype(str) + "_" + df['Lag'].astype(str)


# MAE grouped bar plot
plt.figure(figsize=(15, 8))
ax1 = sns.barplot(data=df, x="Country", y="MAE", hue="Model_Lag", ci=None) 
ax1.set_title('MAE for each Model, Lag and Country')
plt.ylabel('MAE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_mae_country_lag.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Country", y="RMSE", hue="Model_Lag", ci=None) # concatenate Model and Lag
ax2.set_title('RMSE for each Model, Lag and Country')
plt.ylabel('RMSE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_rmse_country_lag.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Country", y="MAPE", hue="Model_Lag", ci=None) # concatenate Model and Lag
ax2.set_title('MAPE for each Model, Lag and Country')
plt.ylabel('MAPE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_mape_country_lag.png', dpi=300)
#plt.show()

######################

df['Model_Submodel'] = df['Model'].astype(str) + "_" + df['Sub-Model'].astype(str)


# MAE grouped bar plot
plt.figure(figsize=(15, 8))
ax1 = sns.barplot(data=df, x="Country", y="MAE", hue="Model_Submodel", ci=None) 
ax1.set_title('MAE for each Model, Sub-Model and Country')
plt.ylabel('MAE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_mae_country_subm.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Country", y="RMSE", hue="Model_Submodel", ci=None) # concatenate Model and Sub-Model
ax2.set_title('RMSE for each Model, Sub-Model and Country')
plt.ylabel('RMSE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_rmse_country_subm.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Country", y="MAPE", hue="Model_Submodel", ci=None) # concatenate Model and Sub-Model
ax2.set_title('MAPE for each Model, Sub-Model and Country')
plt.ylabel('MAPE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_mape_country_subm.png', dpi=300)
#plt.show()

######################

df['SubModel_lag'] = df['Sub-Model'].astype(str) + "_" + df['Lag'].astype(str)


# MAE grouped bar plot
plt.figure(figsize=(15, 8))
ax1 = sns.barplot(data=df, x="Model", y="MAE", hue="SubModel_lag", ci=None) 
ax1.set_title('MAE for each Model, Sub-Model and lag')
plt.ylabel('MAE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_mae_model_subm_lag.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Model", y="RMSE", hue="SubModel_lag", ci=None) # concatenate Model and Sub-Model
ax2.set_title('RMSE for each Model, Sub-Model and lag')
plt.ylabel('RMSE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_rmse_model_subm_lag.png', dpi=300)
#plt.show()

# RMSE grouped bar plot
plt.figure(figsize=(15, 8))
ax2 = sns.barplot(data=df, x="Model", y="MAPE", hue="SubModel_lag", ci=None) # concatenate Model and Sub-Model
ax2.set_title('MAPE for each Model, Sub-Model and lag')
plt.ylabel('MAPE')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_Combined_Evaluation/combined_results_V3_barplot2_mape_model_subm_lag.png', dpi=300)
#plt.show()


