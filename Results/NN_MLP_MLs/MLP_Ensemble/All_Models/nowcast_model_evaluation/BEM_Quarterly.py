import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataframe (you should read your csv here)
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_estimate.csv")

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Group by 'Country' and then resample to aggregate the estimated R&D expenditure quarterly
df_quarterly = df.groupby('Country').resample('Q').agg({'estimated_rd_expenditure': 'sum'}).reset_index()

# Rename the 'estimated_rd_expenditure' column to indicate it's now quarterly
df_quarterly.rename(columns={'estimated_rd_expenditure': 'quarterly_estimated_rd_expenditure'}, inplace=True)
# Save the dataframe to a CSV file
df_quarterly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/quarterly_rd_expenditure.csv", index=False)

# Plotting
plt.figure(figsize=(15, 8))

# Iterate over each country and plot its trend
for country in df_quarterly['Country'].unique():
    country_data = df_quarterly[df_quarterly['Country'] == country]
    plt.plot(country_data['date'], country_data['quarterly_estimated_rd_expenditure'], label=country)

# Decorating the plot
plt.title("Quarterly R&D Expenditure Estimate Over Time")
plt.xlabel("Date")
plt.ylabel("Quarterly Estimated R&D Expenditure")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Saving the plot to a file
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/quarterly_rd_expenditure_plot.png", format='png', dpi=300)

plt.show()

print(df_quarterly)

breakpoint()

# Sample dataframe (you should read your csv here)
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_estimate.csv")

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Group by 'Country' and then resample to aggregate the estimated R&D expenditure yearly
df_yearly = df.groupby('Country').resample('Y').agg({'estimated_rd_expenditure': 'sum'}).reset_index()

# Rename the 'estimated_rd_expenditure' column to indicate it's now yearly
df_yearly.rename(columns={'estimated_rd_expenditure': 'yearly_estimated_rd_expenditure'}, inplace=True)

# Save the dataframe to a CSV file
df_yearly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/yearly_rd_expenditure.csv", index=False)

# Plotting Yearly Data
plt.figure(figsize=(15, 8))

# Iterate over each country and plot its trend
for country in df_yearly['Country'].unique():
    country_data = df_yearly[df_yearly['Country'] == country]
    plt.plot(country_data['date'], country_data['yearly_estimated_rd_expenditure'], label=country)

# Decorating the plot
plt.title("Yearly R&D Expenditure Estimate Over Time")
plt.xlabel("Date")
plt.ylabel("Yearly Estimated R&D Expenditure")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Saving the plot to a file
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/yearly_rd_expenditure_plot.png", format='png', dpi=300)

plt.show()


#################################################

# Sample dataframe (you should read your csv here)
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/df_trends_monthly_estimate.csv")

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Group by 'Country' and then resample to aggregate the estimated R&D expenditure quarterly
df_quarterly = df.groupby('Country').resample('Q').agg({'estimated_rd_expenditure': 'sum'}).reset_index()

# Rename the 'estimated_rd_expenditure' column to indicate it's now quarterly
df_quarterly.rename(columns={'estimated_rd_expenditure': 'quarterly_estimated_rd_expenditure'}, inplace=True)
# Save the dataframe to a CSV file
df_quarterly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/quarterly_rd_expenditure.csv", index=False)

# Plotting
plt.figure(figsize=(15, 8))

# Iterate over each country and plot its trend
for country in df_quarterly['Country'].unique():
    country_data = df_quarterly[df_quarterly['Country'] == country]
    plt.plot(country_data['date'], country_data['quarterly_estimated_rd_expenditure'], label=country)

# Decorating the plot
plt.title("Quarterly R&D Expenditure Estimate Over Time")
plt.xlabel("Date")
plt.ylabel("Quarterly Estimated R&D Expenditure")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Saving the plot to a file
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/quarterly_rd_expenditure_plot.png", format='png', dpi=300)

plt.show()

print(df_quarterly)

breakpoint()

# Sample dataframe (you should read your csv here)
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/df_trends_monthly_estimate.csv")

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Group by 'Country' and then resample to aggregate the estimated R&D expenditure yearly
df_yearly = df.groupby('Country').resample('Y').agg({'estimated_rd_expenditure': 'sum'}).reset_index()

# Rename the 'estimated_rd_expenditure' column to indicate it's now yearly
df_yearly.rename(columns={'estimated_rd_expenditure': 'yearly_estimated_rd_expenditure'}, inplace=True)

# Save the dataframe to a CSV file
df_yearly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/yearly_rd_expenditure.csv", index=False)

# Plotting Yearly Data
plt.figure(figsize=(15, 8))

# Iterate over each country and plot its trend
for country in df_yearly['Country'].unique():
    country_data = df_yearly[df_yearly['Country'] == country]
    plt.plot(country_data['date'], country_data['yearly_estimated_rd_expenditure'], label=country)

# Decorating the plot
plt.title("Yearly R&D Expenditure Estimate Over Time")
plt.xlabel("Date")
plt.ylabel("Yearly Estimated R&D Expenditure")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Saving the plot to a file
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/test10-mlp-pytorch/yearly_rd_expenditure_plot.png", format='png', dpi=300)

plt.show()
