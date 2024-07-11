import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

pred_monthly = pd.read_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AGT/test10_related_topic_final_states_fixed_dimension_bem/df_trends_monthly_elast_adjusted_onpreds.csv')
pred_monthly = pred_monthly[['Year','Month','Country','monthly_rd_expenditure','monthly_rd_expenditure_onpreds']]
pred_monthly = pred_monthly.rename(columns={'monthly_rd_expenditure': 'Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'monthly_rd_expenditure_onpreds':'Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity_onPreds'})  

combined_df = pred_monthly
combined_df = combined_df[combined_df['Country'] == 'US']
combined_df = combined_df[['Year','Month','Country','Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity']]
combined_df.drop_duplicates()

combined_df_org = combined_df

tempdisagg_sax_df = pd.read_csv('/Nowcasting/temporal_disaggregation/classical_regression_tempdisagg/Disaggregated_Monthly_RD_Expenditure_Sax.csv')
tempdisagg_sax_df = tempdisagg_sax_df.rename(columns={'Monthly_RD_Expenditure': 'Monthly_RD_Expenditure_Tempdisagg_Sax'})  

tempdisagg_mosley_df = pd.read_csv('/Nowcasting/temporal_disaggregation/classical_regression_tempdisagg/test1/Disaggregated_RD_Expenditure_spTD.csv')
tempdisagg_mosley_df = tempdisagg_mosley_df.rename(columns={'Monthly_RD_Expenditure': 'Monthly_RD_Expenditure_Tempdisagg_Mosley'})  

combined_df = combined_df.merge(tempdisagg_sax_df, on=['Year', 'Month'], how='left')
combined_df = combined_df.merge(tempdisagg_mosley_df, on=['Year', 'Month'], how='left')

combined_df.to_csv('/Nowcasting/temporal_disaggregation/results/combined_estimates.csv', index=False)

#################################################################################################################

# Load the data
df = combined_df

#################################################################################################################

# Ensuring Year and Month columns are integers and creating a datetime column
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

# Plotting the data with a primary and secondary y-axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the first variable as both line and scatter
ax.plot(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], label='GT_NNelasticity Estimate', color='tab:blue')
ax.scatter(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], color='tab:blue', s=10)  # s is the size of the scatter points

# Assume another variable for secondary axis (example only)
# Uncomment the following lines if another series for the secondary axis is available
# ax2 = ax.twinx()
# ax2.plot(df['Date'], df['Some_Other_Variable'], label='Secondary Variable', color='tab:orange')
# ax2.scatter(df['Date'], df['Some_Other_Variable'], color='tab:orange', s=10)
# ax2.set_ylabel('Secondary Variable Label')

# Setting titles and labels
ax.set_title('Monthly R&D Expenditure Estimates with GT_NNelasticity')
ax.set_xlabel('Date')
ax.set_ylabel('GT_NNelasticity Expenditure')

# Adding legends
ax.legend(loc='upper left')
# ax2.legend(loc='upper right')  # Uncomment if using a secondary axis

plt.grid(True)
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.savefig('/Nowcasting/temporal_disaggregation/results/GT_NNelasticity_time_series_with_scatter.png')  # Save the figure
plt.show()

#################################################################################################################

# Creating the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting each estimate as both line and scatter
ax.plot(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], label='GT_NNelasticity Estimate', color='tab:blue')
ax.scatter(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], color='tab:blue', s=10)

ax.plot(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_Sax'], label='Chow-Lin Estimate', color='tab:green')
ax.scatter(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_Sax'], color='tab:green', s=10)

ax.plot(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_Mosley'], label='Mosley Estimate', color='tab:red')
ax.scatter(df['Date'], df['Monthly_RD_Expenditure_Tempdisagg_Mosley'], color='tab:red', s=10)

# Setting titles and labels
ax.set_title('Monthly R&D Expenditure Estimates')
ax.set_xlabel('Date')
ax.set_ylabel('R&D Expenditure')

# Adding a legend
ax.legend(loc='upper left')

# Enhancing plot aesthetics
plt.grid(True)
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.savefig('/Nowcasting/temporal_disaggregation/results/all_estimates_time_series.png')  # Save the figure
plt.show()


#################################################################################################################
# Calculate the min and max across the specified columns for each point in time
df['Min_Estimate'] = df[['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley']].min(axis=1)
df['Max_Estimate'] = df[['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley']].max(axis=1)

df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Plot 0: NN-driven Elasticity-based Estimator
plt.figure(figsize=(12, 6))
plt.fill_between(df.index, df['Min_Estimate'], df['Max_Estimate'], color='gray', alpha=0.3, label='Range (Min-Max)')
plt.plot(df['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], label='GT_NNelasticity', color='blue', marker='o')
plt.title('Range of Monthly R&D Expenditure Estimates with NN-driven Elasticity-based Estimates Highlighted')

# Setting date formatter for x-axis to display only the year
year_locator = mdates.YearLocator()  # Locate years
year_formatter = mdates.DateFormatter('%Y')  # Format as YYYY
plt.gca().xaxis.set_major_locator(year_locator)
plt.gca().xaxis.set_major_formatter(year_formatter)

# Rotate date labels for better legibility
plt.xticks(rotation=45)

plt.xlabel('Date')
plt.ylabel('R&D Expenditure')
plt.legend()
plt.grid(True)
plt.savefig('/Nowcasting/temporal_disaggregation/results/all_time_series_plot_on_GT_NNelasticity.pdf')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(df['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], label='GT_NNelasticity', color='blue', marker='o')
plt.title('NN-driven Elasticity-based Estimates for Monthly R&D Expenditures in the US')

# Setting date formatter for x-axis to display only the year
year_locator = mdates.YearLocator()  # Locate years
year_formatter = mdates.DateFormatter('%Y')  # Format as YYYY
plt.gca().xaxis.set_major_locator(year_locator)
plt.gca().xaxis.set_major_formatter(year_formatter)

# Rotate date labels for better legibility
plt.xticks(rotation=45)

plt.xlabel('Date')
plt.ylabel('R&D Expenditure')
plt.legend()
plt.grid(True)
plt.savefig('/Nowcasting/temporal_disaggregation/results/GT_NNelasticity_wo_range.pdf')
plt.show()

#################################################################################################################

# Plot 1: Tempdisagg_Sax
plt.figure(figsize=(12, 6))
plt.fill_between(df.index, df['Min_Estimate'], df['Max_Estimate'], color='gray', alpha=0.3, label='Range (Min-Max)')
plt.plot(df['Monthly_RD_Expenditure_Tempdisagg_Sax'], label='Tempdisagg_Chow_Lin', color='purple', marker='o')
plt.title('Range of Monthly R&D Expenditure Estimates with Temporal Disaggregation by Chow-Lin Highlighted')
plt.xlabel('Date')
plt.ylabel('R&D Expenditure')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(year_locator)
plt.gca().xaxis.set_major_formatter(year_formatter)
plt.savefig('/Nowcasting/temporal_disaggregation/results/all_time_series_plot_on_Tempdisagg_Sax.pdf')
plt.show()

# Plot 2: Tempdisagg_Mosley
plt.figure(figsize=(12, 6))
plt.fill_between(df.index, df['Min_Estimate'], df['Max_Estimate'], color='gray', alpha=0.3, label='Range (Min-Max)')
plt.plot(df['Monthly_RD_Expenditure_Tempdisagg_Mosley'], label='Tempdisagg_Mosley', color='green', marker='o')
plt.title('Range of Monthly R&D Expenditure Estimates with Temporal Disaggregation by Mosley Highlighted')
plt.xlabel('Date')
plt.ylabel('R&D Expenditure')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(year_locator)
plt.gca().xaxis.set_major_formatter(year_formatter)
plt.savefig('/Nowcasting/temporal_disaggregation/results/all_time_series_plot_on_Tempdisagg_Mosley.pdf')
plt.show()

# Plot 3: Combined plot
plt.figure(figsize=(12, 6))
plt.fill_between(df.index, df['Min_Estimate'], df['Max_Estimate'], color='gray', alpha=0.3, label='Range (Min-Max)')
plt.plot(df['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], label='GT_NNelasticity', color='blue', marker='o')
plt.plot(df['Monthly_RD_Expenditure_Tempdisagg_Sax'], label='Tempdisagg_Chow_Lin', color='purple', marker='x')
plt.plot(df['Monthly_RD_Expenditure_Tempdisagg_Mosley'], label='Tempdisagg_Mosley', color='green', marker='^')
plt.title('Range of Monthly R&D Expenditure Estimates with GT_NNelasticity Highlight')
plt.xlabel('Date')
plt.ylabel('R&D Expenditure')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(year_locator)
plt.gca().xaxis.set_major_formatter(year_formatter)
plt.savefig('/Nowcasting/temporal_disaggregation/results/all_time_series_plot_allest.pdf')
plt.show()

