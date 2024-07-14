import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import statsmodels.api as sm

combined_df = pd.read_csv('/Nowcasting/temporal_disaggregation/results/combined_estimates.csv')

combined_df['rd_exp_yr_true'] = combined_df.groupby(['Year'])['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'].transform('sum')

# Perform the groupby operation while keeping 'Year' and 'Country' as columns
grouped = combined_df.groupby(['Year'], as_index=False)

# Calculate the monthly naive estimate and ensure the result is a DataFrame
monthly_naive_estimate = grouped['rd_exp_yr_true'].mean().rename(columns={'rd_exp_yr_true': 'rd_exp_month_naive_estimate'})

# Since we used .mean(), we now need to divide by 12 to get the monthly estimate
monthly_naive_estimate['rd_exp_month_naive_estimate'] /= 12

# Since the original dataframe has 'Month' and the estimate does not, we only merge on 'Year' and 'Country'
combined_df = combined_df.merge(monthly_naive_estimate, on=['Year'], how='left')

#############################################################################
## Tests

# Load US additional data
us_employment = pd.read_csv('/Nowcasting/data/datausa.io/Monthly Employment.csv')
us_employment=us_employment[['Month of Year ID', 'Month of Year', 'NSA Employees']]

# Extract year and month from 'Month of Year ID'
us_employment['Year'] = us_employment['Month of Year ID'].apply(lambda x: int(x.split('-')[0]))
us_employment['Month'] = us_employment['Month of Year ID'].apply(lambda x: int(x.split('-')[1]))
us_employment=us_employment[['Year', 'Month', 'NSA Employees']]

us_employment['Year'] = us_employment['Year'].astype(int)
us_employment['Month'] = us_employment['Month'].astype(int)

# Assuming combined_df needs the same adjustment
combined_df['Year'] = combined_df['Year'].astype(int)
combined_df['Month'] = combined_df['Month'].astype(int)

combined_df = combined_df[combined_df['Country'] == 'US']

########################################

# Merging all data together based on 'Year' and 'Month'
combined_df_us = combined_df.merge(us_employment, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.drop_duplicates()

combined_df_us.to_csv('/Nowcasting/temporal_disaggregation/results/combined_df_us_summary.csv', index=False)


# Calculate growth rates and correlations
growth_columns = ['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley', 'rd_exp_month_naive_estimate', 'NSA Employees']
rd_growth_columns = ['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley', 'rd_exp_month_naive_estimate']

#combined_df_us = combined_df_us[combined_df_us['Year']>=2008]
combined_df_us.reset_index(inplace=True)

combined_df_us.sort_values(by=['Year', 'Month'], inplace=True)

for col in growth_columns:
    combined_df_us[f'{col}_growth'] = combined_df_us[col].pct_change() * 100

combined_df_us.to_csv('/Nowcasting/temporal_disaggregation/results/combined_df_us_growths.csv', index=False)

# Drop rows with NaN in any of the growth columns
combined_df_us.dropna(subset=[f'{col}_growth' for col in growth_columns], inplace=True)

# Calculate correlations with lags
correlation_results = []
lags = range(-6, 7)  # From -2 to 2
for lag in lags:
    for col1 in rd_growth_columns:
        for col2 in growth_columns:
            #if col1 != col2:
                shifted_df = combined_df_us.copy()
                shifted_df[col1 + '_growth'] = shifted_df[col1 + '_growth'].shift(lag)
                valid_rows = shifted_df.dropna(subset=[col1 + '_growth', col2 + '_growth'])
                if not valid_rows.empty:
                    correlation, p_value = pearsonr(valid_rows[col1 + '_growth'], valid_rows[col2 + '_growth'])
                    correlation_results.append({'Variable 1': f'{col1}_growth', 'Variable 2': f'{col2}_growth', 'Lag': lag, 'Correlation': correlation, 'P-Value': p_value})

correlation_df = pd.DataFrame(correlation_results)
correlation_df.to_csv('/Nowcasting/temporal_disaggregation/results/correlations_with_lags_all.csv', index=False)

#################################################################################
#################################################################################

# Merging all data together based on 'Year' and 'Month'
combined_df_us = combined_df.merge(us_employment, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.drop_duplicates()

combined_df_us.to_csv('/Nowcasting/temporal_disaggregation/results/combined_df_us_summary.csv', index=False)

# Calculate growth rates and correlations
growth_columns = ['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley', 'rd_exp_month_naive_estimate', 'NSA Employees']
rd_growth_columns = ['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley', 'rd_exp_month_naive_estimate']

combined_df_us = combined_df_us[combined_df_us['Year']>=2008]
combined_df_us.reset_index(drop=True, inplace=True)

combined_df_us.sort_values(by=['Year', 'Month'], inplace=True)

for col in growth_columns:
    combined_df_us[f'{col}_growth'] = combined_df_us[col].pct_change() * 100

combined_df_us.to_csv('/Nowcasting/temporal_disaggregation/results/combined_df_us_growths.csv', index=False)

# Drop rows with NaN in any of the growth columns
combined_df_us.dropna(subset=[f'{col}_growth' for col in growth_columns], inplace=True)

# Exclude data for January
combined_df_us_correl = combined_df_us[combined_df_us['Month'] != 1]

# Calculate correlations with lags
correlation_results = []
lags = range(-6, 7)  # From -2 to 2
for lag in lags:
    for col1 in rd_growth_columns:
        for col2 in growth_columns:
            #if col1 != col2:
                shifted_df = combined_df_us_correl.copy()
                shifted_df[col1 + '_growth'] = shifted_df[col1 + '_growth'].shift(lag)
                valid_rows = shifted_df.dropna(subset=[col1 + '_growth', col2 + '_growth'])
                if not valid_rows.empty:
                    correlation, p_value = pearsonr(valid_rows[col1 + '_growth'], valid_rows[col2 + '_growth'])
                    correlation_results.append({'Variable 1': f'{col1}_growth', 'Variable 2': f'{col2}_growth', 'Lag': lag, 'Correlation': correlation, 'P-Value': p_value})

correlation_df = pd.DataFrame(correlation_results)

correlation_df.to_csv('/Nowcasting/temporal_disaggregation/results/correlations_with_lags.csv', index=False)


#################################################################################
#################################################################################

## extension

# Assuming combined_df_us is already defined and contains the necessary columns
combined_df_us['Date'] = pd.to_datetime(combined_df_us[['Year', 'Month']].assign(DAY=1))

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the first line and scatter plot
ax.plot(combined_df_us['Date'], combined_df_us['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], label='R&D Expenditure Monthly Estimate', color='tab:blue')
ax.scatter(combined_df_us['Date'], combined_df_us['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity'], color='tab:blue', s=10)  # s is the size of the points

# Creating a secondary y-axis for the second variable
ax2 = ax.twinx()

# Plotting the second line and scatter plot
ax2.plot(combined_df_us['Date'], combined_df_us['NSA Employees'], label='Scientific R&D Services Employees', color='tab:orange')
ax2.scatter(combined_df_us['Date'], combined_df_us['NSA Employees'], color='tab:orange', s=10)  # s is the size of the points

# Setting the title and labels
ax.set_title('R&D Expenditure Monthly Estimates for US')
ax.set_xlabel('Date')
ax.set_ylabel('R&D Expenditure Monthly Estimate (Bio.)')
ax2.set_ylabel('Scientific R&D Services Employees')

# Adding legends
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Showing the plot
#plt.show()

# Saving the figure
plt.savefig(f"/Nowcasting/temporal_disaggregation/results/combined_df_us_monthly_data_employee.pdf")
combined_df_us.to_csv('/Nowcasting/temporal_disaggregation/results/combined_df_us_4plotcheck_employee.csv', index=False)

#################################################################################

## Plots for US: between the naive estimate and the variable

# Assuming combined_df_us is already defined and contains the necessary columns
combined_df_us['Date'] = pd.to_datetime(combined_df_us[['Year', 'Month']].assign(DAY=1))

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the first line and scatter plot
ax.plot(combined_df_us['Date'], combined_df_us['rd_exp_month_naive_estimate'], label='R&D Expenditure Monthly Naive Estimate', color='tab:blue')
ax.scatter(combined_df_us['Date'], combined_df_us['rd_exp_month_naive_estimate'], color='tab:blue', s=10)  # s is the size of the points

# Creating a secondary y-axis for the second variable
ax2 = ax.twinx()

# Plotting the second line and scatter plot
ax2.plot(combined_df_us['Date'], combined_df_us['NSA Employees'], label='Scientific R&D Services Employees', color='tab:orange')
ax2.scatter(combined_df_us['Date'], combined_df_us['NSA Employees'], color='tab:orange', s=10)  # s is the size of the points

# Setting the title and labels
ax.set_title('R&D Expenditure Naive Monthly Estimates for US')
ax.set_xlabel('Date')
ax.set_ylabel('R&D Expenditure Naive Monthly Estimate (Bio.)')
ax2.set_ylabel('Scientific R&D Services Employees')

# Adding legends
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Showing the plot
#plt.show()

# Saving the figure
plt.savefig(f"/Nowcasting/temporal_disaggregation/results/combined_df_us_naive_monthly_data_employee.pdf")


