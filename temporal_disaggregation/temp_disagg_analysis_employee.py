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
us_gdp = pd.read_csv('/Nowcasting/data/FRED/realGDP_FRED.csv')
us_employment = pd.read_csv('/Nowcasting/data/datausa.io/Monthly Employment.csv')

us_pat_application_pre_grant = pd.read_csv('/Nowcasting/data/USPTO/pg_published_application.tsv', sep='\t')
us_pat_assignee_pre_grant = pd.read_csv('/Nowcasting/data/USPTO/pg_assignee_disambiguated.tsv', sep='\t')

us_pat_application_granted = pd.read_csv('/Nowcasting/data/USPTO/g_application_USPTO.tsv', sep='\t')
us_pat_assignee_granted = pd.read_csv('/Nowcasting/data/USPTO/g_assignee_disambiguated.tsv', sep='\t')


us_pat_application_pre_grant = us_pat_application_pre_grant[['pgpub_id', 'filing_date']]
us_pat_assignee_pre_grant = us_pat_assignee_pre_grant[['pgpub_id', 'assignee_type']]

#us_pat_assignee_pre_grant = us_pat_assignee_pre_grant[(us_pat_assignee_pre_grant['assignee_type'] != 1) & (us_pat_assignee_pre_grant['assignee_type'] != 3) & (us_pat_assignee_pre_grant['assignee_type'] != 5) & (us_pat_assignee_pre_grant['assignee_type'] != 7)]

us_pat_application_pre_grant['filing_date'] = us_pat_application_pre_grant['filing_date'].astype(str)


# Extract Year and Month, handling cases where filing_date might be 'nan'
us_pat_application_pre_grant['Year'] = us_pat_application_pre_grant['filing_date'].apply(
    lambda x: int(x.split('-')[0]) if '-' in x else np.nan
)
us_pat_application_pre_grant['Month'] = us_pat_application_pre_grant['filing_date'].apply(
    lambda x: int(x.split('-')[1]) if '-' in x else np.nan
)

us_pat_application_pre_grant = us_pat_application_pre_grant.drop(columns=['filing_date'])

#us_pat_appl_pre_grant = us_pat_application_pre_grant.merge(us_pat_assignee_pre_grant, on=['pgpub_id'], how='inner')
us_pat_appl_pre_grant = us_pat_application_pre_grant

us_pat_appl_pre_grant.dropna(inplace=True)
# Remove duplicates, assuming unique entries are defined by both 'patent_id' and 'application_id'
us_pat_appl_pre_grant = us_pat_appl_pre_grant.drop_duplicates()

# Calculate monthly patent filings and add as a new column directly within df
us_pat_appl_pre_grant['Monthly_Filings'] = us_pat_appl_pre_grant.groupby(['Year', 'Month'])['pgpub_id'].transform('nunique')
us_pat_appl_pre_grant['Yearly_Filings'] = us_pat_appl_pre_grant.groupby(['Year'])['pgpub_id'].transform('nunique')

us_pat_appl_pre_grant['Monthly_Filings'] = us_pat_appl_pre_grant['Monthly_Filings'].astype(int)
us_pat_appl_pre_grant['Yearly_Filings'] = us_pat_appl_pre_grant['Yearly_Filings'].astype(int)

us_pat_appl_pre_grant = us_pat_appl_pre_grant[['Year', 'Month', 'Monthly_Filings', 'Yearly_Filings']]
us_pat_appl_pre_grant = us_pat_appl_pre_grant.drop_duplicates()

#####

us_pat_application = us_pat_application_granted[['patent_id', 'filing_date']]
us_pat_assignee_granted = us_pat_assignee_granted[['patent_id', 'assignee_type']]

us_pat_assignee_granted = us_pat_assignee_granted[(us_pat_assignee_granted['assignee_type'] != 1) & (us_pat_assignee_granted['assignee_type'] != 3) & (us_pat_assignee_granted['assignee_type'] != 5) & (us_pat_assignee_granted['assignee_type'] != 7)]

us_pat_application['Year'] = us_pat_application['filing_date'].apply(lambda x: int(x.split('-')[0]))
us_pat_application['Month'] = us_pat_application['filing_date'].apply(lambda x: int(x.split('-')[1]))

# Drop 'filing_date' column
us_pat_application = us_pat_application.drop(columns=['filing_date'])
us_pat_application = us_pat_application.drop_duplicates()

us_pat_application = us_pat_application.merge(us_pat_assignee_granted, on=['patent_id'], how='inner')

# Remove duplicates, assuming unique entries are defined by both 'patent_id' and 'application_id'
us_pat_application = us_pat_application.drop_duplicates()

# Calculate monthly patent filings and add as a new column directly within df
us_pat_application['Monthly_Granted'] = us_pat_application.groupby(['Year', 'Month'])['patent_id'].transform('nunique')
us_pat_application['Yearly_Granted'] = us_pat_application.groupby(['Year'])['patent_id'].transform('nunique')

us_pat_application['Monthly_Granted'] = us_pat_application['Monthly_Granted'].astype(int)
us_pat_application['Yearly_Granted'] = us_pat_application['Yearly_Granted'].astype(int)

us_pat_application=us_pat_application[['Year', 'Month', 'Monthly_Granted', 'Yearly_Granted']]
us_pat_application = us_pat_application.drop_duplicates()

########################################

us_gdp['DATE'] = pd.to_datetime(us_gdp['DATE'])
us_gdp.set_index('DATE', inplace=True)
us_gdp_monthly = us_gdp.resample('M').ffill()
us_gdp_monthly.reset_index(inplace=True)
us_gdp_monthly['Month'] = us_gdp_monthly['DATE'].dt.month
us_gdp_monthly['Year'] = us_gdp_monthly['DATE'].dt.year
us_gdp_monthly = us_gdp_monthly.drop(columns=['DATE'])
us_gdp = us_gdp_monthly

########################################

us_employment=us_employment[['Month of Year ID', 'Month of Year', 'NSA Employees']]

# Extract year and month from 'Month of Year ID'
us_employment['Year'] = us_employment['Month of Year ID'].apply(lambda x: int(x.split('-')[0]))
us_employment['Month'] = us_employment['Month of Year ID'].apply(lambda x: int(x.split('-')[1]))
us_employment=us_employment[['Year', 'Month', 'NSA Employees']]

########################################

us_gdp['Year'] = us_gdp['Year'].astype(int)
us_gdp['Month'] = us_gdp['Month'].astype(int)

us_pat_application['Year'] = us_pat_application['Year'].astype(int)
us_pat_application['Month'] = us_pat_application['Month'].astype(int)

us_employment['Year'] = us_employment['Year'].astype(int)
us_employment['Month'] = us_employment['Month'].astype(int)

# Assuming combined_df needs the same adjustment
combined_df['Year'] = combined_df['Year'].astype(int)
combined_df['Month'] = combined_df['Month'].astype(int)

combined_df = combined_df[combined_df['Country'] == 'US']

########################################

# Merging all data together based on 'Year' and 'Month'
combined_df_us = combined_df.merge(us_gdp, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.merge(us_pat_application, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.merge(us_pat_appl_pre_grant, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.merge(us_employment, on=['Year', 'Month'], how='left')

combined_df_us = combined_df_us.drop_duplicates()

combined_df_us.to_csv('/Nowcasting/temporal_disaggregation/results/combined_df_us_summary.csv', index=False)


# Calculate growth rates and correlations
growth_columns = ['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley', 'rd_exp_month_naive_estimate', 'NSA Employees', 'Monthly_Filings', 'Monthly_Granted', 'GDP']
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
combined_df_us = combined_df.merge(us_gdp, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.merge(us_pat_application, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.merge(us_pat_appl_pre_grant, on=['Year', 'Month'], how='left')
combined_df_us = combined_df_us.merge(us_employment, on=['Year', 'Month'], how='left')

combined_df_us = combined_df_us.drop_duplicates()

combined_df_us.to_csv('/Nowcasting/temporal_disaggregation/results/combined_df_us_summary.csv', index=False)

# Calculate growth rates and correlations
growth_columns = ['Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity', 'Monthly_RD_Expenditure_Tempdisagg_Sax', 'Monthly_RD_Expenditure_Tempdisagg_Mosley', 'rd_exp_month_naive_estimate', 'NSA Employees', 'Monthly_Filings', 'Monthly_Granted', 'GDP']
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


