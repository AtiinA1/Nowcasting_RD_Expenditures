import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

#---------------------------------------------------------------------------------------------------------------------------

country_cd_df = pd.read_csv('/Users/atin/Nowcasting/data/country_code.csv')


# Read in and filter rd_expenditure_df to only include rows with sector of performance equal to Business enterprise + considering total fundings (millions-LOCAL CURRENCY)
rd_expenditure_df = pd.read_csv('/Users/atin/Nowcasting/data/GERD/DP_LIVE_08052023154811337.csv')
rd_expenditure_df = rd_expenditure_df.rename(columns={'Value': 'rd_expenditure'})  
rd_expenditure_df = rd_expenditure_df[rd_expenditure_df['MEASURE'] == 'MLN_USD'] #USD constant prices using 2015 base year 
print(rd_expenditure_df.columns)


# Read and filter n_patents
n_patents_df = pd.read_csv('/Users/atin/Nowcasting/data/OECD/PATS_IPC_11062023234902217.csv')
n_patents_df = n_patents_df[n_patents_df['IPC'] == 'TOTAL']
n_patents_df = n_patents_df[n_patents_df['KINDDATE'] == 'PRIORITY']
n_patents_df = n_patents_df[n_patents_df['KINDCOUNTRY'] == 'INVENTORS']
print(n_patents_df.columns)
n_patents_df = n_patents_df[['LOCATION','TIME','Value']]
n_patents_df = n_patents_df.rename(columns={'Value': 'n_patents'})
print(n_patents_df.columns)


# Read and filter IMF (various macro variables)
imf_df = pd.read_csv('/Users/atin/Nowcasting/data/IMF/WEOApr2023all.csv')

# Define the id_vars - these are the columns that we want to keep as they are.
id_vars = ['WEO Country Code', 'ISO', 'WEO Subject Code', 'Country', 
           'Subject Descriptor', 'Subject Notes', 'Units', 'Scale', 
           'Country/Series-specific Notes', 'Estimates Start After']

# All other columns are the years. We can use a list comprehension to get this list.
year_columns = [col for col in imf_df.columns if col not in id_vars]

# Use the melt function to reshape the DataFrame
imf_df_rev = imf_df.melt(id_vars=id_vars, 
                  value_vars=year_columns, 
                  var_name='TIME', 
                  value_name='Value')

print(imf_df_rev.columns)

gdpca_df = imf_df_rev[imf_df_rev['WEO Subject Code'] == 'PPPPC'] #GDP Per Capita, current prices, PPP; international dollars
gdpca_df = gdpca_df[['ISO','TIME','Value']]
gdpca_df = gdpca_df.rename(columns={'Value': 'gdpca', 'ISO':'LOCATION'})   

unemp_rate_df = imf_df_rev[imf_df_rev['WEO Subject Code'] == 'LUR'] #unemployment rate
unemp_rate_df = unemp_rate_df[['ISO','TIME','Value']]
unemp_rate_df = unemp_rate_df.rename(columns={'Value': 'unemp_rate', 'ISO':'LOCATION'})       


population_df = imf_df_rev[imf_df_rev['WEO Subject Code'] == 'LP'] #population
population_df = population_df[['ISO','TIME','Value']]
population_df = population_df.rename(columns={'Value': 'population', 'ISO':'LOCATION'}) 


inflation_df = imf_df_rev[imf_df_rev['WEO Subject Code'] == 'PCPI'] #Inflation, average consumer prices
inflation_df = inflation_df[['ISO','TIME','Value']]
inflation_df = inflation_df.rename(columns={'Value': 'inflation', 'ISO':'LOCATION'}) 


export_df = imf_df_rev[imf_df_rev['WEO Subject Code'] == 'TX_RPCH'] #Volume of exports of goods and services
export_df = export_df[['ISO','TIME','Value']]
export_df = export_df.rename(columns={'Value': 'export_vol', 'ISO':'LOCATION'}) 

import_df = imf_df_rev[imf_df_rev['WEO Subject Code'] == 'TM_RPCH'] #Volume of imports of goods and services
import_df = import_df[['ISO','TIME','Value']]
import_df = import_df.rename(columns={'Value': 'import_vol', 'ISO':'LOCATION'}) 


gdpca_df['TIME'] = gdpca_df['TIME'].astype(int)
unemp_rate_df['TIME'] = unemp_rate_df['TIME'].astype(int)
population_df['TIME'] = population_df['TIME'].astype(int)
inflation_df['TIME'] = inflation_df['TIME'].astype(int)
export_df['TIME'] = export_df['TIME'].astype(int)
import_df['TIME'] = import_df['TIME'].astype(int)

# Assuming that 'TIME' column in oecd_df is also a string
rd_expenditure_df['TIME'] = rd_expenditure_df['TIME'].astype(int)
n_patents_df['TIME'] = n_patents_df['TIME'].astype(int)

#oecd_df = pd.merge(rd_expenditure_df, n_patents_df, how='left' , on=['LOCATION', 'TIME'])
oecd_df = pd.merge(rd_expenditure_df, gdpca_df, how='left' , on=['LOCATION', 'TIME'])
oecd_df = pd.merge(oecd_df, unemp_rate_df, how='left' , on=['LOCATION', 'TIME'])
oecd_df = pd.merge(oecd_df, population_df, how='left' , on=['LOCATION', 'TIME'])
oecd_df = pd.merge(oecd_df, inflation_df, how='left' , on=['LOCATION', 'TIME'])
oecd_df = pd.merge(oecd_df, export_df, how='left' , on=['LOCATION', 'TIME'])
oecd_df = pd.merge(oecd_df, import_df, how='left' , on=['LOCATION', 'TIME'])

# Merge the two DataFrames
country_cd_df = country_cd_df[['alpha-2', 'alpha-3']]
oecd_df_rev = pd.merge(oecd_df, country_cd_df, how='left' , left_on='LOCATION', right_on='alpha-3')
# Rename the columns
oecd_df_rev = oecd_df_rev.rename(columns={'alpha-2': 'Country', 'TIME': 'Year'})

print(oecd_df_rev.columns)

oecd_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol']]

# Define a list of all variables in oecd_df_rev that you want to check for missing values
variables = ['rd_expenditure', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol']

# Convert problematic strings to NaN and other string numbers to proper numeric type
for var in variables:
    if oecd_df_rev[var].dtype == 'object':
        # Convert "--" to NaN
        oecd_df_rev[var] = oecd_df_rev[var].replace('--', np.nan)
        
        # Remove commas and convert to float
        oecd_df_rev[var] = oecd_df_rev[var].str.replace(',', '').astype(float)

# Loop through each variable
for var in variables:
    # Create a binary column to indicate missing values for the variable
    oecd_df_rev[var + '_missing'] = oecd_df_rev[var].isna().astype(int)
    
    if var != 'rd_expenditure':
        # Fill NA values in the variable with the mean of the respective country over all available years
        oecd_df_rev[var] = oecd_df_rev.groupby('Country')[var].transform(lambda x: x.fillna(x.mean()))
        
        # Optional: If some countries have all missing values and you still get NaN after group-wise filling, 
        # you can fill those remaining NaNs with a global mean (across all countries).
        oecd_df_rev[var].fillna(oecd_df_rev[var].mean(), inplace=True)
    else:
        # Drop rows with missing values for rd_expenditure
        oecd_df_rev.dropna(subset=['rd_expenditure'], inplace=True)


# oecd_df_rev_4lag = oecd_df_rev[['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol']]
#oecd_df_rev_4lag = oecd_df_rev_4lag.dropna(subset=['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol'])

rd_expenditure_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'rd_expenditure_missing']]
oecd_df_rev = rd_expenditure_df_rev

#--------------------------------------------------------------------------------------------------------------------------

df_trends= pd.read_csv('/Users/atin/Nowcasting/data/GT/kw_only_approach/trends_data_resampled.csv')
df_trends_monthly = pd.read_csv('/Users/atin/Nowcasting/data/GT/kw_only_approach/trends_data.csv')

print(df_trends.columns)
print(df_trends_monthly.columns)

# Yearly G-Trend data

df_trends = df_trends[df_trends.columns.drop(list(df_trends.filter(regex='isPartial')))]

# Melt the DataFrame so that each row is a country-year pair
df_trends = df_trends.reset_index(drop=True).melt(id_vars='Year', var_name='Country_Keyword', value_name='Value')

# Extract the country code and keyword from the "Country_Keyword" column
df_trends[['Country', 'Keyword']] = df_trends['Country_Keyword'].str.split('_', 1, expand=True)

# Pivot the DataFrame so that each keyword becomes a column
df_trends = df_trends.pivot(index=['Year', 'Country'], columns='Keyword', values='Value')

print(df_trends.columns)

df_trends = df_trends.reset_index()
print(df_trends.columns)
print(f'Nb. Rows: {df_trends.shape[0]}')
print(df_trends.head(5))

# Monthly G-Trend data

df_trends_monthly = df_trends_monthly[df_trends_monthly.columns.drop(list(df_trends_monthly.filter(regex='isPartial')))]

# Melt the DataFrame so that each row is a country-year pair
df_trends_monthly = df_trends_monthly.reset_index(drop=True).melt(id_vars='date', var_name='Country_Keyword', value_name='Value')

# Extract the country code and keyword from the "Country_Keyword" column
df_trends_monthly[['Country', 'Keyword']] = df_trends_monthly['Country_Keyword'].str.split('_', 1, expand=True)

# Pivot the DataFrame so that each keyword becomes a column
df_trends_monthly = df_trends_monthly.pivot(index=['date', 'Country'], columns='Keyword', values='Value')

print(df_trends_monthly.columns)

df_trends_monthly = df_trends_monthly.reset_index()
print(df_trends_monthly.columns)
print(f'Nb. Rows: {df_trends_monthly.shape[0]}')
print(df_trends_monthly.head(5))

# Create a copy of the monthly data to avoid modifying the original data
df_trends_monthly_aggregated = df_trends_monthly.copy()

# Extract the year and month from the date
df_trends_monthly_aggregated['date'] = pd.to_datetime(df_trends_monthly_aggregated['date'])
df_trends_monthly_aggregated['Year'] = df_trends_monthly_aggregated['date'].dt.year
df_trends_monthly_aggregated['Month'] = df_trends_monthly_aggregated['date'].dt.month

print(df_trends_monthly_aggregated.columns)

# Convert 'date' column to datetime and sort by date
df_trends_monthly_aggregated['date'] = pd.to_datetime(df_trends_monthly_aggregated['date'])
df_trends_monthly_aggregated.sort_values(by='date', inplace=True)

# Loop over each keyword column and create new columns representing the mean of each variable 
# for the months of the current year up to the current month (not including the current month)
for column in df_trends_monthly.columns:
    if column not in ['date', 'Country', 'Year', 'Month']:
        df_trends_monthly_aggregated[f'{column}_mean_YTD'] = df_trends_monthly_aggregated.groupby(['Year', 'Country'])[column].transform(lambda x: x.expanding().mean().shift())

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/Baselines/df_trends_monthly_YTD.csv')

# Obtain a list of keyword columns, excluding those containing 'YTD'
keyword_columns = [col for col in df_trends.columns if col not in ['date', 'Country', 'Year', 'Month'] and 'YTD' not in col]

print(keyword_columns)

# Loop over each keyword column and merge
for keyword in keyword_columns:
    temp_df = df_trends[['Year', 'Country', keyword]].copy()
    temp_df.rename(columns={keyword: f'{keyword}_yearly_avg'}, inplace=True)
    
    df_trends_monthly_aggregated = df_trends_monthly_aggregated.merge(
        temp_df,
        on=['Year', 'Country']
    )

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/Baselines/df_trends_monthly_YTD_YR.csv')
print(df_trends_monthly_aggregated.columns)

# Generate a list of columns to be dropped
columns_to_drop = [column for column in df_trends_monthly_aggregated.columns if ('_mean_YTD' not in column) and ('_yearly_avg' not in column) and (column not in ['date', 'Country', 'Year'])]

# Drop the columns
df_trends_rev = df_trends_monthly_aggregated.drop(columns=columns_to_drop)


#%%--------------------------------------------------------------------------

# Merge the two DataFrames
merged_df = pd.merge(oecd_df_rev, df_trends_rev, on=['Country', 'Year'], how='left')
merged_df = merged_df.drop_duplicates()

print(f'Nb. Rows: {merged_df.shape[0]}')
print(merged_df.columns)
print(merged_df.head(5))

merged_df.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/Baselines/merged_df.csv')

#%%--------------------------------------------------------------------------

all_important_features = []

for col in merged_df.columns:
    if col != 'Country':  # Exclude the 'Country' column
        if merged_df[col].dtype == 'object':  # If the column is a string
            merged_df[col] = merged_df[col].str.replace(',', '')  # Remove the commas
            merged_df[col] = merged_df[col].replace('--', np.nan)  # Replace '--' with NaN
            merged_df[col] = merged_df[col].astype(float)  # Convert the column to float
            #merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

print(merged_df['Country'].unique())

#%%--------------------------------------------------------------------------
# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

# Create the lagged values for all variables (GTs only) first
for lag in range(1, max_lag+1):
    for col in rd_expenditure_df_rev.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
            # Drop rows with NA in 'rd_expenditure' and its lagged columns
            merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', 'rd_expenditure'])

# Get unique countries from df_trends
unique_countries = df_trends['Country'].unique()

# Limit merged_df to those countries
merged_df = merged_df[merged_df['Country'].isin(unique_countries)]

cols_dropna = [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]
cols_dropna += ['Year']

# Drop rows with NaN in specific columns
for col in cols_dropna:
    if col in merged_df.columns:
        merged_df.dropna(subset=[col], inplace=True)

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/Baselines/merged_df_test_RV01_batches_all_AggMGT_Evolution_0.csv")
#breakpoint()
merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]
merged_df = merged_df[(merged_df['Year'] >= 2004)]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or ('_lag' in col)]]
X_all.fillna(0, inplace=True)

X_all.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/Baselines/X_all_AggMGT_Evolution.csv")

#breakpoint()

#%%--------------------------------------------------------------------------

y = merged_df[['Country', 'Year', 'rd_expenditure']]

# First, split data into training + validation and test sets (e.g., 80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Initialize the list outside the loop so you can keep adding to it
relevant_columns = ['Year', 'Country']

# Loop over the range of lags
for lag in range(1, max_lag+1):
    relevant_columns += [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]    # relevant_columns += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year', 'Month']]

# After the loop, relevant_columns will have all the columns for lags from 1 to max_lag
#####################
# Apply the relevant_columns to the train and test datasets
X_train_lag = X_train[relevant_columns]
X_test_lag = X_test[relevant_columns]

y_train_lag = y_train[['rd_expenditure']]
y_test_lag = y_test[['rd_expenditure']]

y_train_lag = y_train_lag.dropna()
y_test_lag = y_test_lag.dropna()

#######################################################################
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = y_true.squeeze()  # Convert y_true to a Series or 1D array
    return 100 * (abs((y_true - y_pred) / y_true)).mean()


train = y_train[['rd_expenditure']]
test = y_test[['rd_expenditure']]

# 1. ARIMA
# Fit the ARIMA model on the training data
model = ARIMA(y_train_lag.values, order=(3,1,0))
model_fit = model.fit()

print(len(y_test_lag))
breakpoint()
# Forecast the future values, which is the length of the test set
#predictions_arima = model_fit.forecast(steps=len(y_test_lag))[0]
predictions_arima = model_fit.predict(start=len(y_train_lag), end=len(y_train_lag) + len(y_test_lag) - 1, typ='levels')


print(predictions_arima)
print(len(predictions_arima))
breakpoint()

# Calculate RMSE, MAE, MAPE for the ARIMA model
rmse_arima = np.sqrt(mean_squared_error(y_test_lag, predictions_arima))
mae_arima = mean_absolute_error(y_test_lag, predictions_arima)
mape_arima = mean_absolute_percentage_error(y_test_lag, predictions_arima)

breakpoint()

#######################################################################

# 2. Naive
#naive_forecast = X_test_lag['rd_expenditure_lag1'].values

def naive_forecast(series, lag=1):
    """Returns a naive forecast with the specified lag."""
    return series.shift(lag)

# Calculate naive forecast
y_train['naive_forecast'] = naive_forecast(y_train['rd_expenditure'])
y_test['naive_forecast'] = naive_forecast(y_test['rd_expenditure'])

y_test['naive_forecast'] = X_test_lag['rd_expenditure_lag3']


# Drop NA rows from training and test set
y_train.dropna(inplace=True)
y_test.dropna(inplace=True)

# Calculate RMSE, MAE, MAPE for the naive forecast
rmse_naive = np.sqrt(mean_squared_error(y_test['rd_expenditure'], y_test['naive_forecast']))
mae_naive = mean_absolute_error(y_test['rd_expenditure'], y_test['naive_forecast'])
mape_naive = mean_absolute_percentage_error(y_test['rd_expenditure'], y_test['naive_forecast'])
breakpoint()
#######################################################################

# 3. Moving Average
# moving_avg_forecast = X_test_lag[['rd_expenditure_lag1', 'rd_expenditure_lag2', 'rd_expenditure_lag3']].mean(axis=1).values

def moving_average_forecast(series, window=3):
    """Returns a moving average forecast with the specified window size."""
    return series.rolling(window=window).mean().shift(1)

# Calculate moving average forecast
y_train['moving_avg_forecast'] = moving_average_forecast(y_train['rd_expenditure'])
y_test['moving_avg_forecast'] = moving_average_forecast(y_test['rd_expenditure'])

y_test['moving_avg_forecast'] = X_test[['rd_expenditure_lag1', 'rd_expenditure_lag2', 'rd_expenditure_lag3']].mean(axis=1)


# Drop NA rows from training and test set
y_train.dropna(inplace=True)
y_test.dropna(inplace=True)

# Calculate RMSE, MAE, MAPE for the moving average forecast
rmse_moving_avg = np.sqrt(mean_squared_error(y_test['rd_expenditure'], y_test['moving_avg_forecast']))
mae_moving_avg = mean_absolute_error(y_test['rd_expenditure'], y_test['moving_avg_forecast'])
mape_moving_avg = mean_absolute_percentage_error(y_test['rd_expenditure'], y_test['moving_avg_forecast'])

#######################################################################

# Create a dataframe to store results
results_df = pd.DataFrame({
    'Model': ['Naive', 'ARIMA', 'Moving Average'],
    'RMSE': [rmse_naive, rmse_arima, rmse_moving_avg],
    'MAE': [mae_naive, mae_arima, mae_moving_avg],
    'MAPE': [mape_naive, mape_arima, mape_moving_avg]
})

# Save the results to a CSV file
results_df.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/Baselines/baseline_model_metrics.csv', index=False)



#####################################################
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    return 100 * (sum(abs((y_true - y_pred) / y_true)) / len(y_true))

# Given your data split
train = y_train_lag[['rd_expenditure']]
test = y_test_lag[['rd_expenditure']]

# 1. ARIMA
model_arima = ARIMA(train, order=(3,1,0))
model_arima_fit = model_arima.fit()
predictions_arima = model_arima_fit.forecast(steps=len(test))

# Evaluation for ARIMA
mae_arima = mean_absolute_error(test, predictions_arima)
rmse_arima = np.sqrt(mean_squared_error(test, predictions_arima))
mape_arima = mean_absolute_percentage_error(test.values.flatten(), predictions_arima)

# 2. Auto ARIMA
model_auto_arima = auto_arima(train, trace=True, suppress_warnings=True, seasonal=False)
predictions_auto_arima = model_auto_arima.predict(n_periods=len(test))

# Evaluation for Auto ARIMA
mae_auto_arima = mean_absolute_error(test, predictions_auto_arima)
rmse_auto_arima = np.sqrt(mean_squared_error(test, predictions_auto_arima))
mape_auto_arima = mean_absolute_percentage_error(test.values.flatten(), predictions_auto_arima)

# Save results to CSV
results = pd.DataFrame({
    'Model': ['ARIMA', 'Auto ARIMA'],
    'MAE': [mae_arima, mae_auto_arima],
    'RMSE': [rmse_arima, rmse_auto_arima],
    'MAPE': [mape_arima, mape_auto_arima]
})

results.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/Baselines/arima_evaluation_results.csv', index=False)

