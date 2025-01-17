import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm
import seaborn as sns
import random
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import Binarizer
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler
import shap
#---------------------------------------------------------------------------------------------------------------------------
## Pre-Processing Macroeconomic Variables + Target Variable (GERD in billion dollars)
#---------------------------------------------------------------------------------------------------------------------------

country_cd_df = pd.read_csv('/Nowcasting/data/country_code.csv')


# Read in and filter rd_expenditure_df to only include rows with sector of performance equal to Business enterprise + considering total fundings (millions-LOCAL CURRENCY)
rd_expenditure_df = pd.read_csv('/Nowcasting/data/GERD/DP_LIVE_08052023154811337.csv')
rd_expenditure_df = rd_expenditure_df.rename(columns={'Value': 'rd_expenditure'})  
rd_expenditure_df = rd_expenditure_df[rd_expenditure_df['MEASURE'] == 'MLN_USD'] #USD constant prices using 2015 base year 
print(rd_expenditure_df.columns)

rd_expenditure_df['rd_expenditure'] = rd_expenditure_df['rd_expenditure'] / 1000

# Read and filter n_patents
n_patents_df = pd.read_csv('/Nowcasting/data/OECD/PATS_IPC_11062023234902217.csv')
n_patents_df = n_patents_df[n_patents_df['IPC'] == 'TOTAL']
n_patents_df = n_patents_df[n_patents_df['KINDDATE'] == 'PRIORITY']
n_patents_df = n_patents_df[n_patents_df['KINDCOUNTRY'] == 'INVENTORS']
print(n_patents_df.columns)
n_patents_df = n_patents_df[['LOCATION','TIME','Value']]
n_patents_df = n_patents_df.rename(columns={'Value': 'n_patents'})
print(n_patents_df.columns)


# Read and filter IMF (various macro variables)
imf_df = pd.read_csv('/Nowcasting/data/IMF/WEOApr2023all.csv')

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

# Define a list of all variables in oecd_df_rev to check for missing values
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
        
        # Optional: If some countries have all missing values and you still get NaN after group-wise filling, you can fill those remaining NaNs with a global mean (across all countries).
        oecd_df_rev[var].fillna(oecd_df_rev[var].mean(), inplace=True)
    else:
        # Drop rows with missing values for rd_expenditure
        oecd_df_rev.dropna(subset=['rd_expenditure'], inplace=True)


oecd_df_rev_4lag = oecd_df_rev[['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol']]
#oecd_df_rev_4lag = oecd_df_rev_4lag.dropna(subset=['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol'])

rd_expenditure_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'rd_expenditure_missing']]

print(rd_expenditure_df_rev.head(5))

#--------------------------------------------------------------------------------------------------------------------------
## Pre-Processing Google Trends (GT)
#--------------------------------------------------------------------------------------------------------------------------

df_trends= pd.read_csv('/Nowcasting/data/GT/trends_data_by_topic_resampled_filtered.csv')
df_trends_monthly = pd.read_csv('/Nowcasting/data/GT/trends_data_by_topic_filtered.csv')

print(df_trends.columns)
print(df_trends_monthly.columns)

# Yearly G-Trend data

df_trends = df_trends[df_trends.columns.drop(list(df_trends.filter(regex='isPartial')))]

# Melt the DataFrame so that each row is a country-year pair
df_trends = df_trends.reset_index(drop=True).melt(id_vars='Year', var_name='Country_Keyword', value_name='Value')

# Extract the country code and keyword from the "Country_Keyword" column
df_trends[['Country', 'Keyword']] = df_trends['Country_Keyword'].str.split('_', 1, expand=True)

# Rename keywords that contain 'Country'
df_trends['Keyword'] = df_trends['Keyword'].apply(lambda x: x + '_topic' if 'Country' in x else x)

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

# Rename keywords that contain 'Country'
df_trends_monthly['Keyword'] = df_trends_monthly['Keyword'].apply(lambda x: x + '_topic' if 'Country' in x else x)


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

df_trends_monthly_aggregated.to_csv('/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/df_trends_monthly_YTD.csv')

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

df_trends_monthly_aggregated.to_csv('/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/df_trends_monthly_YTD_YR.csv')
print(df_trends_monthly_aggregated.columns)

# Generate a list of columns to be dropped
columns_to_drop = [column for column in df_trends_monthly_aggregated.columns if ('_mean_YTD' not in column) and ('_yearly_avg' not in column) and (column not in ['date', 'Country', 'Month', 'Year'])]

# Drop the columns
df_trends_rev = df_trends_monthly_aggregated.drop(columns=columns_to_drop)

#--------------------------------------------------------------------------------------------------------------------------
# Merge the input DataFrames
#--------------------------------------------------------------------------------------------------------------------------

merged_df = pd.merge(oecd_df_rev, df_trends_rev, on=['Country', 'Year'], how='left')
merged_df = merged_df.drop_duplicates()

print(f'Nb. Rows: {merged_df.shape[0]}')
print(merged_df.columns)
print(merged_df.head(5))

merged_df.to_csv('/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/merged_df.csv')

for col in merged_df.columns:
    if col != 'Country':  # Exclude the 'Country' column
        if merged_df[col].dtype == 'object':  # If the column is a string
            merged_df[col] = merged_df[col].str.replace(',', '')  # Remove the commas
            merged_df[col] = merged_df[col].replace('--', np.nan)  # Replace '--' with NaN
            merged_df[col] = merged_df[col].astype(float)  # Convert the column to float
            #merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

print(merged_df['Country'].unique())


# Ensure the DataFrame is sorted by Country, Year, and then by Month
merged_df.sort_values(by=['Country', 'Year', 'Month'], inplace=True)

all_important_features = []

#--------------------------------------------------------------------------------------------------------------------------
## Pre-Processing on Merged df
#--------------------------------------------------------------------------------------------------------------------------

# Define a range of lags to consider
max_lag = 3  # considering up to 3 years of lagged values

# Create the lagged values for all variables (GTs only) first
for lag in range(1, max_lag+1):
    for col in df_trends.columns:
        if col not in ['Country', 'Year', 'Month']:
            # Create a new lagged column
            lagged_col_name = f'{col}_yearly_avg_lag{lag}'
            merged_df[lagged_col_name] = merged_df.groupby('Country')[f'{col}_yearly_avg'].shift(12 * lag)
            # Fill NA values in the new lagged column with zero
            merged_df[lagged_col_name].fillna(0, inplace=True)


# Apply linear interpolation only where 'rd_expenditure_missing' indicates missing data
def conditional_interpolate(group):
    # Only interpolate where 'rd_expenditure_missing' equals 1
    mask = group['rd_expenditure_missing'] == 1
    group.loc[mask, 'rd_expenditure'] = group.loc[mask, 'rd_expenditure'].interpolate()
    return group

# Apply the conditional interpolation to the 'rd_expenditure' column
merged_df = merged_df.groupby('Country').apply(conditional_interpolate)

# Now create the lagged values as before
for lag in range(1, max_lag+1):
    for col in rd_expenditure_df_rev.columns:
        if col not in ['Country', 'Year', 'Month']:
            # Create lagged column
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(12 * lag)

# Fill NaN in columns with keyword 'mean_YTD' 
for col in merged_df.columns[merged_df.columns.str.contains('mean_YTD')]:
    merged_df[col].fillna(0, inplace=True)

# Loop over the range of lags
for lag in range(1, max_lag+1):
    # Create lagged values for macro variables
    for col in oecd_df_rev_4lag.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(12 * lag)
            # Fill NA values in the new lagged column with country-wise mean
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[f'{col}_lag{lag}'].transform(lambda x: x.fillna(x.mean()))


# Get unique countries from df_trends
unique_countries = df_trends['Country'].unique()

# Limit merged_df to those countries
merged_df = merged_df[merged_df['Country'].isin(unique_countries)]

# merged_df[lagged_col_name] = merged_df[lagged_col_name].fillna(0, inplace=True)

# Define groups of columns
cols_fillna_zero = [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year']]
cols_fillna_zero += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']]
# cols_fillna_zero += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]

cols_dropna = [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]
#cols_dropna += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]
cols_dropna += ['Year']

# Fill NaN with 0 in specific columns
for col in cols_fillna_zero:
    if col in merged_df.columns:
        merged_df[col].fillna(0, inplace=True)

cols_to_fill = [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]

# Fill NaN with the mean of each country over all available years for each column
for col in cols_to_fill:
    if col in merged_df.columns:
        merged_df[col] = merged_df.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))

merged_df.to_csv(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/merged_df_test_RV01_batches_all_AggMGT_Evolution_0.csv")

merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]
merged_df = merged_df[(merged_df['Year'] >= 2004)]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or (col == 'Month')  or ('_lag' in col) or ('_mean_YTD' in col)]]
X_all.fillna(0, inplace=True)

X_all.to_csv(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/X_all_AggMGT_Evolution.csv")

y = merged_df[['Country', 'Year', 'Month', 'rd_expenditure']]

#breakpoint()

#--------------------------------------------------------------------------------------------------------------------------
## Set-up Train, Validation and Test Sets
#--------------------------------------------------------------------------------------------------------------------------

# Get unique countries
countries = X_all['Country'].unique()

# Initialize empty DataFrames for the splits
X_train_combined = pd.DataFrame()
X_val_combined = pd.DataFrame()
X_test_combined = pd.DataFrame()
y_train_combined = pd.Series()
y_val_combined = pd.Series()
y_test_combined = pd.Series()

# Split the data for each country
for country in countries:
    X_country = X_all[X_all['Country'] == country]
    y_country = y[X_all['Country'] == country]

    # Split into training + validation and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X_country, y_country, test_size=0.2, random_state=42)
    # Split training + validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Combine the splits
    X_train_combined = pd.concat([X_train_combined, X_train])
    X_val_combined = pd.concat([X_val_combined, X_val])
    X_test_combined = pd.concat([X_test_combined, X_test])
    y_train_combined = pd.concat([y_train_combined, y_train])
    y_val_combined = pd.concat([y_val_combined, y_val])
    y_test_combined = pd.concat([y_test_combined, y_test])

# The final combined datasets
X_train = X_train_combined
X_val = X_val_combined
X_test = X_test_combined
y_train = y_train_combined
y_val = y_val_combined
y_test = y_test_combined

#####################
#####################

# Initialize the list outside the loop to keep adding to it
relevant_columns = ['Year', 'Month','Country']
# relevant_columns = ['Month','Country']
#relevant_columns += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]

# Loop over the range of lags
for lag in range(1, max_lag+1):
    
    # Append the lagged columns for the current lag to the list
    #relevant_columns += [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]
    relevant_columns += [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year', 'Month']]
    #relevant_columns += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year', 'Month']]

# After the loop, relevant_columns will have all the columns for lags from 1 to max_lag

# Apply the relevant_columns to the train and test datasets
X_train_lag = X_train[relevant_columns]
X_val_lag = X_val[relevant_columns]    
X_test_lag = X_test[relevant_columns]

y_train_lag = y_train[['rd_expenditure']]
y_val_lag = y_val[['rd_expenditure']]
y_test_lag = y_test[['rd_expenditure']]

y_train_lag_mlpyt = y_train[['rd_expenditure']]
y_val_lag_mlpyt = y_val[['rd_expenditure']]
y_test_lag_mlpyt = y_test[['rd_expenditure']]

###

# Drop duplicates in X datasets and keep the indices
X_train_lag = X_train_lag.drop_duplicates()
X_val_lag = X_val_lag.drop_duplicates()
X_test_lag = X_test_lag.drop_duplicates()

# Use the indices from X datasets to filter y datasets
y_train_lag = y_train_lag.loc[X_train_lag.index]
y_val_lag = y_val_lag.loc[X_val_lag.index]
y_test_lag = y_test_lag.loc[X_test_lag.index]

y_train_lag_mlpyt = y_train_lag_mlpyt.loc[X_train_lag.index]
y_val_lag_mlpyt = y_val_lag_mlpyt.loc[X_val_lag.index]
y_test_lag_mlpyt = y_test_lag_mlpyt.loc[X_test_lag.index]

# Extract the original country column before any transformation
countries_train_mlpyt = X_train_lag['Country'].values
countries_val_mlpyt = X_val_lag['Country'].values
countries_test_mlpyt = X_test_lag['Country'].values

###

#####################
#####################

# Convert 'Country' column to dummies
X_train_lag = pd.get_dummies(X_train_lag, columns=['Country'])
X_val_lag = pd.get_dummies(X_val_lag, columns=['Country'])    
X_test_lag = pd.get_dummies(X_test_lag, columns=['Country'])

X_train_lag_mlpyt = X_train_lag
X_val_lag_mlpyt = X_val_lag
X_test_lag_mlpyt = X_test_lag

# Save column names before any operation that changes X_train_lag to a numpy array
original_columns = X_train_lag.columns.tolist()

# Adding Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_lag)
X_val_poly = poly.transform(X_val_lag)
X_test_poly = poly.transform(X_test_lag)

X_train_lag = X_train_poly
X_val_lag = X_val_poly
X_test_lag = X_test_poly

#####################
#####################

# Preprocessing for MLP
from sklearn.preprocessing import StandardScaler

# Subset the data using the relevant_columns and then drop duplicates.
X_train = X_train[relevant_columns].drop_duplicates()
X_val = X_val[relevant_columns].drop_duplicates()
X_test = X_test[relevant_columns].drop_duplicates()

# Filter y datasets based on the indices of the X datasets to maintain alignment
y_train = y_train.loc[X_train.index][['rd_expenditure']]
y_val = y_val.loc[X_val.index][['rd_expenditure']]
y_test = y_test.loc[X_test.index][['rd_expenditure']]

# Store 'Year' and 'Month' in separate variables
year_train = X_train['Year']
month_train = X_train['Month']
year_val = X_val['Year']
month_val = X_val['Month']
year_test = X_test['Year']
month_test = X_test['Month']

# One-hot encode 'Month' for the training data
month_encoded_train = pd.get_dummies(X_train['Month'], prefix='Month')

# Align the validation and test data with the columns of the training data
month_encoded_val = pd.get_dummies(X_val['Month'], prefix='Month').reindex(columns=month_encoded_train.columns, fill_value=0)
month_encoded_test = pd.get_dummies(X_test['Month'], prefix='Month').reindex(columns=month_encoded_train.columns, fill_value=0)

# # Concatenate the one-hot encoded columns with the original data
# X_train = pd.concat([X_train.drop('Month', axis=1), month_encoded_train], axis=1)
# X_val = pd.concat([X_val.drop('Month', axis=1), month_encoded_val], axis=1)
# X_test = pd.concat([X_test.drop('Month', axis=1), month_encoded_test], axis=1)

# Concatenate the one-hot encoded columns with the original data (without dropping the original 'Month' column)
X_train = pd.concat([X_train, month_encoded_train], axis=1)
X_val = pd.concat([X_val, month_encoded_val], axis=1)
X_test = pd.concat([X_test, month_encoded_test], axis=1)

# 2. Encode the 'Country' column.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_train['country_encoded'] = le.fit_transform(X_train['Country'])
X_val['country_encoded'] = le.transform(X_val['Country'])
X_test['country_encoded'] = le.transform(X_test['Country'])

num_countries = X_train['Country'].nunique()

# 3. Drop the original 'Country' column.
X_train = X_train.drop(columns=['Country'])
X_val = X_val.drop(columns=['Country'])
X_test = X_test.drop(columns=['Country'])

print(le.classes_)
print(X_train['country_encoded'].unique())


X_train.to_csv(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/X_train_OHC_Month_Cat_Country.csv")


#--------------------------------------------------------------------------------------------------------------------------
## Standardization of Input for Neural Netwoks Model
#--------------------------------------------------------------------------------------------------------------------------

X_train_org = X_train
X_test_org = X_test
X_val_org = X_val

# Step 1: Separate the features to be standardized and the country_encoded column
X_train_continuous = X_train.drop(columns=['country_encoded', 'Year', 'Month'])
X_val_continuous = X_val.drop(columns=['country_encoded', 'Year', 'Month'])
X_test_continuous = X_test.drop(columns=['country_encoded', 'Year', 'Month'])

# Step 2: Apply the standardization
X_scaler = StandardScaler()

# Fit the scaler using the training data and transform the training data
# first fits the scaler with the provided data (i.e., calculates the mean and standard deviation) and then immediately transforms (standardizes) the data.
# fit only on the training data!!
X_train_continuous_standardized = X_scaler.fit_transform(X_train_continuous)

# Transform the validation and test data using the same scaler
X_val_continuous_standardized = X_scaler.transform(X_val_continuous)
X_test_continuous_standardized = X_scaler.transform(X_test_continuous)

# Step 3: Merge them back together (now including Year and Month)
X_train_standardized = np.hstack((X_train_continuous_standardized, 
                                  X_train[['country_encoded', 'Year', 'Month']].values))
X_val_standardized = np.hstack((X_val_continuous_standardized, 
                                X_val[['country_encoded', 'Year', 'Month']].values))
X_test_standardized = np.hstack((X_test_continuous_standardized, 
                                 X_test[['country_encoded', 'Year', 'Month']].values))

# Step 4: Convert the numpy arrays back to DataFrames
X_train_cols = X_train_continuous.columns.tolist() + ['country_encoded', 'Year', 'Month']
X_val_cols = X_val_continuous.columns.tolist() + ['country_encoded', 'Year', 'Month']
X_test_cols = X_test_continuous.columns.tolist() + ['country_encoded', 'Year', 'Month']

X_train_standardized = pd.DataFrame(X_train_standardized, columns=X_train_cols)
X_val_standardized = pd.DataFrame(X_val_standardized, columns=X_val_cols)
X_test_standardized = pd.DataFrame(X_test_standardized, columns=X_test_cols)

X_train = X_train_standardized
X_val = X_val_standardized
X_test = X_test_standardized

X_train.to_csv(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/X_train_std.csv")

#breakpoint()

#--------------------------------------------------------------------------------------------------------------------------
## Set-up Neural Netwoks Model
#--------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn #module import: giving access to NN functionalities in PyTorch
import torch.optim as optim


##################### Tensors Creation #####################

# Define the columns to be excluded from the model input
excluded_columns = ['country_encoded', 'Year', 'Month']

# Select only the columns that are not in the excluded list
feature_columns = [col for col in X_train.columns if col not in excluded_columns]

# For the training set
X_train_tensor = torch.FloatTensor(X_train[feature_columns].values)
#X_train_tensor = torch.FloatTensor(X_train.drop(columns=['country_encoded']).values)
country_indices_train = torch.LongTensor(X_train['country_encoded'].values)
y_train_tensor = torch.FloatTensor(y_train.values)

# For the validation set
X_val_tensor = torch.FloatTensor(X_val[feature_columns].values)
#X_val_tensor = torch.FloatTensor(X_val.drop(columns=['country_encoded']).values)
country_indices_val = torch.LongTensor(X_val['country_encoded'].values)
y_val_tensor = torch.FloatTensor(y_val.values)

# For the test set
X_test_tensor = torch.FloatTensor(X_test[feature_columns].values)
#X_test_tensor = torch.FloatTensor(X_test.drop(columns=['country_encoded']).values)
country_indices_test = torch.LongTensor(X_test['country_encoded'].values)
y_test_tensor = torch.FloatTensor(y_test.values)

#breakpoint()

##################### Hyper-parameters #####################

embedding_dim = 2

# input_dim = X_train.shape[1]
base_input_dim = X_train.drop(columns=['country_encoded', 'Year', 'Month']).shape[1]
input_dim = base_input_dim + embedding_dim # +1 for bias, +embedding_dim for country embeddings

print(base_input_dim)
print(input_dim)

hidden1_dim = 200
hidden2_dim = 20
hidden3_dim = 20
output_dim = 1

num_epochs = 50000
batch_size = 64
patience = 10000
# For many datasets, starting with a batch size of 32 or 64 and training for a fixed number of epochs (like 50 or 100) is common.

##################### Loss Function #####################

# Loss function
criterion = nn.MSELoss() # Using Mean Squared Error for regression tasks

def elastic_net_penalty(model, delta=0.5):
    l1_reg = torch.tensor(0.).to(X_train_tensor.device)
    l2_reg = torch.tensor(0.).to(X_train_tensor.device)
    
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
        l2_reg += torch.norm(param, 2)
        
    return 0.5 * (delta * l1_reg + (1 - delta) * l2_reg)

##################### FC NN Model #####################

# Define the Feedforward Neural Network
class MLP(nn.Module): # MLP class from nn.Module/base class for all NN
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_countries, embedding_dim):
        super(MLP, self).__init__()
        
        # Embedding for countries
        # embedding_dim: size of the vector space in which the countries will be embedded        
        self.country_embedding = nn.Embedding(num_embeddings=num_countries, embedding_dim=embedding_dim)
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden1_dim, bias=True)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim, bias=True)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim, bias=True)    
        self.fc4 = nn.Linear(hidden3_dim, output_dim, bias=True)  # Output layer

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(hidden1_dim)
        self.bn2 = nn.BatchNorm1d(hidden2_dim)
        self.bn3 = nn.BatchNorm1d(hidden3_dim)       
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x, country_indices):
        country_embeds = self.country_embedding(country_indices)
        
        # Add bias term (a tensor of ones) to the input tensor
        bias_term = torch.ones((x.size(0), 1), device=x.device)        

        # Concatenate the country embeddings and bias term with the input
        x = torch.cat([x, country_embeds], dim=1)        
        #x = torch.cat([x, country_embeds, bias_term], dim=1)
        
        # Applying BatchNorm after ReLU for the hidden layers
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.bn2(self.relu(self.fc2(x)))
        x = self.bn3(self.relu(self.fc3(x)))       
        x = self.fc4(x)  # No activation at the output layer for regression tasks

        return x

##################### Ensemble NN Model #####################

# Define the Ensemble Neural Network
class Ensemble:
    def __init__(self, model_class, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, size_ensemble, num_countries, embedding_dim):
        # Using different seeds for each model's initialization
        self.models = []
        for _ in range(size_ensemble):
            torch.manual_seed(_)  # Setting a different seed for each iteration
            self.models.append(model_class(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_countries, embedding_dim))

    def rmse(self, outputs, labels):
        """Calculate Root Mean Squared Error"""
        mse = torch.mean((outputs - labels) ** 2)
        return torch.sqrt(mse)

    def mae(self, outputs, labels):
        """Calculate Mean Absolute Error"""
        return torch.mean(torch.abs(outputs - labels))

    def mape(self, outputs, labels):
        """Calculate Mean Absolute Percentage Error"""
        return torch.mean(torch.abs((labels - outputs) / labels)) * 100

    def r_squared(self, outputs, labels):
        """Calculate R-squared (Coefficient of Determination)"""
        ss_total = torch.sum((labels - torch.mean(labels)) ** 2)
        ss_res = torch.sum((labels - outputs) ** 2)
        r2 = 1 - ss_res / ss_total
        return r2        

    def train(self, X_train, y_train, X_val, y_val, country_indices_train, country_indices_val, criterion, optimizer_class, num_epochs=num_epochs, batch_size=batch_size, patience=patience):

        model_rmse_train = []
        model_rmse_val = []
        model_mae_train = []
        model_mae_val = []
        model_mape_train = []
        model_mape_val = []

        #self.train_losses = []  # List to store training losses for all models
        #self.val_losses = []    # List to store validation losses for all models

        self.train_losses = [[] for _ in range(size_ensemble)]
        self.val_losses = [[] for _ in range(size_ensemble)]

        for idx, model in enumerate(self.models):
            model.train()  # Switch to training mode
            
            #optimizer = optimizer_class(model.parameters(), lr=0.1)
            optimizer = optimizer_class(model.parameters(), lr=0.01)

            # Since we want to change the learning rate after 20 iterations, 
            # we set milestones=[20] and gamma=0.1 to change lr to lr*gamma.
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)

            best_val_loss = float('inf')
            no_improve_count = 0

            # Initialize the path for the best model for this iteration (model idx in the ensemble)
            best_model_path = f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/best_model_{idx}.pt"

            for epoch in range(num_epochs):
                #model.train()  # Switch to training mode

                train_loss_epoch = 0.0  # Collect loss over the epoch
                num_batches = 0

                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_country_indices = country_indices_train[i:i+batch_size]  # Extract the corresponding country indices
                    batch_y = y_train[i:i+batch_size]

                    outputs = model(batch_X, batch_country_indices)
                    loss = criterion(outputs, batch_y)

                    # penalty = elastic_net_penalty(model, delta=0.5)
                    # loss += penalty 

                    # Store the batch-wise training loss
                    self.train_losses[idx].append(loss)

                    optimizer.zero_grad()
                    loss.backward() # gradient of the loss with respect to each parameter
                    optimizer.step() # adjust our model's parameters (weights/biases) in the direction that reduces the loss.

                    train_loss_epoch += loss.item()
                    num_batches += 1

                    # Update the learning rate
                    scheduler.step()

                # Average training loss for the epoch
                avg_train_loss = train_loss_epoch / num_batches
                #self.train_losses.append(avg_train_loss)

                # Check early stopping after every epoch
                with torch.no_grad():
                    model.eval()  # Switch to evaluation mode
                    #val_outputs = model(X_val)
                    val_outputs = model(X_val, country_indices_val)
                    val_loss = criterion(val_outputs, y_val)

                    # Store the batch-wise validation loss
                    #self.val_losses.append(val_loss.item())
                    self.val_losses[idx].append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        torch.save(model.state_dict(), best_model_path)  # Save the best model state for this model                        
                    else:
                        no_improve_count += 1

                    if no_improve_count == patience:
                        print(f"Model {idx+1} early stopped after {epoch} epochs")
                        break  # Early stop

                print(f"Epoch [{epoch+1}/{num_epochs}], Model {idx+1}, Training Loss: {train_loss_epoch:.4f}")
            
            # After all epochs (or early stopping) for a model, load best state
            model.load_state_dict(torch.load(best_model_path))  # Load the best model state

            # After all epochs (or early stopping) for a model, compute and store the RMSE for that model
            with torch.no_grad():
                model.eval()  # Switch to evaluation mode
                
                # Pass country indices along with X_train and X_val
                train_rmse = self.rmse(model(X_train, country_indices_train), y_train)
                val_rmse = self.rmse(model(X_val, country_indices_val), y_val)
                
                train_mae = self.mae(model(X_train, country_indices_train), y_train)
                val_mae = self.mae(model(X_val, country_indices_val), y_val)
                
                train_mape = self.mape(model(X_train, country_indices_train), y_train)
                val_mape = self.mape(model(X_val, country_indices_val), y_val)
                
                model_rmse_train.append(train_rmse.item())
                model_rmse_val.append(val_rmse.item())
                model_mae_train.append(train_mae.item())
                model_mae_val.append(val_mae.item())
                model_mape_train.append(train_mape.item())
                model_mape_val.append(val_mape.item())

            # Compute ensemble metrics for validation data only
            # For ensemble prediction, assuming the predict method handles country indices internally.
            ensemble_val_predictions = self.predict(X_val, country_indices_val)
            ensemble_val_rmse = self.rmse(ensemble_val_predictions, y_val).item()
            ensemble_val_mae = self.mae(ensemble_val_predictions, y_val).item()
            ensemble_val_mape = self.mape(ensemble_val_predictions, y_val).item()


        return model_rmse_train, model_rmse_val, model_mae_train, model_mae_val, model_mape_train, model_mape_val, ensemble_val_rmse, ensemble_val_mae, ensemble_val_mape

    def predict(self, X, country_indices):
        with torch.no_grad():
            # Setting models to evaluation mode for prediction
            for model in self.models:
                model.eval()
            
            predictions = [model(X, country_indices) for model in self.models]
            mean_predictions = torch.mean(torch.stack(predictions), dim=0)
            
            return mean_predictions

    def get_parameters(self):
        """Return the number of trainable parameters for each model in the ensemble."""
        return [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in self.models]

#####################

def r_squared(outputs, labels):
    ss_res = torch.sum((labels - outputs) ** 2)
    ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def adjusted_r_squared(outputs, labels, input_dim):
    # Calculate the residual sum of squares
    ss_res = torch.sum((labels - outputs) ** 2)
    # Calculate the total sum of squares
    ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
    # Calculate the unadjusted R-squared
    r2 = 1 - ss_res / ss_tot
    # Calculate the number of observations
    n = labels.size(0)
    # Calculate the adjusted R-squared
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - input_dim - 1)
    return adj_r2

#--------------------------------------------------------------------------------------------------------------------------
## Set-up Ensemble of Neural Netwoks Models
#--------------------------------------------------------------------------------------------------------------------------

# Create an ensemble
size_ensemble = 10
# ensemble = Ensemble(MLP, input_dim, hidden1_dim, hidden2_dim, output_dim, size_ensemble=size_ensemble)
ensemble = Ensemble(MLP, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, size_ensemble=size_ensemble, num_countries=num_countries, embedding_dim=embedding_dim)

# Over time, as the model trains, countries with similar behaviors or characteristics might get vectors that are closer together in this 10-dimensional space. This is the model's way of saying "these countries behave similarly in terms of the target variable I'm predicting."

# Get the number of trainable parameters for each model
parameters_list = ensemble.get_parameters()

# For simplicity, let's assume all models in the ensemble have the same input dimension (number of features)
input_dimensions_list = [input_dim] * size_ensemble

# Prepare the data for the CSV
data = {
    'Model': [f"Model_{i+1}" for i in range(size_ensemble)],
    'Input_Dimensions': input_dimensions_list,
    'Trainable_Parameters': parameters_list
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = "/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/ensemble_info.csv"
df.to_csv(csv_path, index=False)

print(f"Ensemble information saved to {csv_path}")

#--------------------------------------------------------------------------------------------------------------------------
## Train the ensemble
#--------------------------------------------------------------------------------------------------------------------------

# Now these will hold all the different metrics
train_rmse_list, val_rmse_list, train_mae_list, val_mae_list, train_mape_list, val_mape_list, ensemble_val_rmse, ensemble_val_mae, ensemble_val_mape = ensemble.train(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, country_indices_train, country_indices_val, criterion=criterion, optimizer_class=optim.AdamW)

print(len(train_rmse_list))
print(len(val_rmse_list))

print(len([f"Model_{i+1}" for i in range(size_ensemble)] + ["Ensemble"]))
print(len(train_rmse_list + [None]))
print(len(val_rmse_list + [ensemble_val_rmse]))

#--------------------------------------------------------------------------------------------------------------------------
## Predictions from the ensemble
#--------------------------------------------------------------------------------------------------------------------------

# Predictions for training, validation, and test datasets
y_train_pred = ensemble.predict(X_train_tensor, country_indices_train).numpy()  # convert tensor to numpy if it's not
y_val_pred = ensemble.predict(X_val_tensor, country_indices_val).numpy()  # convert tensor to numpy if it's not
y_test_pred = ensemble.predict(X_test_tensor, country_indices_test).numpy()  # convert tensor to numpy if it's not

#--------------------------------------------------------------------------------------------------------------------------
## Calculate R-squared values
#--------------------------------------------------------------------------------------------------------------------------

adjusted_r_squared_train = adjusted_r_squared(y_train_pred, y_train_tensor, input_dim)
adjusted_r_squared_test = adjusted_r_squared(y_test_pred, y_test_tensor, input_dim)

# Create a DataFrame to save R-squared values
adjusted_r_squared_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'adjusted_r_squared': [adjusted_r_squared_train.item(), adjusted_r_squared_test.item()]
})

adjusted_r_squared_csv_path = "/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/adjusted_r_squared_values.csv"
adjusted_r_squared_df.to_csv(adjusted_r_squared_csv_path, index=False)

print(f"R-squared values saved to {adjusted_r_squared_csv_path}")

#####################

# Constructing the DataFrame for train, validation, and test datasets
df_train = pd.DataFrame({
    'Year': X_train['Year'],
    'Month': X_train['Month'], 
    'Country': countries_train_mlpyt,
    'True_Values': y_train_tensor.numpy().flatten(),
    'Predicted_Values': y_train_pred.flatten(),
    'Type': 'Train'
})

df_val = pd.DataFrame({
    'Year': X_val['Year'],
    'Month': X_val['Month'], 
    'Country': countries_val_mlpyt,
    'True_Values': y_val_tensor.numpy().flatten(),
    'Predicted_Values': y_val_pred.flatten(),
    'Type': 'Validation'
})

df_test = pd.DataFrame({
    'Year': X_test['Year'],
    'Month': X_test['Month'], 
    'Country': countries_test_mlpyt,
    'True_Values': y_test_tensor.numpy().flatten(),
    'Predicted_Values': y_test_pred.flatten(),
    'Type': 'Test'
})


df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

csv_path = "/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/df_combined_pred_vs_true.csv"
df_combined.to_csv(csv_path, index=False)

# Unique countries in the dataset
countries = df_combined['Country'].unique()
    
#--------------------------------------------------------------------------------------------------------------------------
## ScatterPlots of True vs. Predicted Values
#--------------------------------------------------------------------------------------------------------------------------

def plot_scatter_per_country(df, title_prefix):
    # Prepare the combined scatter plot
    plt.figure(figsize=(12, 8))

    countries = df['Country'].unique()
    for country in countries:
        df_country = df[df['Country'] == country]
        
        # Individual plot for each country
        plt.figure(figsize=(10, 6))
        plt.scatter(df_country['True_Values'], df_country['Predicted_Values'], alpha=0.6)
        plt.plot(np.sort(df_country['True_Values']), np.sort(df_country['True_Values']), color='red')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f"{title_prefix} Data for {country}: True vs Predicted Values")
        plt.grid(True)
        plt.savefig(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/{title_prefix}_Predictions_vs_TrueValues_{country}.pdf")        
        plt.close()  # Close the individual plot

        # Add data to the combined plot (this should be outside of the individual figure context)
        plt.figure(1)  # Switch back to the first figure
        sns.scatterplot(x=df_country['True_Values'], y=df_country['Predicted_Values'], label=country)

    # Finalize the combined plot
    plt.plot(np.sort(df['True_Values']), np.sort(df['True_Values']), color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f"{title_prefix} Combined Data: True vs Predicted Values")
    plt.grid(True)
    plt.legend(title="Country")
    plt.savefig(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/{title_prefix}_Predictions_vs_TrueValues_Combined.pdf")
    plt.close()  # Close the individual plot

# Apply the function for each dataset
plot_scatter_per_country(df_train, "Training")
plot_scatter_per_country(df_val, "Validation")
plot_scatter_per_country(df_test, "Test")

#breakpoint()

def plot_scatter_per_country_log(df, title_prefix):
    # Prepare the combined scatter plot
    plt.figure(figsize=(12, 8))

    countries = df['Country'].unique()
    for country in countries:
        df_country = df[df['Country'] == country]
        
        # Individual plot for each country in log scale
        plt.figure(figsize=(10, 6))
        plt.scatter(df_country['True_Values'], df_country['Predicted_Values'], alpha=0.6)
        plt.plot(np.sort(df_country['True_Values']), np.sort(df_country['True_Values']), color='red')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f"{title_prefix} Data for {country}: True vs Predicted Values")
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/{title_prefix}_Predictions_vs_TrueValues_{country}_log.pdf")        
        plt.close()  # Close the individual plot

        # Add data to the combined plot (in log scale)
        plt.figure(1)  # Switch back to the first figure
        sns.scatterplot(x=df_country['True_Values'], y=df_country['Predicted_Values'], label=country)

    # Finalize the combined plot
    plt.plot(np.sort(df['True_Values']), np.sort(df['True_Values']), color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f"{title_prefix} Combined Data: True vs Predicted Values")
    plt.grid(True)
    plt.yscale('log')
    plt.legend(title="Country")
    plt.savefig(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/{title_prefix}_Predictions_vs_TrueValues_Combined_log.pdf")
    plt.close()  # Close the individual plot

# Plot scatter plots for each country in train, validation, and test data
plot_scatter_per_country_log(df_train, "Training")
plot_scatter_per_country_log(df_val, "Validation")
plot_scatter_per_country_log(df_test, "Test")

#####################
#####################

# Predicting RMSE for each model on the test set (out-of-sample)
test_rmse_list = []
test_mae_list = []
test_mape_list = []

# Lists to hold the R-squared values for each model
test_r2_list = []

for model in ensemble.models:

    # model_predictions = model(X_test_tensor)
    model_predictions = model(X_test_tensor, country_indices_test)

    model_rmse = ensemble.rmse(model_predictions, y_test_tensor)
    model_mae = ensemble.mae(model_predictions, y_test_tensor)
    model_mape = ensemble.mape(model_predictions, y_test_tensor)

    test_rmse_list.append(model_rmse.item())
    test_mae_list.append(model_mae.item())
    test_mape_list.append(model_mape.item())

    # Calculate R-squared for the current model
    model_r2 = adjusted_r_squared(model_predictions, y_test_tensor, input_dim)
    test_r2_list.append(model_r2.item())  # Assuming adjusted_r_squared returns a tensor


# For the ensemble
ensemble_predictions = ensemble.predict(X_test_tensor, country_indices_test)

ensemble_test_rmse = ensemble.rmse(ensemble_predictions, y_test_tensor).item()
ensemble_test_mae = ensemble.mae(ensemble_predictions, y_test_tensor).item()
ensemble_test_mape = ensemble.mape(ensemble_predictions, y_test_tensor).item()

# Calculate R-squared for the ensemble
ensemble_r2 = adjusted_r_squared(ensemble_predictions, y_test_tensor, input_dim).item()

print(len(train_rmse_list + [None]))
print(len(val_rmse_list + [ensemble_val_rmse]))
print(len(test_rmse_list + [ensemble_test_rmse]))


# Combine RMSE, MSE, and MAPE values
metrics_df = pd.DataFrame({
    "Model": [f"Model_{i+1}" for i in range(size_ensemble)] + ["Ensemble"],
    "Train_RMSE": train_rmse_list + [None],  # since we don't have an ensemble value for training RMSE
    "Validation_RMSE": val_rmse_list + [ensemble_val_rmse],
    "Test_RMSE": test_rmse_list + [ensemble_test_rmse],
    "Train_MAE": train_mae_list + [None],
    "Validation_MAE": val_mae_list + [ensemble_val_mae],
    "Test_MAE": test_mae_list + [ensemble_test_mae],
    "Train_MAPE": train_mape_list + [None],
    "Validation_MAPE": val_mape_list + [ensemble_val_mape],
    "Test_MAPE": test_mape_list + [ensemble_test_mape]
})

# Add the R-squared values
metrics_df["Test_R2"] = test_r2_list + [ensemble_r2]

# Save back to CSV
metrics_df.to_csv("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/metrics_df_minibatchGD.csv", index=False)

#--------------------------------------------------------------------------------------------------------------------------
## Saving Training Loss & Validation Loss
#--------------------------------------------------------------------------------------------------------------------------

# Save Training Losses for each model in the ensemble
for idx, model_losses in enumerate(ensemble.train_losses):
    # Convert the tensor values to numpy and then to a pandas DataFrame
    df_train_loss = pd.DataFrame({'Iteration': list(range(len(model_losses))),
                                  'Training_Loss': [item.detach().numpy() for item in model_losses]})
    
    # Save to CSV
    df_train_loss.to_csv(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/MLP_MinibatchGD_Training_Loss_Iterations_Model_{idx+1}.csv", index=False)

# Save Validation Losses for each model in the ensemble
for idx, model_losses in enumerate(ensemble.val_losses):
    # Convert the tensor values to numpy and then to a pandas DataFrame
    df_val_loss = pd.DataFrame({'Iteration': list(range(len(model_losses))),
                                'Validation_Loss': [item.detach().numpy() for item in model_losses]})
    
    # Save to CSV
    df_val_loss.to_csv(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/MLP_MinibatchGD_Validation_Loss_Iterations_Model_{idx+1}.csv", index=False)

#--------------------------------------------------------------------------------------------------------------------------
## Plotting Training Loss & Validation Loss
#--------------------------------------------------------------------------------------------------------------------------

# Plotting Training Losses for each model in the ensemble
for idx, model_losses in enumerate(ensemble.train_losses):
    plt.figure(figsize=(12,6))
    #plt.plot(model_losses, label=f"Training Loss - Model {idx+1}", alpha=0.8)
    plt.plot([item.detach().numpy() for item in model_losses], label=f"Training Loss - Model {idx+1}", alpha=0.8)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Evolution of Training Loss over Iterations - Model {idx+1}")
    plt.legend()

    # Set y-axis to log scale
    plt.xscale('log')

    # Set y-axis to log scale
    plt.yscale('log')

    plt.savefig(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/MLP_MinibatchGD_Training_Loss_Iterations_Model_{idx+1}.pdf")
    #plt.show()

# Plotting Validation Losses for each model in the ensemble
for idx, model_losses in enumerate(ensemble.val_losses):
    plt.figure(figsize=(12,6))
    #plt.plot(model_losses, label=f"Validation Loss - Model {idx+1}", alpha=0.8, linestyle='--')
    plt.plot([item.detach().numpy() for item in model_losses], label=f"Validation Loss - Model {idx+1}", alpha=0.8, linestyle='--')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Evolution of Validation Loss over Iterations - Model {idx+1}")
    plt.legend()

    # Set y-axis to log scale
    plt.xscale('log')

    # Set y-axis to log scale
    plt.yscale('log')

    plt.savefig(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/MLP_MinibatchGD_Validation_Loss_Iterations_Model_{idx+1}.pdf")
    #plt.show()

#####################

country_metrics_data = []

unique_countries = torch.unique(country_indices_test)

for country in unique_countries:
    country_mask = country_indices_test == country
    X_country = X_test_tensor[country_mask]
    y_country = y_test_tensor[country_mask]
    
    country_metrics = {
        'Country': country.item(),
        'Models': [],
        'RMSE': [],
        'MAE': [],
        'MAPE': []
    }
    
    for idx, model in enumerate(ensemble.models):
        model_predictions = model(X_country, country_indices_test[country_mask])
        
        model_rmse = ensemble.rmse(model_predictions, y_country).item()
        model_mae = ensemble.mae(model_predictions, y_country).item()
        model_mape = ensemble.mape(model_predictions, y_country).item()
        
        country_metrics['Models'].append(f"Model_{idx+1}")
        country_metrics['RMSE'].append(model_rmse)
        country_metrics['MAE'].append(model_mae)
        country_metrics['MAPE'].append(model_mape)

    # For the ensemble on this country subset
    ensemble_predictions = ensemble.predict(X_country, country_indices_test[country_mask])
    
    ensemble_rmse = ensemble.rmse(ensemble_predictions, y_country).item()
    ensemble_mae = ensemble.mae(ensemble_predictions, y_country).item()
    ensemble_mape = ensemble.mape(ensemble_predictions, y_country).item()
    
    country_metrics['Models'].append('Ensemble')
    country_metrics['RMSE'].append(ensemble_rmse)
    country_metrics['MAE'].append(ensemble_mae)
    country_metrics['MAPE'].append(ensemble_mape)
    
    country_metrics_data.append(country_metrics)


# Convert to DataFrame
df = pd.DataFrame(country_metrics_data)
# Save to CSV
df.to_csv("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/country_metrics.csv", index=False)

# Visualization
plt.figure(figsize=(12, 8))

# Assuming each country has the same models
for model in df['Models'][0]:
    country_rmse = df.apply(lambda row: row['RMSE'][row['Models'].index(model)], axis=1)
    plt.plot(df['Country'], country_rmse, label=model, marker='o')

    plt.xlabel('Country')
    plt.ylabel('RMSE')
    plt.title('Model Performance by Country')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/country_metrics_vis_{model}.pdf")
    #plt.show()

#--------------------------------------------------------------------------------------------------------------------------
## Compute the Elasticities ot of the Model
#--------------------------------------------------------------------------------------------------------------------------

def compute_elasticity_torch(model, X_sample, country_indices_sample, feature_index):
    model.eval()  # Switch to evaluation mode
    X_sample = torch.Tensor(X_sample)  # Convert to PyTorch tensor if not already
    country_indices_sample = torch.LongTensor(country_indices_sample)  # Convert to PyTorch tensor if not already
    
    perturb_amount = 0.01  # 1% increase
    X_perturbed = X_sample.clone()
    X_perturbed[:, feature_index] *= (1 + perturb_amount)
    
    with torch.no_grad():  # We won't be backpropagating, so disable gradient calculations
        y_pred_original = model(X_sample, country_indices_sample)
        y_pred_perturbed = model(X_perturbed, country_indices_sample)
    
    elasticity = ((y_pred_perturbed - y_pred_original) / y_pred_original) / perturb_amount
    return elasticity  # return tensor without computing the mean

# Initialize the dictionary to save the elasticities
all_elasticities = {}

# Create an empty DataFrame to store elasticity values along with lag, feature name, and country
df_elasticities = pd.DataFrame(columns=['Country', 'Lag', 'Feature Name', 'Elasticity'])

# After populating df_elasticities with individual elasticities for each sample...
avg_elasticities = df_elasticities.groupby('Country').Elasticity.mean().reset_index()

#####################


# Having a function like compute_elasticity_torch(model, X_sample, country_indices, feature_index)

# Names of the features
feature_names = list(X_train.drop(columns=['country_encoded', 'Year', 'Month']).columns)
#feature_names = [col for col in X_train.columns if col not in ['country_encoded', 'Year', 'Month']]

# Placeholder list to store elasticity results
all_elasticities = []

print(X_train['country_encoded'].dtype)
#breakpoint()

X_train['country_encoded'] = X_train['country_encoded'].astype(int)
X_val['country_encoded'] = X_val['country_encoded'].astype(int)
X_test['country_encoded'] = X_test['country_encoded'].astype(int)

# Decode the country_encoded values to actual country names
decoded_country_names = le.inverse_transform(X_train['country_encoded'].values)

# Loop through each model in the ensemble
for model in ensemble.models:
    model_elasticities = []  # Placeholder for each model's elasticities
    
    # Loop through each sample in the training data
    for i in range(X_train_tensor.size(0)):
        sample_elasticities = []
        year = X_train.iloc[i]['Year']  # Extract the year for the current sample

        for j, feature in enumerate(feature_names):
            if feature != 'Year':

                elasticity_tensor = compute_elasticity_torch(model, X_train_tensor[i:i+1], country_indices_train[i:i+1], j)
                
                # Extract the scalar value from the tensor
                elasticity_value = elasticity_tensor.item()
                sample_elasticities.append(elasticity_value)

            
        # Fetch the decoded country name
        country_name = decoded_country_names[i]
        
        for feature, elasticity in zip(feature_names, sample_elasticities):
            model_elasticities.append({
                'Model': f"Model_{ensemble.models.index(model)+1}",
                'Country': country_name,
                'Year': year,  # Include Year for record-keeping                
                'Feature': feature,
                'Elasticity': elasticity
            })
    
    all_elasticities.extend(model_elasticities)

# Convert the results into a DataFrame
df_elasticities_raw = pd.DataFrame(all_elasticities)

# Compute average elasticity for each feature per country (across all models)
df_elasticities_avg_wyr = df_elasticities_raw.groupby(['Country', 'Year', 'Feature']).agg({'Elasticity': 'mean'}).reset_index()

df_elasticities_avg_woyr = df_elasticities_raw.groupby(['Country', 'Feature']).agg({'Elasticity': 'mean'}).reset_index()

# Save to CSV if needed
csv_path_elasticities_avg_wyr = "/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/elasticities_avg_ensemble_wyr.csv"
df_elasticities_avg_wyr.to_csv(csv_path_elasticities_avg_wyr, index=False)

# Save to CSV if needed
csv_path_elasticities_avg_woyr = "/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/elasticities_avg_ensemble_woyr.csv"
df_elasticities_avg_woyr.to_csv(csv_path_elasticities_avg_woyr, index=False)

csv_path_elasticities_raw = "/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/elasticities_all_models.csv"
df_elasticities_raw.to_csv(csv_path_elasticities_raw, index=False)


#--------------------------------------------------------------------------------------------------------------------------
## Compute the SHAP Values of the Model
#--------------------------------------------------------------------------------------------------------------------------

def f(X_combined_numpy):
    # Assuming the last column represents 'country_indices'
    X_features_numpy = X_combined_numpy[:, :-1]
    country_indices_numpy = X_combined_numpy[:, -1]

    # Convert these arrays to PyTorch tensors
    X_features_tensor = torch.from_numpy(X_features_numpy).float()
    country_indices_tensor = torch.from_numpy(country_indices_numpy).long()

    # Get predictions from the ensemble model
    predictions = ensemble.predict(X_features_tensor, country_indices_tensor).detach().numpy()

    return predictions

# Combine feature data and country indices into a single NumPy array for SHAP
X_train_combined_np = np.hstack([X_train_tensor.numpy(), country_indices_train.unsqueeze(1).numpy()])
X_val_combined_np = np.hstack([X_val_tensor.numpy(), country_indices_val.unsqueeze(1).numpy()])
X_test_combined_np = np.hstack([X_test_tensor.numpy(), country_indices_test.unsqueeze(1).numpy()])

# Assuming the last column in the combined arrays is the country index
X_train_features_np = X_train_combined_np[:, :-1]
X_val_features_np = X_val_combined_np[:, :-1]
X_test_features_np = X_test_combined_np[:, :-1]

# Creating Pandas DataFrames from the NumPy arrays
X_train_df = pd.DataFrame(X_train_features_np, columns=feature_names)
X_val_df = pd.DataFrame(X_val_features_np, columns=feature_names)
X_test_df = pd.DataFrame(X_test_features_np, columns=feature_names)

K = 40  # number of clusters
X_train_summary = shap.kmeans(X_train_combined_np, K)

# Create the SHAP KernelExplainer using the summarized background data
explainer = shap.KernelExplainer(f, X_train_summary)

# SHAP values computation
shap_values_train = explainer.shap_values(X_train_combined_np)
shap_values_val = explainer.shap_values(X_val_combined_np)
shap_values_test = explainer.shap_values(X_test_combined_np)

# Extract the first array from each list (assuming a single-output model)
shap_values_train_array = shap_values_train if not isinstance(shap_values_train, list) else shap_values_train[0]
shap_values_val_array = shap_values_val if not isinstance(shap_values_val, list) else shap_values_val[0]
shap_values_test_array = shap_values_test if not isinstance(shap_values_test, list) else shap_values_test[0]

# Squeeze the SHAP values to ensure they are 2-dimensional
shap_values_train_array = shap_values_train_array.squeeze()
shap_values_val_array = shap_values_val_array.squeeze()
shap_values_test_array = shap_values_test_array.squeeze()

# Adjust the SHAP values if they include an extra column
shap_values_train_adj = shap_values_train_array[:, :-1] if shap_values_train_array.shape[1] == X_train_df.shape[1] + 1 else shap_values_train_array
shap_values_val_adj = shap_values_val_array[:, :-1] if shap_values_val_array.shape[1] == X_val_df.shape[1] + 1 else shap_values_val_array
shap_values_test_adj = shap_values_test_array[:, :-1] if shap_values_test_array.shape[1] == X_test_df.shape[1] + 1 else shap_values_test_array


# #SHAP Summary plot
plt.figure()
shap.summary_plot(shap_values_train_adj, X_train_df, show=False)
plt.savefig("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_summary_train.pdf", bbox_inches='tight')

plt.figure()
shap.summary_plot(shap_values_train_adj, X_train_df, plot_type="bar", show=False)
plt.savefig("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_bar_train_bar.pdf", bbox_inches='tight')

plt.figure()
shap.summary_plot(shap_values_val_adj, X_val_df, show=False)
plt.savefig("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_summary_val.pdf", bbox_inches='tight')

plt.figure()
shap.summary_plot(shap_values_val_adj, X_val_df, plot_type="bar", show=False)
plt.savefig("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_bar_val_bar.pdf", bbox_inches='tight')

plt.figure()
shap.summary_plot(shap_values_test_adj, X_test_df, show=False)
plt.savefig("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_summary_test.pdf", bbox_inches='tight')

plt.figure()
shap.summary_plot(shap_values_test_adj, X_test_df, plot_type="bar", show=False)
plt.savefig("/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_bar_test_bar.pdf", bbox_inches='tight')

df_shap_train = pd.DataFrame(shap_values_train_adj, columns=feature_names)
df_shap_train.to_csv('/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_values_train.csv', index=False)

df_shap_val = pd.DataFrame(shap_values_val_adj, columns=feature_names)
df_shap_val.to_csv('/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_values_val.csv', index=False)

df_shap_test = pd.DataFrame(shap_values_test_adj, columns=feature_names)
df_shap_test.to_csv('/Nowcasting/nn_mlp_nowcasting_model/results/LagRD/shap_values_test.csv', index=False)

