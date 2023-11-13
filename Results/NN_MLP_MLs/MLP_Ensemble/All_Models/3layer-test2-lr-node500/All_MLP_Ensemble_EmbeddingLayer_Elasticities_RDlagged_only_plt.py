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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/df_trends_monthly_YTD.csv')

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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/df_trends_monthly_YTD_YR.csv')
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

merged_df.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/merged_df.csv')

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
    # for col in df_trends.columns:
    #     if col not in ['Country', 'Year', 'Month']:
    #         # Create a new lagged column
    #         lagged_col_name = f'{col}_yearly_avg_lag{lag}'
    #         merged_df[lagged_col_name] = merged_df.groupby('Country')[f'{col}_yearly_avg'].shift(lag)
    #         # Fill NA values in the new lagged column with zero
    #         merged_df[lagged_col_name].fillna(0, inplace=True)

    for col in rd_expenditure_df_rev.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
            # Drop rows with NA in 'rd_expenditure' and its lagged columns
            merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', 'rd_expenditure'])

# Fill NaN in columns with keyword 'mean_YTD' 
# for col in merged_df.columns[merged_df.columns.str.contains('mean_YTD')]:
#     merged_df[col].fillna(0, inplace=True)

# Loop over the range of lags
# for lag in range(1, max_lag+1):
#     # Create lagged values for macro variables
#     for col in oecd_df_rev_4lag.columns:
#         if col not in ['Country', 'Year']:
#             merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
#             # Fill NA values in the new lagged column with country-wise mean
#             merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[f'{col}_lag{lag}'].transform(lambda x: x.fillna(x.mean()))


# Get unique countries from df_trends
unique_countries = df_trends['Country'].unique()

# Limit merged_df to those countries
merged_df = merged_df[merged_df['Country'].isin(unique_countries)]

# merged_df[lagged_col_name] = merged_df[lagged_col_name].fillna(0, inplace=True)

# Define groups of columns
# cols_fillna_zero = [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year']]
# cols_fillna_zero += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']]
# cols_fillna_zero += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]

cols_dropna = [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]
# cols_dropna += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]
cols_dropna += ['Year']

# Fill NaN with 0 in specific columns
# for col in cols_fillna_zero:
#     if col in merged_df.columns:
#         merged_df[col].fillna(0, inplace=True)

# Drop rows with NaN in specific columns
for col in cols_dropna:
    if col in merged_df.columns:
        merged_df.dropna(subset=[col], inplace=True)

# Define the columns you wish to fill NaN values for
# cols_to_fill = [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year']]
# cols_to_fill += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']]
# cols_to_fill = [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]

# Fill NaN with the mean of each country over all available years for each column
# for col in cols_to_fill:
#     if col in merged_df.columns:
#         merged_df[col] = merged_df.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/merged_df_test_RV01_batches_all_AggMGT_Evolution_0.csv")
#breakpoint()
merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]
merged_df = merged_df[(merged_df['Year'] >= 2004)]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or ('_lag' in col)]]
X_all.fillna(0, inplace=True)

X_all.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/X_all_AggMGT_Evolution.csv")

#breakpoint()

#%%--------------------------------------------------------------------------

y = merged_df[['Country', 'Year', 'rd_expenditure']]

# First, split data into training + validation and test sets (e.g., 80-20 split)
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Then, split the training + validation data into training and validation sets (e.g., 80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


# Initialize the list outside the loop so you can keep adding to it
relevant_columns = ['Year', 'Country']

# Loop over the range of lags
for lag in range(1, max_lag+1):
    
    # Append the lagged columns for the current lag to the list
    # relevant_columns += [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]
    # relevant_columns += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]
    relevant_columns += [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]
    # relevant_columns += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year', 'Month']]

# After the loop, relevant_columns will have all the columns for lags from 1 to max_lag


#####################
#####################
# Preprocessing for XGBoost, RF, SVR, EN and MLP-Scikitlearn
# Apply the relevant_columns to the train and test datasets
X_train_lag = X_train[relevant_columns]
X_val_lag = X_val[relevant_columns]    
X_test_lag = X_test[relevant_columns]

y_train_lag = y_train[['rd_expenditure']]
y_val_lag = y_val[['rd_expenditure']]
y_test_lag = y_test[['rd_expenditure']]

#####################

y_train_lag_mlpyt = y_train[['rd_expenditure']]
y_val_lag_mlpyt = y_val[['rd_expenditure']]
y_test_lag_mlpyt = y_test[['rd_expenditure']]

y_train_lag_mlpsci = y_train[['rd_expenditure']]
y_val_lag_mlpsci = y_val[['rd_expenditure']]
y_test_lag_mlpsci = y_test[['rd_expenditure']]

y_train_lag_xgb = y_train[['rd_expenditure']]
y_val_lag_xgb = y_val[['rd_expenditure']]
y_test_lag_xgb = y_test[['rd_expenditure']]

y_train_lag_rf = y_train[['rd_expenditure']]
y_val_lag_rf = y_val[['rd_expenditure']]
y_test_lag_rf = y_test[['rd_expenditure']]

y_train_lag_svr = y_train[['rd_expenditure']]
y_val_lag_svr = y_val[['rd_expenditure']]
y_test_lag_svr = y_test[['rd_expenditure']]

y_train_lag_en = y_train[['rd_expenditure']]
y_val_lag_en = y_val[['rd_expenditure']]
y_test_lag_en = y_test[['rd_expenditure']]

#####################

# Extract the original country column before any transformation
countries_train_mlpyt = X_train['Country'].values
countries_val_mlpyt = X_val['Country'].values
countries_test_mlpyt = X_test['Country'].values

countries_train_mlpsci = X_train['Country'].values
countries_val_mlpsci = X_val['Country'].values
countries_test_mlpsci = X_test['Country'].values

countries_train_xgb = X_train['Country'].values
countries_val_xgb = X_val['Country'].values
countries_test_xgb = X_test['Country'].values

countries_train_rf = X_train['Country'].values
countries_val_rf = X_val['Country'].values
countries_test_rf = X_test['Country'].values

countries_train_svr = X_train['Country'].values
countries_val_svr = X_val['Country'].values
countries_test_svr = X_test['Country'].values

countries_train_en = X_train['Country'].values
countries_val_en = X_val['Country'].values
countries_test_en = X_test['Country'].values

#####################

# Convert 'Country' column to dummies
X_train_lag = pd.get_dummies(X_train_lag, columns=['Country'])
X_val_lag = pd.get_dummies(X_val_lag, columns=['Country'])    
X_test_lag = pd.get_dummies(X_test_lag, columns=['Country'])

X_train_lag_mlpyt = X_train_lag
X_val_lag_mlpyt = X_val_lag
X_test_lag_mlpyt = X_test_lag

X_train_lag_mlpsci = X_train_lag
X_val_lag_mlpsci = X_val_lag
X_test_lag_mlpsci = X_test_lag

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

X_train_lag_xgb = X_train_lag
X_val_lag_xgb = X_val_lag
X_test_lag_xgb = X_test_lag

X_train_lag_rf = X_train_lag
X_val_lag_rf = X_val_lag
X_test_lag_rf = X_test_lag

X_train_lag_svr = X_train_lag
X_val_lag_svr = X_val_lag
X_test_lag_svr = X_test_lag

X_train_lag_en = X_train_lag
X_val_lag_en = X_val_lag
X_test_lag_en = X_test_lag

#####################

# Preprocessing for MLP
# Use the accumulated relevant_columns to subset your train and test datasets
#####

from sklearn.preprocessing import StandardScaler

# 1. Subset the data using the relevant_columns.
X_train = X_train[relevant_columns]
X_val = X_val[relevant_columns]
X_test = X_test[relevant_columns]
y_train = y_train[['rd_expenditure']]
y_val = y_val[['rd_expenditure']]
y_test = y_test[['rd_expenditure']]

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


######### Standardization of input #########
from sklearn.preprocessing import StandardScaler

# X_train_org = X_train
# X_test_org = X_test
# X_val_org = X_val

# # Step 1: Separate the features to be standardized and the country_encoded column
# X_train_continuous = X_train.drop(columns=['country_encoded'])
# X_val_continuous = X_val.drop(columns=['country_encoded'])
# X_test_continuous = X_test.drop(columns=['country_encoded'])

# # Step 2: Apply the standardization
# X_scaler = StandardScaler()

# # Fit the scaler using the training data and transform the training data
# # first fits the scaler with the provided data (i.e., calculates the mean and standard deviation) and then immediately transforms (standardizes) the data.
# # fit only on the training data!!
# X_train_continuous_standardized = X_scaler.fit_transform(X_train_continuous)

# # Transform the validation and test data using the same scaler
# X_val_continuous_standardized = X_scaler.transform(X_val_continuous)
# X_test_continuous_standardized = X_scaler.transform(X_test_continuous)


# # Step 3: Merge them back together
# X_train_standardized = np.hstack((X_train_continuous_standardized, X_train['country_encoded'].values.reshape(-1, 1)))
# X_val_standardized = np.hstack((X_val_continuous_standardized, X_val['country_encoded'].values.reshape(-1, 1)))
# X_test_standardized = np.hstack((X_test_continuous_standardized, X_test['country_encoded'].values.reshape(-1, 1)))

# # Convert the numpy arrays back to DataFrames (optional but can be useful)
# X_train_standardized = pd.DataFrame(X_train_standardized, columns=X_train.columns)
# X_val_standardized = pd.DataFrame(X_val_standardized, columns=X_val.columns)
# X_test_standardized = pd.DataFrame(X_test_standardized, columns=X_test.columns)

# X_train = X_train_standardized
# X_val = X_val_standardized
# X_test = X_test_standardized

# X_train.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/X_train_std.csv")
# #breakpoint()

#####################

#MLP with BatchNorm + Ensemble
import torch
import torch.nn as nn #module import: giving access to NN functionalities in PyTorch
import torch.optim as optim

#####################

# Hyper-parameters
embedding_dim = 2

# input_dim = X_train.shape[1]  # Assuming X_train is your training dataset
base_input_dim = X_train.drop(columns=['country_encoded']).shape[1]
input_dim = base_input_dim + embedding_dim

print(base_input_dim)
print(input_dim)

#breakpoint()

hidden1_dim = 500
hidden2_dim = 50
hidden3_dim = 5
output_dim = 1  # for regression tasks, it's usually 1

# Loss function
criterion = nn.MSELoss() # Using Mean Squared Error for regression tasks

#####################


num_epochs = 100000000
batch_size = 256
patience = 10000
# For many datasets, starting with a batch size of 32 or 64 and training for a fixed number of epochs (like 50 or 100) is common.

#####################

# For the training set
X_train_tensor = torch.FloatTensor(X_train.drop(columns=['country_encoded']).values)
country_indices_train = torch.LongTensor(X_train['country_encoded'].values)
y_train_tensor = torch.FloatTensor(y_train.values)

# For the validation set
X_val_tensor = torch.FloatTensor(X_val.drop(columns=['country_encoded']).values)
country_indices_val = torch.LongTensor(X_val['country_encoded'].values)
y_val_tensor = torch.FloatTensor(y_val.values)

# For the test set
X_test_tensor = torch.FloatTensor(X_test.drop(columns=['country_encoded']).values)
country_indices_test = torch.LongTensor(X_test['country_encoded'].values)
y_test_tensor = torch.FloatTensor(y_test.values)

#####################

# Define the Feedforward Neural Network
class MLP(nn.Module): # MLP class from nn.Module/base class for all NN
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_countries, embedding_dim):
        super(MLP, self).__init__()
        
        # Embedding for countries
        # embedding_dim: size of the vector space in which the countries will be embedded        
        self.country_embedding = nn.Embedding(num_embeddings=num_countries, embedding_dim=embedding_dim)
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, output_dim)  # This becomes the final output layer (1 usually for regression tasks)
        
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(hidden1_dim)
        self.bn2 = nn.BatchNorm1d(hidden2_dim)
        self.bn3 = nn.BatchNorm1d(hidden3_dim)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x, country_indices):
        country_embeds = self.country_embedding(country_indices)
        
        # Concatenate the country embeddings with the input
        x = torch.cat([x, country_embeds], dim=1)
        
        # Applying BatchNorm after ReLU for the hidden layers
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.bn2(self.relu(self.fc2(x)))
        x = self.bn3(self.relu(self.fc3(x)))  # Pass through the third hidden layer
        x = self.fc4(x)  # No activation at the output layer for regression tasks
        # x = self.relu(self.fc4(x))  # Apply ReLU activation after the final layer

        return x

#####################

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
            
            optimizer = optimizer_class(model.parameters(), lr=0.1)

            # Since we want to change the learning rate after 20 iterations, 
            # we set milestones=[20] and gamma=0.1 to change lr to lr*gamma.
            scheduler = MultiStepLR(optimizer, milestones=[30000], gamma=0.1)

            best_val_loss = float('inf')
            no_improve_count = 0

            # Initialize the path for the best model for this iteration (model idx in the ensemble)
            best_model_path = f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/best_model_{idx}.pt"

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

# Create an ensemble
size_ensemble = 2
# ensemble = Ensemble(MLP, input_dim, hidden1_dim, hidden2_dim, output_dim, size_ensemble=size_ensemble)
ensemble = Ensemble(MLP, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, size_ensemble=size_ensemble, num_countries=num_countries, embedding_dim=embedding_dim)

# Over time, as the model trains, countries with similar behaviors or characteristics might get vectors that are closer together in this 10-dimensional space. This is the model's way of saying "these countries behave similarly in terms of the target variable I'm predicting."

#####################

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
csv_path = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ensemble_info.csv"  # Modify this to your desired path
df.to_csv(csv_path, index=False)

print(f"Ensemble information saved to {csv_path}")

#####################

# Train the ensemble
# Now these will hold all the different metrics
train_rmse_list, val_rmse_list, train_mae_list, val_mae_list, train_mape_list, val_mape_list, ensemble_val_rmse, ensemble_val_mae, ensemble_val_mape = ensemble.train(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, country_indices_train, country_indices_val, criterion=criterion, optimizer_class=optim.AdamW)

print(len(train_rmse_list))
print(len(val_rmse_list))

print(len([f"Model_{i+1}" for i in range(size_ensemble)] + ["Ensemble"]))
print(len(train_rmse_list + [None]))
print(len(val_rmse_list + [ensemble_val_rmse]))

#####################

# Predicting RMSE for each model on the test set (out-of-sample)
test_rmse_list = []
test_mae_list = []
test_mape_list = []

for model in ensemble.models:

    # model_predictions = model(X_test_tensor)
    model_predictions = model(X_test_tensor, country_indices_test)

    model_rmse = ensemble.rmse(model_predictions, y_test_tensor)
    model_mae = ensemble.mae(model_predictions, y_test_tensor)
    model_mape = ensemble.mape(model_predictions, y_test_tensor)

    test_rmse_list.append(model_rmse.item())
    test_mae_list.append(model_mae.item())
    test_mape_list.append(model_mape.item())

# For the ensemble
ensemble_predictions = ensemble.predict(X_test_tensor, country_indices_test)

ensemble_test_rmse = ensemble.rmse(ensemble_predictions, y_test_tensor).item()
ensemble_test_mae = ensemble.mae(ensemble_predictions, y_test_tensor).item()
ensemble_test_mape = ensemble.mape(ensemble_predictions, y_test_tensor).item()

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

# Save back to CSV
metrics_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/metrics_df_minibatchGD.csv", index=False)

#######################

# error_data = {
#     'Month': [],
#     'Percentage_Error': []
# }

# for model in ensemble.models:
#     model_predictions = model(X_test_tensor, country_indices_test).detach().numpy().flatten()
#     actual_values = y_test_tensor.numpy().flatten()
    
#     # Calculate individual percentage errors
#     percentage_errors = 100 * (actual_values - model_predictions) / actual_values

#     # Assuming you have a variable 'months' in your X_test that represents the month for each data point
#     error_data['Month'].extend(X_test['Month'].values)
#     error_data['Percentage_Error'].extend(percentage_errors)

# # Convert the data into a DataFrame for easier manipulation and visualization
# error_df = pd.DataFrame(error_data)

# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Month', y='Percentage_Error', data=error_df)
# plt.title("Distribution of Percentage Errors by Month")
# plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/Test_MGTs_MAPE_Month.png")
# #plt.show()

######################


# Save Training Losses for each model in the ensemble
for idx, model_losses in enumerate(ensemble.train_losses):
    # Convert the tensor values to numpy and then to a pandas DataFrame
    df_train_loss = pd.DataFrame({'Iteration': list(range(len(model_losses))),
                                  'Training_Loss': [item.detach().numpy() for item in model_losses]})
    
    # Save to CSV
    df_train_loss.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLP_MinibatchGD_Training_Loss_Iterations_Model_{idx+1}.csv", index=False)

# Save Validation Losses for each model in the ensemble
for idx, model_losses in enumerate(ensemble.val_losses):
    # Convert the tensor values to numpy and then to a pandas DataFrame
    df_val_loss = pd.DataFrame({'Iteration': list(range(len(model_losses))),
                                'Validation_Loss': [item.detach().numpy() for item in model_losses]})
    
    # Save to CSV
    df_val_loss.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLP_MinibatchGD_Validation_Loss_Iterations_Model_{idx+1}.csv", index=False)

# Training Loss
# Plotting Training Losses for each model in the ensemble
for idx, model_losses in enumerate(ensemble.train_losses):
    plt.figure(figsize=(12,6))
    #plt.plot(model_losses, label=f"Training Loss - Model {idx+1}", alpha=0.8)
    plt.plot([item.detach().numpy() for item in model_losses], label=f"Training Loss - Model {idx+1}", alpha=0.8)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Evolution of Training Loss over Iterations - Model {idx+1}")
    plt.legend()
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLP_MinibatchGD_Training_Loss_Iterations_Model_{idx+1}.png")
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
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLP_MinibatchGD_Validation_Loss_Iterations_Model_{idx+1}.png")
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

# Now, 'country_metrics_data' will hold metrics for each model and the ensemble, broken down by country.

import pandas as pd
import matplotlib.pyplot as plt

# Convert to DataFrame
df = pd.DataFrame(country_metrics_data)

# Save to CSV
df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/country_metrics.csv", index=False)

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
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/country_metrics_vis.png")
#plt.show()

###

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

# After you've populated df_elasticities with individual elasticities for each sample...
avg_elasticities = df_elasticities.groupby('Country').Elasticity.mean().reset_index()

#####################

# Assuming you have a function like compute_elasticity_torch(model, X_sample, country_indices, feature_index)

# Names of the features
feature_names = list(X_train.drop(columns=['country_encoded']).columns)

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
        for j, feature in enumerate(feature_names):
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
                'Feature': feature,
                'Elasticity': elasticity
            })
    
    all_elasticities.extend(model_elasticities)

# Convert the results into a DataFrame
df_elasticities_raw = pd.DataFrame(all_elasticities)

# Compute average elasticity for each feature per country (across all models)
df_elasticities_avg = df_elasticities_raw.groupby(['Country', 'Feature']).agg({'Elasticity': 'mean'}).reset_index()

# Save to CSV if needed
csv_path_elasticities = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/elasticities_avg_ensemble.csv"
df_elasticities_avg.to_csv(csv_path_elasticities, index=False)

# You can also save the raw elasticities (per model) to another CSV if you want:
csv_path_elasticities_raw = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/elasticities_all_models.csv"
df_elasticities_raw.to_csv(csv_path_elasticities_raw, index=False)

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################


# Neural Network setup using MLPRegressor
network = MLPRegressor(hidden_layer_sizes=(200, 20),
                       solver="adam",
                       activation="relu",
                       learning_rate_init=0.001,
                       tol=1e-4,
                       max_iter=100000,  # Maximum number of iterations. #For stochastic solvers (‘sgd’, ‘Adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
                       warm_start=True,  # To retain state across loop iterations
                       early_stopping=True,
                       validation_fraction=0.2,  # Fraction of training data to set aside as validation set for early stopping
                       n_iter_no_change=200,  # Number of iterations with no improvement to wait before stopping the solver
                       # alpha=0.0001,  # L2 penalty (regularization term) parameter. Adjust this value as needed.                       
                       random_state=0)

# Initialize the model
model = network
#n_estimators: 100 (lots of observations) to 1000 (few observations)
#The number of trees in our ensemble. Equivalent to the number of boosting rounds.

model_name = type(model).__name__

# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

rmse_dict = {}
mae_dict = {}
mape_dict = {}

rmse_dict_test = {}
mae_dict_test = {}
mape_dict_test = {}

all_important_features = []

# Define a function to compute elasticities for XGBoost
def compute_elasticity(model, X_sample, feature_index):
    X_perturbed = X_sample.copy()
    perturb_amount = 0.01  # 1% increase
    X_perturbed[:, feature_index] *= (1 + perturb_amount)
    
    y_pred_original = model.predict(X_sample)
    y_pred_perturbed = model.predict(X_perturbed)
    
    elasticity = ((y_pred_perturbed - y_pred_original) / y_pred_original) / perturb_amount
    return np.mean(elasticity)

# Initialize the dictionary to save the elasticities
all_elasticities = {}

# Create an empty DataFrame to store elasticity values along with lag and feature name
df_elasticities = pd.DataFrame(columns=['Lag', 'Feature Name', 'Elasticity'])

#%%--------------------------------------------------------------------------

y_train_lag = y_train_lag_mlpsci
y_val_lag = y_val_lag_mlpsci
y_test_lag = y_test_lag_mlpsci

X_train_lag = X_train_lag_mlpsci
X_val_lag = X_val_lag_mlpsci
X_test_lag = X_test_lag_mlpsci

countries_train = countries_train_mlpsci
countries_val = countries_val_mlpsci
countries_test = countries_test_mlpsci

# Combine them
X_combined = np.vstack((X_train_lag, X_val_lag))
y_combined = np.concatenate((y_train_lag, y_val_lag))

y_combined_df = pd.DataFrame(y_combined, columns=['rd_expenditure'])

countries_combined = np.concatenate((countries_train, countries_val))

X_train_lag = X_combined
y_train_lag = y_combined_df
countries_train = countries_combined


# Now pass the data to the model
model_fit = network.fit(X_train_lag, y_train_lag)  # The model trains on the data using early stopping internally

#####################

lag = 3  # For example, consider up to 3 years of lagged values


# Get the model parameters
model_params = model_fit.get_params()
print(f'Training Set - Lag: {lag}', model_params)

# Calculate the accuracy score on the training set (R2?)
train_score = model_fit.score(X_train_lag, y_train_lag)

print("Number of columns: ", X_train_lag.shape[1])

#####################

# Make predictions (Training set)
y_pred = model_fit.predict(X_train_lag)

print(f'Training Set - Lag: {lag}', f'Nb. Rows: {y_pred.shape}')
#print(f'Nb. Rows: {y_train.shape}')

y_train_lag['y_pred'] = y_pred
y_train_lag['residuals'] = y_train_lag['rd_expenditure'] - y_train_lag['y_pred']

print(y_train_lag.head(5))
print(y_train_lag.info())      

sns.set_theme(color_codes=True)
fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
plt.scatter(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])
plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_XGBoost/AllVar/model_scatter_train_{model_name}_{lag}.png")
#plt.show()

sns.set_theme(color_codes=True)
fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
plt.hist(y_train_lag['residuals'])
plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/model_residuals_train_{model_name}_{lag}.png")
#plt.show()

#Testing the Performance of our Model (training set)
print(f'Training Set - Lag: {lag}', metrics.mean_absolute_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
print(f'Training Set - Lag: {lag}', metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
print(f'Training Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])))

 # Calculate RMSE and MAE and add them to the lists (training set)

rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values))
mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values)

rmse_dict[lag] = rmse
mae_dict[lag] = mae

# Compute Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_train_lag[['rd_expenditure']].values - y_train_lag[['y_pred']].values) / y_train_lag[['rd_expenditure']].values)) * 100

mape_dict[lag] = mape

#####################
#####Test_Set#####        
# Now, make predictions for the test set

test_score=model_fit.score(X_test_lag, y_test_lag)
print(f'Test Set - Lag: {lag}', test_score)

# Make predictions
y_pred_test = model_fit.predict(X_test_lag)

y_test_lag['y_pred'] = y_pred_test
y_test_lag['residuals'] = y_test_lag['rd_expenditure'] - y_test_lag['y_pred']

print(y_test_lag.head(5))
print(y_test_lag.info())

sns.set_theme(color_codes=True)
fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])
plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/model_scatter_test_{model_name}_{lag}.png")
#plt.show()

sns.set_theme(color_codes=True)
fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
plt.hist(y_test_lag['residuals'])
plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/model_residuals_test_{model_name}_{lag}.png")
#plt.show()

#Testing the Performance of our Model
print(f'Test Set - Lag: {lag}', metrics.mean_absolute_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
print(f'Test Set - Lag: {lag}', metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
print(f'Test Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred']))) #

# Calculate RMSE, MAE and add them to the lists (for test set)
rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values))
mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values)

# Update your dictionary
rmse_dict_test[lag] = rmse
mae_dict_test[lag] = mae

# Compute Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test_lag[['rd_expenditure']].values - y_test_lag[['y_pred']].values) / y_test_lag[['rd_expenditure']].values)) * 100

# Update your dictionary
mape_dict_test[lag] = mape

#%%--------------------------------------------------------------------------
# Save the DataFrame to CSV
df_elasticities.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/elasticities_by_lag.csv', index=False)

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/merged_df_test_RV01_batches_all_{model_name}.csv")

# Convert all_important_features to a DataFrame and save
df_all_important_features = pd.DataFrame(all_important_features)
df_all_important_features.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/all_important_features_by_lag_{model_name}.csv", index=False)

# Create DataFrames to store the MAE and RMSE results for each lag
evaluation_train_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])
evaluation_test_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])

# Loop over each lag to populate the DataFrames
for lag in rmse_dict.keys():
    train_row = {'Lag': lag, 'MAE': mae_dict[lag], 'RMSE': rmse_dict[lag], 'MAPE': mape_dict[lag]}
    evaluation_train_df = evaluation_train_df.append(train_row, ignore_index=True)

for lag in rmse_dict_test.keys():
    test_row = {'Lag': lag, 'MAE': mae_dict_test[lag], 'RMSE': rmse_dict_test[lag], 'MAPE': mape_dict_test[lag]}
    evaluation_test_df = evaluation_test_df.append(test_row, ignore_index=True)


# Save the DataFrames to CSV files
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/evaluation_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/evaluation_test_{model_name}.csv", index=False)

#%%--------------------------------------------------------------------------

# Function to compute Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, we'll use a small constant. 
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

y_train_preds = model_fit.predict(X_train_lag)

print(len(y_train_lag['rd_expenditure'].values))
print(len(y_train_preds))
print(len(countries_train))

#breakpoint()

# Convert the predictions and actuals into a DataFrame for easier processing
df_predictions_train = pd.DataFrame({
    'y_true': y_train_lag['rd_expenditure'].values,
    'y_pred': y_train_preds,
    'Country': countries_train
})

# Placeholder for metrics by country
country_metrics = []

# Loop through each unique country and compute metrics
for country in df_predictions_train['Country'].unique():
    subset = df_predictions_train[df_predictions_train['Country'] == country]
    
    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
    mape = mean_absolute_percentage_error(subset['y_true'], subset['y_pred'])
    
    country_metrics.append({
        'Country': country,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    })

country_evaluation_df = pd.DataFrame(country_metrics)

# Save the results to CSV
country_evaluation_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/country_evaluation.csv", index=False)

import matplotlib.pyplot as plt

# Setting figure size
plt.figure(figsize=(15, 8))

# Plotting metrics
for metric in ['MAE', 'RMSE', 'MAPE']:
    plt.plot(country_evaluation_df['Country'], country_evaluation_df[metric], marker='o', label=metric)

plt.legend()
plt.title("Evaluation Metrics by Country")
plt.xlabel("Country")
plt.ylabel("Value")
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/MLPSci/Eval_Month_vis.png")
#plt.show()


#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################


# Initialize the model
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)
#n_estimators: 100 (lots of observations) to 1000 (few observations)
#The number of trees in our ensemble. Equivalent to the number of boosting rounds.

model_name = type(model).__name__

# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

rmse_dict = {}
mae_dict = {}
mape_dict = {}

rmse_dict_test = {}
mae_dict_test = {}
mape_dict_test = {}

all_important_features = []

# Define a function to compute elasticities for XGBoost
def compute_elasticity(model, X_sample, feature_index):
    X_perturbed = X_sample.copy()
    perturb_amount = 0.01  # 1% increase
    X_perturbed[:, feature_index] *= (1 + perturb_amount)
    
    y_pred_original = model.predict(X_sample)
    y_pred_perturbed = model.predict(X_perturbed)
    
    elasticity = ((y_pred_perturbed - y_pred_original) / y_pred_original) / perturb_amount
    return np.mean(elasticity)

# Initialize the dictionary to save the elasticities
all_elasticities = {}

# Create an empty DataFrame to store elasticity values along with lag and feature name
df_elasticities = pd.DataFrame(columns=['Lag', 'Feature Name', 'Elasticity'])

#%%--------------------------------------------------------------------------
y_train_lag = y_train_lag_xgb
y_val_lag = y_val_lag_xgb
y_test_lag = y_test_lag_xgb

X_train_lag = X_train_lag_xgb
X_val_lag = X_val_lag_xgb
X_test_lag = X_test_lag_xgb

countries_train = countries_train_xgb
countries_val = countries_val_xgb
countries_test = countries_test_xgb

# # Combine them
# X_combined = np.vstack((X_train_lag, X_val_lag))
# y_combined = np.concatenate((y_train_lag, y_val_lag))

# y_combined_df = pd.DataFrame(y_combined, columns=['rd_expenditure'])

# countries_combined = np.concatenate((countries_train, countries_val))

# X_train_lag = X_combined
# y_train_lag = y_combined_df
# countries_train = countries_combined

# Loop over the range of lags
for lag in range(max_lag, max_lag+1):

    # Now pass the data to the model
    model_fit = model.fit(X_train_lag, y_train_lag, early_stopping_rounds=10, eval_set=[(X_val_lag, y_val_lag)], verbose=False)

    # Feature Importance
    plot_importance(model_fit, max_num_features=15, importance_type='gain')  # top 15 features
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/feature_importance_plt_importance_{lag}.png", dpi=300)
    plt.close()  # Close the current figure
    
    # Use the `get_feature_names_out` method with the original column names
    feature_names = poly.get_feature_names_out(input_features=original_columns)

    # Sort features based on importance
    sorted_idx = model_fit.feature_importances_.argsort()[::-1]  # descending order

    # Plot top 15 features
    plt.figure(figsize=(10,7))  # Create a new figure for the second plot with specified size
    plt.barh(feature_names[sorted_idx][:15], model_fit.feature_importances_[sorted_idx][:15])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.title('Top 15 Important Features')
    plt.gca().invert_yaxis()  # to have the most important feature at the top
    plt.tight_layout()  # Adjust the layout to fit all labels
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/feature_importance_norm_name_{lag}.png", dpi=300)
    plt.close()  # Close the current figure

    # Use model.feature_importances_ directly to filter columns
    importance_threshold = 0.05  # example threshold

    # Filtering important columns from polynomial features
    important_features_mask = model_fit.feature_importances_ > importance_threshold #Create a Boolean Mask for Important Features
    important_cols_poly = [name for i, name in enumerate(feature_names) if important_features_mask[i]]


    # Save this iteration's important columns to a list
    all_important_features.append({"lag": lag, "important_features": ", ".join(important_cols_poly)})      

    #####################

    # Get the model parameters
    model_params = model_fit.get_params()
    print(f'Training Set - Lag: {lag}', model_params)

    # Calculate the accuracy score on the training set (R2?)
    train_score = model_fit.score(X_train_lag, y_train_lag)

    # Perform cross-validation using the model
    cv_scores = cross_val_score(model_fit, X_train_lag, y_train_lag, cv=4)
    print(f'Training Set - Lag: {lag}', cv_scores)

    print("Number of columns: ", X_train_lag.shape[1])
    
    #####################

    # Make predictions (Training set)
    y_pred = model_fit.predict(X_train_lag)

    print(f'Training Set - Lag: {lag}', f'Nb. Rows: {y_pred.shape}')
    #print(f'Nb. Rows: {y_train.shape}')

    y_train_lag['y_pred'] = y_pred
    y_train_lag['residuals'] = y_train_lag['rd_expenditure'] - y_train_lag['y_pred'] 

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_XGBoost/AllVar/model_scatter_train_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_train_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/model_residuals_train_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model (training set)
    print(f'Training Set - Lag: {lag}', metrics.mean_absolute_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])))

     # Calculate RMSE and MAE and add them to the lists (training set)
    
    rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values))
    mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values)

    rmse_dict[lag] = rmse
    mae_dict[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_train_lag[['rd_expenditure']].values - y_train_lag[['y_pred']].values) / y_train_lag[['rd_expenditure']].values)) * 100

    mape_dict[lag] = mape

    #####################
    #####Test_Set#####        
    # Now, make predictions for the test set

    test_score=model_fit.score(X_test_lag, y_test_lag)
    print(f'Test Set - Lag: {lag}', test_score)

    #define df for cv scores of different models
    df_cv_scores = pd.DataFrame(columns=['model', 'tr_cv_scores_means', 'tr_cv_scores_std', 'ts_cv_score'])

    # new row content 
    row_dict = {'model': [f"{model_name}"], 'tr_cv_scores_means': [cv_scores.mean()], 'tr_cv_scores_std': [cv_scores.std()],'ts_cv_score': [test_score]}
    row_df = pd.DataFrame (row_dict)
    df_cv_scores = pd.concat([df_cv_scores, row_df], ignore_index=True)
    print(f'Sets - Lag: {lag}', df_cv_scores.head(5))

    # Make predictions
    y_pred_test = model_fit.predict(X_test_lag)

    y_test_lag['y_pred'] = y_pred_test
    y_test_lag['residuals'] = y_test_lag['rd_expenditure'] - y_test_lag['y_pred']

    print(y_test_lag.head(5))
    print(y_test_lag.info())

    # Compute elasticities for each feature at this lag
    elasticities = {}
    for feature_index in range(X_test_lag.shape[1]):
        elasticity = compute_elasticity(model_fit, X_test_lag, feature_index)
        elasticities[feature_names[feature_index]] = elasticity

        # Save to DataFrame
        new_row = {'Lag': lag, 'Feature Name': feature_names[feature_index], 'Elasticity': elasticity}
        df_elasticities = df_elasticities.append(new_row, ignore_index=True)

    # Save the computed elasticities in the dictionary
    all_elasticities[lag] = elasticities

    print(f"Elasticities for lag {lag}: {elasticities}")

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/model_residuals_test_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model
    print(f'Test Set - Lag: {lag}', metrics.mean_absolute_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred']))) #

    # Calculate RMSE, MAE and add them to the lists (for test set)
    rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values))
    mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values)

    # Update your dictionary
    rmse_dict_test[lag] = rmse
    mae_dict_test[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test_lag[['rd_expenditure']].values - y_test_lag[['y_pred']].values) / y_test_lag[['rd_expenditure']].values)) * 100

    # Update your dictionary
    mape_dict_test[lag] = mape

    #####################

#%%--------------------------------------------------------------------------
# Save the DataFrame to CSV
df_elasticities.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/elasticities_by_lag.csv', index=False)

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/merged_df_test_RV01_batches_all_{model_name}.csv")

# Convert all_important_features to a DataFrame and save
df_all_important_features = pd.DataFrame(all_important_features)
df_all_important_features.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/all_important_features_by_lag_{model_name}.csv", index=False)

# Create DataFrames to store the MAE and RMSE results for each lag
evaluation_train_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])
evaluation_test_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])

# Loop over each lag to populate the DataFrames
for lag in rmse_dict.keys():
    train_row = {'Lag': lag, 'MAE': mae_dict[lag], 'RMSE': rmse_dict[lag], 'MAPE': mape_dict[lag]}
    evaluation_train_df = evaluation_train_df.append(train_row, ignore_index=True)

for lag in rmse_dict_test.keys():
    test_row = {'Lag': lag, 'MAE': mae_dict_test[lag], 'RMSE': rmse_dict_test[lag], 'MAPE': mape_dict_test[lag]}
    evaluation_test_df = evaluation_test_df.append(test_row, ignore_index=True)


# Save the DataFrames to CSV files
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/evaluation_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/evaluation_test_{model_name}.csv", index=False)

#%%--------------------------------------------------------------------------

# Function to compute Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, we'll use a small constant. 
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

#####

# After predicting using XGBoost
y_train_preds = model_fit.predict(X_train_lag)

# Convert the predictions and actuals into a DataFrame for easier processing
df_predictions_train = pd.DataFrame({
    'y_true': y_train_lag['rd_expenditure'].values,
    'y_pred': y_train_preds,
    'Country': countries_train
})

# Placeholder for metrics by country
country_metrics = []

# Loop through each unique country and compute metrics
for country in df_predictions_train['Country'].unique():
    subset = df_predictions_train[df_predictions_train['Country'] == country]
    
    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
    mape = mean_absolute_percentage_error(subset['y_true'], subset['y_pred'])
    
    country_metrics.append({
        'Country': country,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    })

country_evaluation_df = pd.DataFrame(country_metrics)

# Save the results to CSV
country_evaluation_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/country_evaluation.csv", index=False)

import matplotlib.pyplot as plt

# Setting figure size
plt.figure(figsize=(15, 8))

# Plotting metrics
for metric in ['MAE', 'RMSE', 'MAPE']:
    plt.plot(country_evaluation_df['Country'], country_evaluation_df[metric], marker='o', label=metric)

plt.legend()
plt.title("Evaluation Metrics by Country")
plt.xlabel("Country")
plt.ylabel("Value")
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/XGBoost/Eval_Month_vis.png")
#plt.show()


#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

model = RandomForestRegressor(random_state=2)

model_name = type(model).__name__

# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

rmse_dict = {}
mae_dict = {}
mape_dict = {}

rmse_dict_test = {}
mae_dict_test = {}
mape_dict_test = {}

all_important_features = []

# Define a function to compute elasticities for XGBoost
def compute_elasticity(model, X_sample, feature_index):
    X_perturbed = X_sample.copy()
    perturb_amount = 0.01  # 1% increase
    X_perturbed[:, feature_index] *= (1 + perturb_amount)
    
    y_pred_original = model.predict(X_sample)
    y_pred_perturbed = model.predict(X_perturbed)
    
    elasticity = ((y_pred_perturbed - y_pred_original) / y_pred_original) / perturb_amount
    return np.mean(elasticity)

# Initialize the dictionary to save the elasticities
all_elasticities = {}

# Create an empty DataFrame to store elasticity values along with lag and feature name
df_elasticities = pd.DataFrame(columns=['Lag', 'Feature Name', 'Elasticity'])

#%%--------------------------------------------------------------------------
y_train_lag = y_train_lag_rf
y_val_lag = y_val_lag_rf
y_test_lag = y_test_lag_rf

X_train_lag = X_train_lag_rf
X_val_lag = X_val_lag_rf
X_test_lag = X_test_lag_rf

countries_train = countries_train_rf
countries_val = countries_val_rf
countries_test = countries_test_rf

# Combine them
X_combined = np.vstack((X_train_lag, X_val_lag))
y_combined = np.concatenate((y_train_lag, y_val_lag))

y_combined_df = pd.DataFrame(y_combined, columns=['rd_expenditure'])

countries_combined = np.concatenate((countries_train, countries_val))

X_train_lag = X_combined
y_train_lag = y_combined_df
countries_train = countries_combined

# Loop over the range of lags
for lag in range(max_lag, max_lag+1):

    # Now pass the data to the model
    model_fit = model.fit(X_train_lag, y_train_lag)

    #####################

    # Make predictions (Training set)
    y_pred = model_fit.predict(X_train_lag)

    print(f'Training Set - Lag: {lag}', f'Nb. Rows: {y_pred.shape}')
    #print(f'Nb. Rows: {y_train.shape}')

    y_train_lag['y_pred'] = y_pred
    y_train_lag['residuals'] = y_train_lag['rd_expenditure'] - y_train_lag['y_pred'] 

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/model_scatter_train_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_train_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/model_residuals_train_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model (training set)
    print(f'Training Set - Lag: {lag}', metrics.mean_absolute_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])))

     # Calculate RMSE and MAE and add them to the lists (training set)
    
    rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values))
    mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values)

    rmse_dict[lag] = rmse
    mae_dict[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_train_lag[['rd_expenditure']].values - y_train_lag[['y_pred']].values) / y_train_lag[['rd_expenditure']].values)) * 100

    mape_dict[lag] = mape

    #####################
    #####Test_Set#####        
    # Make predictions
    y_pred_test = model_fit.predict(X_test_lag)

    y_test_lag['y_pred'] = y_pred_test
    y_test_lag['residuals'] = y_test_lag['rd_expenditure'] - y_test_lag['y_pred']

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/model_residuals_test_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model
    print(f'Test Set - Lag: {lag}', metrics.mean_absolute_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred']))) #

    # Calculate RMSE, MAE and add them to the lists (for test set)
    rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values))
    mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values)

    # Update your dictionary
    rmse_dict_test[lag] = rmse
    mae_dict_test[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test_lag[['rd_expenditure']].values - y_test_lag[['y_pred']].values) / y_test_lag[['rd_expenditure']].values)) * 100

    # Update your dictionary
    mape_dict_test[lag] = mape

#%%--------------------------------------------------------------------------
# Save the DataFrame to CSV
df_elasticities.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/elasticities_by_lag.csv', index=False)

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/merged_df_test_RV01_batches_all_{model_name}.csv")

# Convert all_important_features to a DataFrame and save
df_all_important_features = pd.DataFrame(all_important_features)
df_all_important_features.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/all_important_features_by_lag_{model_name}.csv", index=False)

# Create DataFrames to store the MAE and RMSE results for each lag
evaluation_train_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])
evaluation_test_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])

# Loop over each lag to populate the DataFrames
for lag in rmse_dict.keys():
    train_row = {'Lag': lag, 'MAE': mae_dict[lag], 'RMSE': rmse_dict[lag], 'MAPE': mape_dict[lag]}
    evaluation_train_df = evaluation_train_df.append(train_row, ignore_index=True)

for lag in rmse_dict_test.keys():
    test_row = {'Lag': lag, 'MAE': mae_dict_test[lag], 'RMSE': rmse_dict_test[lag], 'MAPE': mape_dict_test[lag]}
    evaluation_test_df = evaluation_test_df.append(test_row, ignore_index=True)


# Save the DataFrames to CSV files
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/evaluation_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/evaluation_test_{model_name}.csv", index=False)

#%%--------------------------------------------------------------------------
# Function to compute Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, we'll use a small constant. 
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

#####

# After predicting using XGBoost
y_train_preds = model_fit.predict(X_train_lag)

# Convert the predictions and actuals into a DataFrame for easier processing
df_predictions_train = pd.DataFrame({
    'y_true': y_train_lag['rd_expenditure'].values,
    'y_pred': y_train_preds,
    'Country': countries_train
})

# Placeholder for metrics by country
country_metrics = []

# Loop through each unique country and compute metrics
for country in df_predictions_train['Country'].unique():
    subset = df_predictions_train[df_predictions_train['Country'] == country]
    
    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
    mape = mean_absolute_percentage_error(subset['y_true'], subset['y_pred'])
    
    country_metrics.append({
        'Country': country,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    })

country_evaluation_df = pd.DataFrame(country_metrics)

# Save the results to CSV
country_evaluation_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/country_evaluation.csv", index=False)

import matplotlib.pyplot as plt

# Setting figure size
plt.figure(figsize=(15, 8))

# Plotting metrics
for metric in ['MAE', 'RMSE', 'MAPE']:
    plt.plot(country_evaluation_df['Country'], country_evaluation_df[metric], marker='o', label=metric)

plt.legend()
plt.title("Evaluation Metrics by Country")
plt.xlabel("Country")
plt.ylabel("Value")
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/RF/Eval_Month_vis.png")
#plt.show()


#breakpoint()

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

model = LinearSVR(random_state=123)

model_name = type(model).__name__

# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

rmse_dict = {}
mae_dict = {}
mape_dict = {}

rmse_dict_test = {}
mae_dict_test = {}
mape_dict_test = {}

all_important_features = []

# Define a function to compute elasticities for XGBoost
def compute_elasticity(model, X_sample, feature_index):
    X_perturbed = X_sample.copy()
    perturb_amount = 0.01  # 1% increase
    X_perturbed[:, feature_index] *= (1 + perturb_amount)
    
    y_pred_original = model.predict(X_sample)
    y_pred_perturbed = model.predict(X_perturbed)
    
    elasticity = ((y_pred_perturbed - y_pred_original) / y_pred_original) / perturb_amount
    return np.mean(elasticity)

# Initialize the dictionary to save the elasticities
all_elasticities = {}

# Create an empty DataFrame to store elasticity values along with lag and feature name
df_elasticities = pd.DataFrame(columns=['Lag', 'Feature Name', 'Elasticity'])

#%%--------------------------------------------------------------------------

y_train_lag = y_train_lag_svr
y_val_lag = y_val_lag_svr
y_test_lag = y_test_lag_svr

X_train_lag = X_train_lag_svr
X_val_lag = X_val_lag_svr
X_test_lag = X_test_lag_svr

countries_train = countries_train_svr
countries_val = countries_val_svr
countries_test = countries_test_svr

# Combine them
X_combined = np.vstack((X_train_lag, X_val_lag))
y_combined = np.concatenate((y_train_lag, y_val_lag))

y_combined_df = pd.DataFrame(y_combined, columns=['rd_expenditure'])

countries_combined = np.concatenate((countries_train, countries_val))

X_train_lag = X_combined
y_train_lag = y_combined_df
countries_train = countries_combined

# Loop over the range of lags
for lag in range(max_lag, max_lag+1):

    # Now pass the data to the model
    model_fit = model.fit(X_train_lag, y_train_lag)

    # # Feature Importance
    # plot_importance(model_fit, max_num_features=15, importance_type='gain')  # top 15 features
    # plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/feature_importance_plt_importance_{lag}.png", dpi=300)
    # plt.close()  # Close the current figure
    
    # # Use the `get_feature_names_out` method with the original column names
    # feature_names = poly.get_feature_names_out(input_features=original_columns)

    # # Sort features based on importance
    # sorted_idx = model_fit.feature_importances_.argsort()[::-1]  # descending order

    # # Plot top 15 features
    # plt.figure(figsize=(10,7))  # Create a new figure for the second plot with specified size
    # plt.barh(feature_names[sorted_idx][:15], model_fit.feature_importances_[sorted_idx][:15])
    # plt.xlabel('Feature Importance')
    # plt.ylabel('Feature Names')
    # plt.title('Top 15 Important Features')
    # plt.gca().invert_yaxis()  # to have the most important feature at the top
    # plt.tight_layout()  # Adjust the layout to fit all labels
    # plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/feature_importance_norm_name_{lag}.png", dpi=300)
    # plt.close()  # Close the current figure

    # # Use model.feature_importances_ directly to filter columns
    # importance_threshold = 0.05  # example threshold

    # # Filtering important columns from polynomial features
    # important_features_mask = model_fit.feature_importances_ > importance_threshold #Create a Boolean Mask for Important Features
    # important_cols_poly = [name for i, name in enumerate(feature_names) if important_features_mask[i]]


    # # Save this iteration's important columns to a list
    # all_important_features.append({"lag": lag, "important_features": ", ".join(important_cols_poly)})      

    #####################

    # # Get the model parameters
    # model_params = model_fit.get_params()
    # print(f'Training Set - Lag: {lag}', model_params)

    # # Calculate the accuracy score on the training set (R2?)
    # train_score = model_fit.score(X_train_lag, y_train_lag)

    # # Perform cross-validation using the model
    # cv_scores = cross_val_score(model_fit, X_train_lag, y_train_lag, cv=4)
    # print(f'Training Set - Lag: {lag}', cv_scores)

    # print("Number of columns: ", X_train_lag.shape[1])
    
    #####################

    # Make predictions (Training set)
    y_pred = model_fit.predict(X_train_lag)

    print(f'Training Set - Lag: {lag}', f'Nb. Rows: {y_pred.shape}')
    #print(f'Nb. Rows: {y_train.shape}')

    y_train_lag['y_pred'] = y_pred
    y_train_lag['residuals'] = y_train_lag['rd_expenditure'] - y_train_lag['y_pred'] 

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/model_scatter_train_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_train_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/model_residuals_train_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model (training set)
    print(f'Training Set - Lag: {lag}', metrics.mean_absolute_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])))

     # Calculate RMSE and MAE and add them to the lists (training set)
    
    rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values))
    mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values)

    rmse_dict[lag] = rmse
    mae_dict[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_train_lag[['rd_expenditure']].values - y_train_lag[['y_pred']].values) / y_train_lag[['rd_expenditure']].values)) * 100

    mape_dict[lag] = mape

    #####################
    #####Test_Set#####        
    # Now, make predictions for the test set

    # test_score=model_fit.score(X_test_lag, y_test_lag)
    # print(f'Test Set - Lag: {lag}', test_score)

    # #define df for cv scores of different models
    # df_cv_scores = pd.DataFrame(columns=['model', 'tr_cv_scores_means', 'tr_cv_scores_std', 'ts_cv_score'])

    # # new row content 
    # row_dict = {'model': [f"{model_name}"], 'tr_cv_scores_means': [cv_scores.mean()], 'tr_cv_scores_std': [cv_scores.std()],'ts_cv_score': [test_score]}
    # row_df = pd.DataFrame (row_dict)
    # df_cv_scores = pd.concat([df_cv_scores, row_df], ignore_index=True)
    # print(f'Sets - Lag: {lag}', df_cv_scores.head(5))

    # Make predictions
    y_pred_test = model_fit.predict(X_test_lag)

    y_test_lag['y_pred'] = y_pred_test
    y_test_lag['residuals'] = y_test_lag['rd_expenditure'] - y_test_lag['y_pred']

    print(y_test_lag.head(5))
    print(y_test_lag.info())

    # # Compute elasticities for each feature at this lag
    # elasticities = {}
    # for feature_index in range(X_test_lag.shape[1]):
    #     elasticity = compute_elasticity(model_fit, X_test_lag, feature_index)
    #     elasticities[feature_names[feature_index]] = elasticity

    #     # Save to DataFrame
    #     new_row = {'Lag': lag, 'Feature Name': feature_names[feature_index], 'Elasticity': elasticity}
    #     df_elasticities = df_elasticities.append(new_row, ignore_index=True)

    # # Save the computed elasticities in the dictionary
    # all_elasticities[lag] = elasticities

    # print(f"Elasticities for lag {lag}: {elasticities}")

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/model_residuals_test_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model
    print(f'Test Set - Lag: {lag}', metrics.mean_absolute_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred']))) #

    # Calculate RMSE, MAE and add them to the lists (for test set)
    rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values))
    mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values)

    # Update your dictionary
    rmse_dict_test[lag] = rmse
    mae_dict_test[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test_lag[['rd_expenditure']].values - y_test_lag[['y_pred']].values) / y_test_lag[['rd_expenditure']].values)) * 100

    # Update your dictionary
    mape_dict_test[lag] = mape

#%%--------------------------------------------------------------------------
# Save the DataFrame to CSV
df_elasticities.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/elasticities_by_lag.csv', index=False)

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/merged_df_test_RV01_batches_all_{model_name}.csv")

# Convert all_important_features to a DataFrame and save
df_all_important_features = pd.DataFrame(all_important_features)
df_all_important_features.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/all_important_features_by_lag_{model_name}.csv", index=False)

# Create DataFrames to store the MAE and RMSE results for each lag
evaluation_train_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])
evaluation_test_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])

# Loop over each lag to populate the DataFrames
for lag in rmse_dict.keys():
    train_row = {'Lag': lag, 'MAE': mae_dict[lag], 'RMSE': rmse_dict[lag], 'MAPE': mape_dict[lag]}
    evaluation_train_df = evaluation_train_df.append(train_row, ignore_index=True)

for lag in rmse_dict_test.keys():
    test_row = {'Lag': lag, 'MAE': mae_dict_test[lag], 'RMSE': rmse_dict_test[lag], 'MAPE': mape_dict_test[lag]}
    evaluation_test_df = evaluation_test_df.append(test_row, ignore_index=True)


# Save the DataFrames to CSV files
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/evaluation_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/evaluation_test_{model_name}.csv", index=False)

#%%--------------------------------------------------------------------------

# error_data = {
#     'Month': [],
#     'Percentage_Error': []
# }

# # Compute the percentage error:

# model_predictions = y_pred_test
# actual_values = y_test_lag['rd_expenditure'].values

# # Calculate individual percentage errors
# percentage_errors = 100 * (actual_values - model_predictions) / actual_values

# # Assuming you have a variable 'Month' in your X_test that represents the month for each data point
# error_data['Month'].extend(X_test['Month'].values)
# error_data['Percentage_Error'].extend(percentage_errors)

# # Convert the data into a DataFrame for easier manipulation and visualization
# error_df = pd.DataFrame(error_data)

# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Month', y='Percentage_Error', data=error_df)
# plt.title("Distribution of Percentage Errors by Month")
# plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/Test_MGTs_MAPE_Month.png")
# #plt.show()

#%%--------------------------------------------------------------------------
# Function to compute Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, we'll use a small constant. 
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

#####

# After predicting using XGBoost
y_train_preds = model_fit.predict(X_train_lag)

# Convert the predictions and actuals into a DataFrame for easier processing
df_predictions_train = pd.DataFrame({
    'y_true': y_train_lag['rd_expenditure'].values,
    'y_pred': y_train_preds,
    'Country': countries_train
})

# Placeholder for metrics by country
country_metrics = []

# Loop through each unique country and compute metrics
for country in df_predictions_train['Country'].unique():
    subset = df_predictions_train[df_predictions_train['Country'] == country]
    
    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
    mape = mean_absolute_percentage_error(subset['y_true'], subset['y_pred'])
    
    country_metrics.append({
        'Country': country,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    })

country_evaluation_df = pd.DataFrame(country_metrics)

# Save the results to CSV
country_evaluation_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/country_evaluation.csv", index=False)

import matplotlib.pyplot as plt

# Setting figure size
plt.figure(figsize=(15, 8))

# Plotting metrics
for metric in ['MAE', 'RMSE', 'MAPE']:
    plt.plot(country_evaluation_df['Country'], country_evaluation_df[metric], marker='o', label=metric)

plt.legend()
plt.title("Evaluation Metrics by Country")
plt.xlabel("Country")
plt.ylabel("Value")
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/SVR/Eval_Month_vis.png")
#plt.show()

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

cv_value = 2 
model = ElasticNetCV(cv=cv_value, random_state=123)

model_name = type(model).__name__

# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

rmse_dict = {}
mae_dict = {}
mape_dict = {}

rmse_dict_test = {}
mae_dict_test = {}
mape_dict_test = {}

all_important_features = []

# Define a function to compute elasticities for XGBoost
def compute_elasticity(model, X_sample, feature_index):
    X_perturbed = X_sample.copy()
    perturb_amount = 0.01  # 1% increase
    X_perturbed[:, feature_index] *= (1 + perturb_amount)
    
    y_pred_original = model.predict(X_sample)
    y_pred_perturbed = model.predict(X_perturbed)
    
    elasticity = ((y_pred_perturbed - y_pred_original) / y_pred_original) / perturb_amount
    return np.mean(elasticity)

# Initialize the dictionary to save the elasticities
all_elasticities = {}

# Create an empty DataFrame to store elasticity values along with lag and feature name
df_elasticities = pd.DataFrame(columns=['Lag', 'Feature Name', 'Elasticity'])

#%%--------------------------------------------------------------------------

y_train_lag = y_train_lag_en
y_val_lag = y_val_lag_en
y_test_lag = y_test_lag_en

X_train_lag = X_train_lag_en
X_val_lag = X_val_lag_en
X_test_lag = X_test_lag_en

countries_train = countries_train_en
countries_val = countries_val_en
countries_test = countries_test_en

# Combine them
X_combined = np.vstack((X_train_lag, X_val_lag))
y_combined = np.concatenate((y_train_lag, y_val_lag))

y_combined_df = pd.DataFrame(y_combined, columns=['rd_expenditure'])

countries_combined = np.concatenate((countries_train, countries_val))

X_train_lag = X_combined
y_train_lag = y_combined_df
countries_train = countries_combined

# Loop over the range of lags
for lag in range(max_lag, max_lag+1):

    # Now pass the data to the model
    model_fit = model.fit(X_train_lag, y_train_lag)

    # # Feature Importance
    # plot_importance(model_fit, max_num_features=15, importance_type='gain')  # top 15 features
    # plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/feature_importance_plt_importance_{lag}.png", dpi=300)
    # plt.close()  # Close the current figure
    
    # # Use the `get_feature_names_out` method with the original column names
    # feature_names = poly.get_feature_names_out(input_features=original_columns)

    # # Sort features based on importance
    # sorted_idx = model_fit.feature_importances_.argsort()[::-1]  # descending order

    # # Plot top 15 features
    # plt.figure(figsize=(10,7))  # Create a new figure for the second plot with specified size
    # plt.barh(feature_names[sorted_idx][:15], model_fit.feature_importances_[sorted_idx][:15])
    # plt.xlabel('Feature Importance')
    # plt.ylabel('Feature Names')
    # plt.title('Top 15 Important Features')
    # plt.gca().invert_yaxis()  # to have the most important feature at the top
    # plt.tight_layout()  # Adjust the layout to fit all labels
    # plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/feature_importance_norm_name_{lag}.png", dpi=300)
    # plt.close()  # Close the current figure

    # # Use model.feature_importances_ directly to filter columns
    # importance_threshold = 0.05  # example threshold

    # # Filtering important columns from polynomial features
    # important_features_mask = model_fit.feature_importances_ > importance_threshold #Create a Boolean Mask for Important Features
    # important_cols_poly = [name for i, name in enumerate(feature_names) if important_features_mask[i]]


    # # Save this iteration's important columns to a list
    # all_important_features.append({"lag": lag, "important_features": ", ".join(important_cols_poly)})      

    # #####################

    # # Get the model parameters
    # model_params = model_fit.get_params()
    # print(f'Training Set - Lag: {lag}', model_params)

    # # Calculate the accuracy score on the training set (R2?)
    # train_score = model_fit.score(X_train_lag, y_train_lag)

    # # Perform cross-validation using the model
    # cv_scores = cross_val_score(model_fit, X_train_lag, y_train_lag, cv=4)
    # print(f'Training Set - Lag: {lag}', cv_scores)

    # print("Number of columns: ", X_train_lag.shape[1])
    
    #####################

    # Make predictions (Training set)
    y_pred = model_fit.predict(X_train_lag)

    print(f'Training Set - Lag: {lag}', f'Nb. Rows: {y_pred.shape}')
    #print(f'Nb. Rows: {y_train.shape}')

    y_train_lag['y_pred'] = y_pred
    y_train_lag['residuals'] = y_train_lag['rd_expenditure'] - y_train_lag['y_pred'] 

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/model_scatter_train_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_train_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/model_residuals_train_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model (training set)
    print(f'Training Set - Lag: {lag}', metrics.mean_absolute_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])))

     # Calculate RMSE and MAE and add them to the lists (training set)
    
    rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values))
    mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values)

    rmse_dict[lag] = rmse
    mae_dict[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_train_lag[['rd_expenditure']].values - y_train_lag[['y_pred']].values) / y_train_lag[['rd_expenditure']].values)) * 100

    mape_dict[lag] = mape

    #####################
    #####Test_Set#####        
    # Now, make predictions for the test set

    # test_score=model_fit.score(X_test_lag, y_test_lag)
    # print(f'Test Set - Lag: {lag}', test_score)

    # #define df for cv scores of different models
    # df_cv_scores = pd.DataFrame(columns=['model', 'tr_cv_scores_means', 'tr_cv_scores_std', 'ts_cv_score'])

    # # new row content 
    # row_dict = {'model': [f"{model_name}"], 'tr_cv_scores_means': [cv_scores.mean()], 'tr_cv_scores_std': [cv_scores.std()],'ts_cv_score': [test_score]}
    # row_df = pd.DataFrame (row_dict)
    # df_cv_scores = pd.concat([df_cv_scores, row_df], ignore_index=True)
    # print(f'Sets - Lag: {lag}', df_cv_scores.head(5))

    # Make predictions
    y_pred_test = model_fit.predict(X_test_lag)

    y_test_lag['y_pred'] = y_pred_test
    y_test_lag['residuals'] = y_test_lag['rd_expenditure'] - y_test_lag['y_pred']

    print(y_test_lag.head(5))
    print(y_test_lag.info())

    # # Compute elasticities for each feature at this lag
    # elasticities = {}
    # for feature_index in range(X_test_lag.shape[1]):
    #     elasticity = compute_elasticity(model_fit, X_test_lag, feature_index)
    #     elasticities[feature_names[feature_index]] = elasticity

    #     # Save to DataFrame
    #     new_row = {'Lag': lag, 'Feature Name': feature_names[feature_index], 'Elasticity': elasticity}
    #     df_elasticities = df_elasticities.append(new_row, ignore_index=True)

    # # Save the computed elasticities in the dictionary
    # all_elasticities[lag] = elasticities

    # print(f"Elasticities for lag {lag}: {elasticities}")

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/model_residuals_test_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model
    print(f'Test Set - Lag: {lag}', metrics.mean_absolute_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred']))) #

    # Calculate RMSE, MAE and add them to the lists (for test set)
    rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values))
    mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values)

    # Update your dictionary
    rmse_dict_test[lag] = rmse
    mae_dict_test[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test_lag[['rd_expenditure']].values - y_test_lag[['y_pred']].values) / y_test_lag[['rd_expenditure']].values)) * 100

    # Update your dictionary
    mape_dict_test[lag] = mape

    #####################

#%%--------------------------------------------------------------------------
# Save the DataFrame to CSV
df_elasticities.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/elasticities_by_lag.csv', index=False)

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/merged_df_test_RV01_batches_all_{model_name}.csv")

# Convert all_important_features to a DataFrame and save
df_all_important_features = pd.DataFrame(all_important_features)
df_all_important_features.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/all_important_features_by_lag_{model_name}.csv", index=False)

# Create DataFrames to store the MAE and RMSE results for each lag
evaluation_train_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])
evaluation_test_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE'])

# Loop over each lag to populate the DataFrames
for lag in rmse_dict.keys():
    train_row = {'Lag': lag, 'MAE': mae_dict[lag], 'RMSE': rmse_dict[lag], 'MAPE': mape_dict[lag]}
    evaluation_train_df = evaluation_train_df.append(train_row, ignore_index=True)

for lag in rmse_dict_test.keys():
    test_row = {'Lag': lag, 'MAE': mae_dict_test[lag], 'RMSE': rmse_dict_test[lag], 'MAPE': mape_dict_test[lag]}
    evaluation_test_df = evaluation_test_df.append(test_row, ignore_index=True)


# Save the DataFrames to CSV files
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/evaluation_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/evaluation_test_{model_name}.csv", index=False)

#%%--------------------------------------------------------------------------

# error_data = {
#     'Month': [],
#     'Percentage_Error': []
# }

# # Compute the percentage error:

# model_predictions = y_pred_test
# actual_values = y_test_lag['rd_expenditure'].values

# # Calculate individual percentage errors
# percentage_errors = 100 * (actual_values - model_predictions) / actual_values

# # Assuming you have a variable 'Month' in your X_test that represents the month for each data point
# error_data['Month'].extend(X_test['Month'].values)
# error_data['Percentage_Error'].extend(percentage_errors)

# # Convert the data into a DataFrame for easier manipulation and visualization
# error_df = pd.DataFrame(error_data)

# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Month', y='Percentage_Error', data=error_df)
# plt.title("Distribution of Percentage Errors by Month")
# plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/Test_MGTs_MAPE_Month.png")
# #plt.show()

#%%--------------------------------------------------------------------------
# Function to compute Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, we'll use a small constant. 
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

#####

# After predicting using XGBoost
y_train_preds = model_fit.predict(X_train_lag)

# Convert the predictions and actuals into a DataFrame for easier processing
df_predictions_train = pd.DataFrame({
    'y_true': y_train_lag['rd_expenditure'].values,
    'y_pred': y_train_preds,
    'Country': countries_train
})

# Placeholder for metrics by country
country_metrics = []

# Loop through each unique country and compute metrics
for country in df_predictions_train['Country'].unique():
    subset = df_predictions_train[df_predictions_train['Country'] == country]
    
    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
    mape = mean_absolute_percentage_error(subset['y_true'], subset['y_pred'])
    
    country_metrics.append({
        'Country': country,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    })

country_evaluation_df = pd.DataFrame(country_metrics)

# Save the results to CSV
country_evaluation_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/country_evaluation.csv", index=False)

import matplotlib.pyplot as plt

# Setting figure size
plt.figure(figsize=(15, 8))

# Plotting metrics
for metric in ['MAE', 'RMSE', 'MAPE']:
    plt.plot(country_evaluation_df['Country'], country_evaluation_df[metric], marker='o', label=metric)

plt.legend()
plt.title("Evaluation Metrics by Country")
plt.xlabel("Country")
plt.ylabel("Value")
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/ElasticNet/Eval_Month_vis.png")
#plt.show()

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################
## Meta Model?
#%%--------------------------------------------------------------------------

for model in ensemble.models:
    model_predictions = model(X_test_tensor, country_indices_test).detach().numpy().flatten()
    actual_values = y_test_tensor.numpy().flatten()

# Get base model predictions on validation set (Base model 1: Ensemble MLP)
val_predictions = []
for model in ensemble.models:
    model_preds = model(X_val_tensor, country_indices_val)
    val_predictions.append(model_preds.detach().numpy())

# Include XGBoost predictions for the validation set (Base model 2: Single gradient Boosting XGBoost)
xgb_val_preds = model_fit.predict(X_val_lag) # Assuming 'model_fit' is your trained XGBoost model
val_predictions.append(xgb_val_preds)

# Stack predictions to create new input features for meta-model
X_meta_val = np.column_stack(val_predictions)


# Get base model predictions on test set
test_predictions = []
for model in ensemble.models:
    model_preds = model(X_test_tensor, country_indices_test)
    test_predictions.append(model_preds.detach().numpy())

# Include XGBoost predictions for the test set
xgb_test_preds = model_fit.predict(X_test_lag)
test_predictions.append(xgb_test_preds)

# Stack predictions to create new input features for meta-model
X_meta_test = np.column_stack(test_predictions)

#breakpoint()

###
# Checking the correlation between base models before training the meta-model
# ideally the correlation between two models should not be high. resulting in more effective model and less overfitting
# Assuming ensemble_predictions is the average of all individual model predictions for the validation set
ensemble_val_predictions = ensemble.predict(X_val_tensor, country_indices_val).detach().numpy()

# correlation_matrix = np.corrcoef(ensemble_val_predictions, xgb_val_preds)
# correlation = correlation_matrix[0, 1]
# print(f"Correlation between ensemble and XGBoost predictions: {correlation:.4f}")


# Ensuring both are 1D arrays
ensemble_val_predictions = ensemble_val_predictions.flatten()
xgb_val_preds = xgb_val_preds.flatten()

correlation_matrix = np.corrcoef(ensemble_val_predictions, xgb_val_preds)
correlation = correlation_matrix[0, 1]
print(f"Correlation between ensemble and XGBoost predictions: {correlation:.4f}")


#breakpoint()

meta_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
meta_model.fit(X_meta_val, y_val_tensor.numpy())

# Placeholder for metrics
metrics_data = []

# Evaluate each model in the ensemble
for idx, model in enumerate(ensemble.models):
    with torch.no_grad():
        model.eval()
        preds = model(X_test_tensor, country_indices_test).detach().numpy().flatten()
        actuals = y_test_tensor.numpy().flatten()
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        mape = mean_absolute_percentage_error(actuals, preds)
        
        metrics_data.append({
            'Model': f"Model_{idx+1}",
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

# Evaluate the ensemble
ensemble_test_predictions = ensemble.predict(X_test_tensor, country_indices_test).detach().numpy()
ensemble_rmse = np.sqrt(mean_squared_error(y_test_tensor.detach().numpy(), ensemble_test_predictions))
ensemble_mae = mean_absolute_error(y_test_tensor.detach().numpy(), ensemble_test_predictions)
ensemble_mape = mean_absolute_percentage_error(y_test_tensor.detach().numpy(), ensemble_test_predictions)

metrics_data.append({
    'Model': 'Ensemble',
    'RMSE': ensemble_rmse,
    'MAE': ensemble_mae,
    'MAPE': ensemble_mape
})

# Evaluate XGBoost
xgb_test_rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, xgb_test_preds))
xgb_test_mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, xgb_test_preds)
xgb_test_mape = mean_absolute_percentage_error(y_test_lag[['rd_expenditure']].values, xgb_test_preds)

metrics_data.append({
    'Model': 'XGBoost',
    'RMSE': xgb_test_rmse,
    'MAE': xgb_test_mae,
    'MAPE': xgb_test_mape
})

# Ensure the meta-model is defined and trained before this step
# Evaluate the meta-model
meta_test_preds = meta_model.predict(X_meta_test)
meta_rmse = np.sqrt(mean_squared_error(y_test_tensor.detach().numpy(), meta_test_preds))
meta_mae = mean_absolute_error(y_test_tensor.detach().numpy(), meta_test_preds)
meta_mape = mean_absolute_percentage_error(y_test_tensor.detach().numpy(), meta_test_preds)

metrics_data.append({
    'Model': 'Meta-Model',
    'RMSE': meta_rmse,
    'MAE': meta_mae,
    'MAPE': meta_mape
})

# Convert to DataFrame and save to CSV
df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/LagRD/test10_node500/meta_model_metrics.csv", index=False)
