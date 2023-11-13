import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm
import seaborn as sns
import random
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
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
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

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

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

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

df_trends_monthly_org = df_trends_monthly

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

df_trends_monthly = df_trends_monthly_aggregated

# Loop over each keyword column and create new columns representing the mean of each variable 
# for the months of the current year up to the current month (not including the current month)
for column in df_trends_monthly.columns:
    if column not in ['date', 'Country', 'Year', 'Month']:
        df_trends_monthly_aggregated[f'{column}_mean_YTD'] = df_trends_monthly_aggregated.groupby(['Year', 'Country'])[column].transform(lambda x: x.expanding().mean().shift())

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_YTD.csv')

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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_YTD_YR.csv')
print(df_trends_monthly_aggregated.columns)

# Generate a list of columns to be dropped
columns_to_drop = [column for column in df_trends_monthly_aggregated.columns if ('_mean_YTD' not in column) and ('_yearly_avg' not in column) and (column not in ['date', 'Country', 'Month', 'Year'])]

# Drop the columns
df_trends_rev = df_trends_monthly_aggregated.drop(columns=columns_to_drop)

#%%--------------------------------------------------------------------------

# Merge the two DataFrames
merged_df = pd.merge(oecd_df_rev, df_trends_rev, on=['Country', 'Year'], how='left')
merged_df = merged_df.drop_duplicates()

print(f'Nb. Rows: {merged_df.shape[0]}')
print(merged_df.columns)
print(merged_df.head(5))

merged_df.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/merged_df.csv')

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

for col in df_trends_monthly.columns:
    if col != 'Country':  # Exclude the 'Country' column
        if df_trends_monthly[col].dtype == 'object':  # If the column is a string
            df_trends_monthly[col] = df_trends_monthly[col].str.replace(',', '')  # Remove the commas
            df_trends_monthly[col] = df_trends_monthly[col].replace('--', np.nan)  # Replace '--' with NaN
            df_trends_monthly[col] = df_trends_monthly[col].astype(float)  # Convert the column to float
            #merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

df_trends_monthly.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly.csv')
breakpoint()

#%%--------------------------------------------------------------------------
# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

# Create the lagged values for all variables (GTs only) first
for lag in range(1, max_lag+1):
    for col in df_trends.columns:
        if col not in ['Country', 'Year', 'Month']:
            # Create a new lagged column
            lagged_col_name = f'{col}_yearly_avg_lag{lag}'
            merged_df[lagged_col_name] = merged_df.groupby('Country')[f'{col}_yearly_avg'].shift(lag)
            # Fill NA values in the new lagged column with zero
            merged_df[lagged_col_name].fillna(0, inplace=True)

    for col in rd_expenditure_df_rev.columns:
        if col not in ['Country', 'Year', 'Month']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
            # Drop rows with NA in 'rd_expenditure' and its lagged columns
            merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', 'rd_expenditure'])

# Fill NaN in columns with keyword 'mean_YTD' 
for col in merged_df.columns[merged_df.columns.str.contains('mean_YTD')]:
    merged_df[col].fillna(0, inplace=True)


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
# cols_dropna += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]
cols_dropna += ['Year']

# Fill NaN with 0 in specific columns
for col in cols_fillna_zero:
    if col in merged_df.columns:
        merged_df[col].fillna(0, inplace=True)

# Drop rows with NaN in specific columns
for col in cols_dropna:
    if col in merged_df.columns:
        merged_df.dropna(subset=[col], inplace=True)

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/merged_df_test_RV01_batches_all_AggMGT_Evolution_0.csv")
#breakpoint()
merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]
merged_df = merged_df[(merged_df['Year'] >= 2004)]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or (col == 'Month')  or ('_lag' in col) or ('_mean_YTD' in col)]]
X_all.fillna(0, inplace=True)

X_all.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/X_all_AggMGT_Evolution.csv")

#breakpoint()

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

y = merged_df[['Country', 'Year', 'Month', 'rd_expenditure']]

# First, split data into training + validation and test sets (e.g., 80-20 split)
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Then, split the training + validation data into training and validation sets (e.g., 80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


# Initialize the list outside the loop so you can keep adding to it
relevant_columns = ['Year', 'Month','Country']

# Loop over the range of lags
for lag in range(1, max_lag+1):
    
    # Append the lagged columns for the current lag to the list
    relevant_columns += [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]
    relevant_columns += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]
    relevant_columns += [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year', 'Month']]

# After the loop, relevant_columns will have all the columns for lags from 1 to max_lag

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

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


#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

# Save to CSV if needed
csv_path_elasticities = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/test10/XGBoost/elasticities_by_lag.csv"
df_elasticities_avg = pd.read_csv(csv_path_elasticities)
df_elasticities = df_elasticities_avg

df_elasticities = df_elasticities[df_elasticities['Lag'] == 3]

countries = ["CA", "CH", "CN", "DE", "GB", "JP", "KR", "US"]

# Duplicate elasticities for each country
duplicated_data = []
for country in countries:
    for index, row in df_elasticities.iterrows():
        duplicated_data.append([country, row['Lag'], row['Feature Name'], row['Elasticity']])

# Create a new dataframe with the duplicated data
df_elasticities_expanded = pd.DataFrame(duplicated_data, columns=['Country', 'Lag', 'Feature Name', 'Elasticity'])

print(df_elasticities_expanded)
breakpoint()

df_elasticities = df_elasticities_expanded

#############################################################################

# Exclude non-GoogleTrend columns to get a list of GoogleTrend keyword columns
# Exclude columns with '_mean_YTD'
exclude_cols = ['Year', 'Month', 'Country', 'date'] + [col for col in df_trends_monthly.columns if '_mean_YTD' in col]
google_trend_cols = [col for col in df_trends_monthly.columns if col not in exclude_cols]

# Get a list of columns that do not contain "_mean_YTD"
cols_to_keep = [col for col in df_trends_monthly.columns if '_mean_YTD' not in col]

# Filter the dataframe to keep only those columns
df_trends_monthly = df_trends_monthly[cols_to_keep]

# Step 1: Calculate Monthly Proportions for each country
for col in google_trend_cols:
    # Calculate the yearly sum for each country
    yearly_sums = df_trends_monthly.groupby(['Year', 'Country'])[col].transform('sum')
    df_trends_monthly[f'{col}_proportion'] = df_trends_monthly[col] / yearly_sums


# Filter df_elasticities to include only entries with '_mean_YTD'
df_elasticities_YTD = df_elasticities[df_elasticities['Feature Name'].str.contains('_mean_YTD')]

# Prepare the df_elasticities_YTD DataFrame by removing the '_mean_YTD' suffix
df_elasticities_YTD['base_feature'] = df_elasticities_YTD['Feature Name'].str.replace('_mean_YTD', '')

# Merge the df_trends_monthly DataFrame with the prepared df_elasticities_YTD DataFrame
for col in google_trend_cols:
    merged_dff = pd.merge(df_trends_monthly, df_elasticities_YTD[df_elasticities_YTD['base_feature'] == col], on='Country', how='left')
    df_trends_monthly[f'{col}_elasticity'] = merged_dff['Elasticity']
    df_trends_monthly[f'{col}_adjusted_proportion'] = df_trends_monthly[f'{col}_elasticity'] * df_trends_monthly[f'{col}_proportion']


#reakpoint()
# Sum all the _adjusted_proportion columns to get the total_adjusted_proportion for each row
df_trends_monthly['total_adjusted_proportion'] = df_trends_monthly[[f'{col}_adjusted_proportion' for col in google_trend_cols]].sum(axis=1)

# Calculate the total_adjusted_proportion_yearly for each year and country
df_trends_monthly['total_adjusted_proportion_yearly'] = df_trends_monthly.groupby(['Year', 'Country'])['total_adjusted_proportion'].transform('sum')

df_trends_monthly = pd.merge(df_trends_monthly, oecd_df_rev, on=['Country', 'Year'], how='left')
df_trends_monthly = df_trends_monthly.drop_duplicates()

# Calculate monthly_rd_expenditure
df_trends_monthly['monthly_rd_expenditure'] = df_trends_monthly['rd_expenditure'] * (df_trends_monthly['total_adjusted_proportion'] / df_trends_monthly['total_adjusted_proportion_yearly'])

# Save to CSV
df_trends_monthly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_elast_adjusted.csv", index=False)

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################
df = df_trends_monthly

# Sort the dataframe based on 'Country', 'Year', and 'Month'
df_trends_monthly = df_trends_monthly.sort_values(by=['Country', 'Year', 'Month'])

# Define the lags you want to create
lags = [1, 2, 3]

# For each column in google_trend_cols, create its lagged values
for col in google_trend_cols:
    for lag in lags:
        df_trends_monthly[f"{col}_lag{lag}"] = df_trends_monthly.groupby('Country')[col].shift(lag)

for lag in lags:
    df_trends_monthly[f"monthly_rd_expenditure_lag{lag}"] = df_trends_monthly.groupby('Country')['monthly_rd_expenditure'].shift(lag)


df_trends_monthly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_elast_adjusted_lagged.csv", index=False)
print(df_trends_monthly.columns)

df_trends_monthly = df_trends_monthly.fillna(0)

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

# Filter df_elasticities to include only entries with '_yearly_avg_lag'
df_elasticities_lagged = df_elasticities[df_elasticities['Feature Name'].str.contains('_yearly_avg_lag')]

# Extract the base column name from the "Feature" column by removing the '_yearly_avg_lag' suffix
df_elasticities_lagged['base_feature'] = df_elasticities_lagged['Feature Name'].str.replace('_yearly_avg', '')

# Merge the elasticities with the df_trends_monthly DataFrame
for col in google_trend_cols:
    for lag in lags:
        lagged_col = f"{col}_lag{lag}"
        merged_df = pd.merge(df_trends_monthly, df_elasticities_lagged[df_elasticities_lagged['base_feature'] == lagged_col], on='Country', how='left')
        df_trends_monthly[f"{lagged_col}_elasticity"] = merged_df['Elasticity']


#############################################################################

# 1. Filter df_elasticities to include only entries with 'rd_expenditure_lag'
df_elasticities_lagged = df_elasticities[df_elasticities['Feature Name'].str.contains('rd_expenditure_lag')]

# 2. Since the base feature in df_elasticities is the same as the column name in df_trends_monthly (just with "monthly_" prefixed), 
# there's no need for a separate 'base_feature' column. We can directly match them.

# 3. Merge the elasticities with the df_trends_monthly DataFrame
for lag in lags:
    col = f"monthly_rd_expenditure_lag{lag}"
    merged_df = pd.merge(df_trends_monthly, df_elasticities_lagged[df_elasticities_lagged['Feature Name'] == f"rd_expenditure_lag{lag}"], on='Country', how='left')
    df_trends_monthly[f"{col}_elasticity"] = merged_df['Elasticity']


df_trends_monthly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_elast_adjusted_lagged_final.csv", index=False)

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

# 1. Fill NaN values in Google trend columns and their lags with zero.
for col in google_trend_cols:
    df_trends_monthly[col].fillna(0, inplace=True)
    for lag in lags:
        lagged_col = f"{col}_lag{lag}"
        df_trends_monthly[lagged_col].fillna(0, inplace=True)


# Drop rows with NaN values in the 'monthly_rd_expenditure' lags.
lagged_rd_cols = [f"monthly_rd_expenditure_lag{lag}" for lag in lags]
df_trends_monthly.dropna(subset=lagged_rd_cols, inplace=True)

# Save the results
df_trends_monthly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_NAs.csv", index=False)

# missing_elasticities = []

# for col in google_trend_cols:
#     if f"{col}_elasticity" not in df_trends_monthly.columns:
#         missing_elasticities.append(f"{col}_elasticity")
#     for lag in lags:
#         if f"{col}_lag{lag}_elasticity" not in df_trends_monthly.columns:
#             missing_elasticities.append(f"{col}_lag{lag}_elasticity")

# # Check if any elasticity columns for rd_expenditure lags are missing
# for lag in lags:
#     if f"monthly_rd_expenditure_lag{lag}_elasticity" not in df_trends_monthly.columns:
#         missing_elasticities.append(f"monthly_rd_expenditure_lag{lag}_elasticity")

# print("Missing Elasticities: ", missing_elasticities)
# nan_values = df_trends_monthly.isna().sum()
# print(nan_values[nan_values > 0])

breakpoint()

df_trends_monthly['estimated_rd_expenditure'] = 0.0

# Multiply each column by its elasticity and accumulate
for col in google_trend_cols:
    df_trends_monthly['estimated_rd_expenditure'] += df_trends_monthly[col] * df_trends_monthly.get(f"{col}_elasticity", 1)
    for lag in lags:
        lagged_col = f"{col}_lag{lag}"
        df_trends_monthly['estimated_rd_expenditure'] += df_trends_monthly[lagged_col] * df_trends_monthly.get(f"{lagged_col}_elasticity", 1)

# Add the contributions from the rd_expenditure lags
for lag in lags:
    col = f"monthly_rd_expenditure_lag{lag}"
    df_trends_monthly['estimated_rd_expenditure'] += df_trends_monthly[col] * df_trends_monthly.get(f"{col}_elasticity", 1)

# Save the results
df_trends_monthly.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/df_trends_monthly_estimate.csv", index=False)

#############################################################################
#%%--------------------------------------------------------------------------
#############################################################################

# Define the MAPE function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load the dataframe
# df_trends_monthly = pd.read_csv("YOUR_CSV_PATH_HERE.csv")

# Define the actual and forecast columns
actual_col = 'monthly_rd_expenditure'
forecast_col = 'estimated_rd_expenditure'

print(df_trends_monthly['Country'].unique())
#df_trends_monthly = df_trends_monthly.dropna()
print(df_trends_monthly['Country'].unique())

breakpoint()

# Filter out rows where either actual_col or forecast_col have NaN values
df_trends_monthly = df_trends_monthly.dropna(subset=[actual_col, forecast_col])

# # Calculate metrics using the filtered data
# rmse_overall = np.sqrt(mean_squared_error(filtered_df[actual_col], filtered_df[forecast_col]))
# mape_overall = mean_absolute_percentage_error(filtered_df[actual_col], filtered_df[forecast_col])
# mae_overall = mean_absolute_error(filtered_df[actual_col], filtered_df[forecast_col])

# Group by Country and Year and sum up the actual and estimated columns
yearly_aggregated = df_trends_monthly.groupby(['Country', 'Year']).agg({
    actual_col: 'sum',
    forecast_col: 'sum'
}).reset_index()

# Calculate the metrics for these aggregated values
yearly_aggregated['Yearly_RMSE'] = np.sqrt(mean_squared_error(yearly_aggregated[actual_col], yearly_aggregated[forecast_col]))
yearly_aggregated['Yearly_MAPE'] = mean_absolute_percentage_error(yearly_aggregated[actual_col], yearly_aggregated[forecast_col])
yearly_aggregated['Yearly_MAE'] = mean_absolute_error(yearly_aggregated[actual_col], yearly_aggregated[forecast_col])

# Save the aggregated data to a CSV
yearly_aggregated.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/yearly_aggregated_metrics.csv", index=False)


# Calculate metrics for the entire dataset
rmse_overall = np.sqrt(mean_squared_error(df_trends_monthly[actual_col], df_trends_monthly[forecast_col]))
mape_overall = mean_absolute_percentage_error(df_trends_monthly[actual_col], df_trends_monthly[forecast_col])
mae_overall = mean_absolute_error(df_trends_monthly[actual_col], df_trends_monthly[forecast_col])

print("Overall Metrics:")
print(f"RMSE: {rmse_overall}, MAPE: {mape_overall}, MAE: {mae_overall}")

overall_metrics = pd.DataFrame({
    'RMSE': [rmse_overall],
    'MAPE': [mape_overall],
    'MAE': [mae_overall]
})

overall_metrics.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/overall_metrics.csv", index=False)


# Calculate metrics grouped by Country, Year, and Month
grouped_metrics = df_trends_monthly.groupby(['Country', 'Year', 'Month']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(mean_squared_error(group[actual_col], group[forecast_col])),
        'MAPE': mean_absolute_percentage_error(group[actual_col], group[forecast_col]),
        'MAE': mean_absolute_error(group[actual_col], group[forecast_col])
    })
).reset_index()

# Save the metrics to a CSV
grouped_metrics.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/metrics_per_group.csv", index=False)


# Metrics grouped by Country
grouped_by_country = df_trends_monthly.groupby(['Country']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(mean_squared_error(group[actual_col], group[forecast_col])),
        'MAPE': mean_absolute_percentage_error(group[actual_col], group[forecast_col]),
        'MAE': mean_absolute_error(group[actual_col], group[forecast_col])
    })
).reset_index()

# Metrics grouped by Country and Year
grouped_by_year = df_trends_monthly.groupby(['Year']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(mean_squared_error(group[actual_col], group[forecast_col])),
        'MAPE': mean_absolute_percentage_error(group[actual_col], group[forecast_col]),
        'MAE': mean_absolute_error(group[actual_col], group[forecast_col])
    })
).reset_index()

# Metrics grouped by Country, Year, and Month
grouped_by_month = df_trends_monthly.groupby(['Month']).apply(
    lambda group: pd.Series({
        'RMSE': np.sqrt(mean_squared_error(group[actual_col], group[forecast_col])),
        'MAPE': mean_absolute_percentage_error(group[actual_col], group[forecast_col]),
        'MAE': mean_absolute_error(group[actual_col], group[forecast_col])
    })
).reset_index()

# Save the metrics to different CSVs
grouped_by_country.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/metrics_by_country.csv", index=False)
grouped_by_year.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/metrics_by_year.csv", index=False)
grouped_by_month.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/MGTwRD/bem/metrics_by_month.csv", index=False)



