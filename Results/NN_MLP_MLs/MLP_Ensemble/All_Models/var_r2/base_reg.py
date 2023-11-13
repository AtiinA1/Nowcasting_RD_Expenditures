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
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#---------------------------------------------------------------------------------------------------------------------------

country_cd_df = pd.read_csv('/Users/atin/Nowcasting/data/country_code.csv')


# Read in and filter rd_expenditure_df to only include rows with sector of performance equal to Business enterprise + considering total fundings (millions-LOCAL CURRENCY)
rd_expenditure_df = pd.read_csv('/Users/atin/Nowcasting/data/GERD/DP_LIVE_08052023154811337.csv')
rd_expenditure_df = rd_expenditure_df.rename(columns={'Value': 'rd_expenditure'})  
rd_expenditure_df = rd_expenditure_df[rd_expenditure_df['MEASURE'] == 'MLN_USD'] #USD constant prices using 2015 base year 
print(rd_expenditure_df.columns)

rd_expenditure_df['rd_expenditure'] = rd_expenditure_df['rd_expenditure'] / 1000


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


oecd_df_rev_4lag = oecd_df_rev[['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol']]
#oecd_df_rev_4lag = oecd_df_rev_4lag.dropna(subset=['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol'])

rd_expenditure_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'rd_expenditure_missing']]
#rd_expenditure_df_rev['rd_expenditure'] = rd_expenditure_df_rev['rd_expenditure'] / 1000

print(rd_expenditure_df_rev.head(5))
#breakpoint()
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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/df_trends_monthly_YTD.csv')

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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/df_trends_monthly_YTD_YR.csv')
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

merged_df.to_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/merged_df.csv')

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

# Loop over the range of lags
for lag in range(1, max_lag+1):
    # Create lagged values for macro variables
    for col in oecd_df_rev_4lag.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
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
cols_dropna += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]
cols_dropna += ['Year']

# Fill NaN with 0 in specific columns
for col in cols_fillna_zero:
    if col in merged_df.columns:
        merged_df[col].fillna(0, inplace=True)

# Drop rows with NaN in specific columns
for col in cols_dropna:
    if col in merged_df.columns:
        merged_df.dropna(subset=[col], inplace=True)

# Define the columns you wish to fill NaN values for
# cols_to_fill = [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year']]
# cols_to_fill += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']]
cols_to_fill = [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]

# Fill NaN with the mean of each country over all available years for each column
for col in cols_to_fill:
    if col in merged_df.columns:
        merged_df[col] = merged_df.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/merged_df_test_RV01_batches_all_AggMGT_Evolution_0.csv")
#breakpoint()
merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]
merged_df = merged_df[(merged_df['Year'] >= 2004)]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or (col == 'Month')  or ('_lag' in col) or ('_mean_YTD' in col)]]
X_all.fillna(0, inplace=True)

X_all.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/X_all_AggMGT_Evolution.csv")

#breakpoint()
#%%--------------------------------------------------------------------------

y = merged_df[['Country', 'Year', 'Month', 'rd_expenditure']]

# First, split data into training + validation and test sets (e.g., 80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Initialize the list outside the loop so you can keep adding to it
relevant_columns = ['Year', 'Month','Country']
relevant_columns += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]

# Loop over the range of lags
for lag in range(1, max_lag+1):
    
    # Append the lagged columns for the current lag to the list
    relevant_columns += [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]
    relevant_columns += [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year', 'Month']]
    relevant_columns += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year', 'Month']]

# After the loop, relevant_columns will have all the columns for lags from 1 to max_lag
X_train.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/X_train_df.csv")

#####################
# Apply the relevant_columns to the train and test datasets
X_train_lag = X_train[relevant_columns]
X_test_lag = X_test[relevant_columns]

y_train_lag = y_train[['rd_expenditure']]
y_test_lag = y_test[['rd_expenditure']]

#####################
# Convert 'Country' column to dummies
X_train_lag = pd.get_dummies(X_train_lag, columns=['Country'])
X_test_lag = pd.get_dummies(X_test_lag, columns=['Country'])

# Save column names before any operation that changes X_train_lag to a numpy array
original_columns = X_train_lag.columns.tolist()

# Adding Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_lag)
X_test_poly = poly.transform(X_test_lag)

X_train_lag = X_train_poly
X_test_lag = X_test_poly

###########################################################################

# Initialize the model
model = LinearRegression()

model_name = type(model).__name__

# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

rmse_dict = {}
mae_dict = {}
mape_dict = {}

rmse_dict_test = {}
mae_dict_test = {}
mape_dict_test = {}

# Initialize dictionaries to store R^2 values
r2_dict_train = {}
r2_dict_test = {}

###########################################################################


# Loop over the range of lags
for lag in range(max_lag, max_lag+1):

    # Now pass the data to the model
    model_fit = model.fit(X_train_lag, y_train_lag)

    # Get the model parameters
    model_params = model_fit.get_params()
    print(f'Training Set - Lag: {lag}', model_params)

    # Calculate the accuracy score on the training set (R2?)
    train_score = model_fit.score(X_train_lag, y_train_lag)

    # Perform cross-validation using the model
    cv_scores = cross_val_score(model_fit, X_train_lag, y_train_lag, cv=4)
    print(f'Training Set - Lag: {lag}', cv_scores)

    # Make predictions (Training set)
    y_pred_train = model_fit.predict(X_train_lag)

    train_r2 = r2_score(y_train_lag[['rd_expenditure']].values, y_pred_train)
    print(f'Training Set - Lag: {lag} R^2: {train_r2}')

    y_train_lag['y_pred_train'] = y_pred_train
    y_train_lag['residuals'] = y_train_lag['rd_expenditure'] - y_train_lag['y_pred_train']

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_train_lag['rd_expenditure'], y_train_lag['y_pred_train'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_XGBoost/AllVar/model_scatter_train_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_train_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/model_residuals_train_{model_name}_{lag}.png")

    #Testing the Performance of our Model (training set)
    print(f'Training Set - Lag: {lag}', metrics.mean_absolute_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred_train']))
    print(f'Training Set - Lag: {lag}', metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred_train']))
    print(f'Training Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred_train'])))

     # Calculate RMSE and MAE and MAPE and add them to the lists (training set)
    
    rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred_train']].values))
    mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred_train']].values)
    mape = np.mean(np.abs((y_train_lag[['rd_expenditure']].values - y_train_lag[['y_pred_train']].values) / y_train_lag[['rd_expenditure']].values)) * 100


    rmse_dict[lag] = rmse
    mae_dict[lag] = mae
    mape_dict[lag] = mape

    r2_dict_train[lag] = train_r2    

    #####Test_Set#####        
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

    y_test_lag['y_pred_test'] = y_pred_test
    y_test_lag['residuals'] = y_test_lag['rd_expenditure'] - y_test_lag['y_pred_test']

    test_r2 = r2_score(y_test_lag[['rd_expenditure']].values, y_pred_test)
    print(f'Test Set - Lag: {lag} R^2: {test_r2}')

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred_test'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/model_residuals_test_{model_name}_{lag}.png")
    #plt.show()


    # Inside your loop, after making predictions
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred_test'])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted Values for Lag {lag}')
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/scatterplot_lag_{lag}.png")
    plt.close(fig)  # Close the figure to free up memory

    #Testing the Performance of our Model
    print(f'Test Set - Lag: {lag}', metrics.mean_absolute_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred_test'])) #
    print(f'Test Set - Lag: {lag}', metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred_test'])) #
    print(f'Test Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred_test']))) #

    # Calculate RMSE, MAE, MAPE and add them to the lists (for test set)
    rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred_test']].values))
    mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred_test']].values)
    mape = np.mean(np.abs((y_test_lag[['rd_expenditure']].values - y_test_lag[['y_pred_test']].values) / y_test_lag[['rd_expenditure']].values)) * 100

    # Update your dictionary
    rmse_dict_test[lag] = rmse
    mae_dict_test[lag] = mae
    mape_dict_test[lag] = mape

    r2_dict_test[lag] = test_r2


    # Concatenate the training and test DataFrames
    combined_df = pd.concat([y_train_lag, y_test_lag])

    # Select relevant columns and rename as needed
    df_combined_pred_vs_true = combined_df[['rd_expenditure', 'y_pred_test']]
    df_combined_pred_vs_true.rename(columns={'rd_expenditure': 'True_Values', 'y_pred_test': 'Predicted_Values'}, inplace=True)

    # Save to CSV
    csv_file_path = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/df_combined_pred_vs_true.csv"
    df_combined_pred_vs_true.to_csv(csv_file_path, index=False)



#%%--------------------------------------------------------------------------

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/merged_df_test_RV01_batches_all_{model_name}.csv")

# Create DataFrames to store the MAE and RMSE results for each lag
evaluation_train_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE', 'R^2'])
evaluation_test_df = pd.DataFrame(columns=['Lag', 'MAE', 'RMSE', 'MAPE', 'R^2'])

# Loop over each lag to populate the DataFrames


# Loop over each lag to populate the DataFrames
for lag in rmse_dict.keys():
    train_row = {
        'Lag': lag,
        'MAE': mae_dict[lag],
        'RMSE': rmse_dict[lag],
        'MAPE': mape_dict[lag],
        'R^2': r2_dict_train[lag]
    }
    evaluation_train_df = evaluation_train_df.append(train_row, ignore_index=True)

for lag in rmse_dict_test.keys():
    test_row = {
        'Lag': lag,
        'MAE': mae_dict_test[lag],
        'RMSE': rmse_dict_test[lag],
        'MAPE': mape_dict_test[lag],
        'R^2': r2_dict_test[lag]
    }
    evaluation_test_df = evaluation_test_df.append(test_row, ignore_index=True)

# Save the DataFrames to CSV files
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/evaluation_batches_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_reg/evaluation_testAGT1_{model_name}.csv", index=False)

