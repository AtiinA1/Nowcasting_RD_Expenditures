

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm
import seaborn as sns
import random
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

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


# Read and filter GDP per Capita (Local currency)
#gdp_df = pd.read_csv('./data/OECD/EAG_FIN_ANNEX_2_11062023222623819.csv')
#gdp_df = gdp_df[gdp_df['STATISTICS'] == 'GDP_CAPITA']
#gdp_df = gdp_df[['LOCATION','TIME', 'Value']]
#gdp_df = gdp_df.rename(columns={'Value': 'GDP_CAPITA'})
#print(gdp_df.columns)


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

# Loop through each variable
for var in variables:
    # Create a binary column to indicate missing values for the variable
    oecd_df_rev[var + '_missing'] = oecd_df_rev[var].isna().astype(int)
    # Fill NA values in the variable with zero
    oecd_df_rev[var].fillna(0, inplace=True)

oecd_df_rev_4lag = oecd_df_rev[['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol']]

oecd_df_rev_4lag = oecd_df_rev_4lag.dropna(subset=['Country', 'Year', 'gdpca', 'unemp_rate', 'population', 'inflation', 'export_vol', 'import_vol'])

rd_expenditure_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'rd_expenditure_missing']]

print(rd_expenditure_df_rev.head())
print(rd_expenditure_df_rev['Country'].unique())
print(rd_expenditure_df_rev.nunique())
print(rd_expenditure_df_rev.info())

print(oecd_df_rev.head())
print(oecd_df_rev['Country'].unique())
print(oecd_df_rev.nunique())
print(oecd_df_rev.info())



#%%--------------------------------------------------------------------------

merged_df = oecd_df_rev
merged_df = merged_df.drop_duplicates()

print(f'Nb. Rows: {merged_df.shape[0]}')
print(merged_df.columns)
print(merged_df.head(5))

merged_df.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/merged_df.csv')

breakpoint()

#%%--------------------------------------------------------------------------

#Model 3: Annual GT + Monthly GT (if available)


# Initialize the model
model = RandomForestRegressor(random_state=2)

model_name = type(model).__name__


# Define a range of lags to consider
max_lag = 3  # For example, consider up to 3 years of lagged values

# Initialize lists to store the RMSE and MAE for each lag

#rmse_list = []
#mae_list = []

#rmse_list_test = []
#mae_list_test = []

rmse_dict = {}
mae_dict = {}
mape_dict = {}

rmse_dict_test = {}
mae_dict_test = {}
mape_dict_test = {}


for col in merged_df.columns:
    if col != 'Country':  # Exclude the 'Country' column
        if merged_df[col].dtype == 'object':  # If the column is a string
            merged_df[col] = merged_df[col].str.replace(',', '')  # Remove the commas
            merged_df[col] = merged_df[col].replace('--', np.nan)  # Replace '--' with NaN
            merged_df[col] = merged_df[col].astype(float)  # Convert the column to float

print(merged_df['Country'].nunique())
print(merged_df.isna().sum())

#%%--------------------------------------------------------------------------

# Create the lagged values for all variables (GTs only) first
for lag in range(1, max_lag+1):
    for col in rd_expenditure_df_rev.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
            # Drop rows with NA in 'rd_expenditure' and its lagged columns
            #merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', 'rd_expenditure'])

# Loop over the range of lags
for lag in range(1, max_lag+1):
    # Create lagged values for macro variables
    for col in oecd_df_rev_4lag.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
            print(merged_df.columns) # Debugging line
            #merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', f'{col}'])


# Get unique countries
#unique_countries = rd_expenditure_df_rev['Country'].unique()

# Limit merged_df to those countries
#merged_df = merged_df[merged_df['Country'].isin(unique_countries)]

#merged_df[lagged_col_name] = merged_df[lagged_col_name].fillna(0, inplace=True)
merged_df = merged_df.dropna()
print(merged_df.isna().sum())
print(merged_df.head())
print(merged_df['Country'].nunique())

merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or ('_lag' in col)]]
X_all.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/X_all.csv')


#%%--------------------------------------------------------------------------

y = merged_df[['Country', 'Year', 'rd_expenditure']]

    
#%%--------------------------------------------------------------------------
    


# Loop over the unique countries - country specific training

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, train_size=0.8, test_size=0.2, random_state=123
)

# Loop over the range of lags
for lag in range(1, max_lag+1):
    
    # Initialize the list outside the loop so you can keep adding to it
    relevant_columns = ['Year', 'Country']

    # Loop over the range of lags
    for cur_lag in range(1, lag+1):
        
        # Append the lagged columns for the current lag to the list
        relevant_columns += [f"{col}_lag{cur_lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]
        relevant_columns += [f"{col}_lag{cur_lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]

    # Apply the relevant_columns to the train and test datasets
    X_train_lag = X_train[relevant_columns]
    X_test_lag = X_test[relevant_columns]
    y_train_lag = y_train[['rd_expenditure']]
    y_test_lag = y_test[['rd_expenditure']]
    
    # Convert 'Country' column to dummies
    X_train_lag = pd.get_dummies(X_train_lag, columns=['Country'])
    X_test_lag = pd.get_dummies(X_test_lag, columns=['Country'])

    # Now pass the data to the model
    model_fit = model.fit(X_train_lag, y_train_lag)


    # Get the model parameters
    model_params = model_fit.get_params()
    print(f'Training Set - Lag: {lag}', model_params)

    # Calculate the accuracy score on the training set (R2?)
    train_score = model_fit.score(X_train_lag, y_train_lag)

    #print(f'Training Set - Lag: {lag}', model_fit.coef_, model_fit.intercept_)

    # Perform cross-validation using the model
    cv_scores = cross_val_score(model_fit, X_train_lag, y_train_lag, cv=4)
    print(f'Training Set - Lag: {lag}', cv_scores)

    #model_coeff=model_fit.coef_
    #model_coeff=model_coeff.T
    
    #model_coeff=pd.DataFrame(model_coeff, X_train_lag.columns, columns = ['Coeff'])
    #model_coeff.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/AllVar/model_coeff_{lag}.csv")
    #print(f'Training Set - Lag: {lag}', model_coeff.head(5))

    #print("Shape of coefficients: ", model_coeff.shape)
    print("Number of columns: ", len(X_train_lag.columns))
    
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
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/model_scatter_train_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_train_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/model_residuals_train_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model (training set)
    print(f'Training Set - Lag: {lag}', metrics.mean_absolute_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred']))
    print(f'Training Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_train_lag['rd_expenditure'], y_train_lag['y_pred'])))

     # Calculate RMSE and MAE and add them to the lists (training set)
    #rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values))
    #mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values)

    #Formula: RMSE = √((1/n) * Σ(y_i - ŷ_i)²)
    #Formula: MAE = (1/n) * Σ|y_i - ŷ_i|
    # Add the RMSE, MAE, country, and lag to their respective lists

    #rmse_list.append(rmse)
    #mae_list.append(mae)
    
    rmse = np.sqrt(mean_squared_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values))
    mae = mean_absolute_error(y_train_lag[['rd_expenditure']].values, y_train_lag[['y_pred']].values)

    rmse_dict[lag] = rmse
    mae_dict[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_train_lag[['rd_expenditure']].values - y_train_lag[['y_pred']].values) / y_train_lag[['rd_expenditure']].values)) * 100

    mape_dict[lag] = mape

    #####Test_Set#####        
    # Now, make predictions for the test set
    
    test_score=model_fit.score(X_test_lag, y_test_lag)
    print(f'Test Set - Lag: {lag}', test_score)

    #define df for cv scores of different models
    df_cv_scores = pd.DataFrame(columns=['model', 'tr_cv_scores_means', 'tr_cv_scores_std', 'ts_cv_score'])

    # new row content 
    row_dict = {'model': ['LinRegression'], 'tr_cv_scores_means': [cv_scores.mean()], 'tr_cv_scores_std': [cv_scores.std()],'ts_cv_score': [test_score]}
    row_df = pd.DataFrame (row_dict)
    df_cv_scores = pd.concat([df_cv_scores, row_df], ignore_index=True)
    print(f'Sets - Lag: {lag}', df_cv_scores.head(5))

    # Make predictions
    y_pred_test = model_fit.predict(X_test_lag)

    y_test_lag['y_pred'] = y_pred_test
    y_test_lag['residuals'] = y_test_lag['rd_expenditure'] - y_test_lag['y_pred']

    print(y_test_lag.head(5))
    print(y_test_lag.info())

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/model_residuals_test_{model_name}_{lag}.png")
    #plt.show()

    #Testing the Performance of our Model
    print(f'Test Set - Lag: {lag}', metrics.mean_absolute_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])) #
    print(f'Test Set - Lag: {lag}', np.sqrt(metrics.mean_squared_error(y_test_lag['rd_expenditure'], y_test_lag['y_pred']))) #

     # Calculate RMSE and MAE and add them to the lists (for training set)
    #rmse_test = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values))
    #mae_test = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values)

    #Formula: RMSE = √((1/n) * Σ(y_i - ŷ_i)²)
    #Formula: MAE = (1/n) * Σ|y_i - ŷ_i|

    #rmse_list_test.append(rmse_test)
    #mae_list_test.append(mae_test)

    rmse = np.sqrt(mean_squared_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values))
    mae = mean_absolute_error(y_test_lag[['rd_expenditure']].values, y_test_lag[['y_pred']].values)

    rmse_dict_test[lag] = rmse
    mae_dict_test[lag] = mae

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test_lag[['rd_expenditure']].values - y_test_lag[['y_pred']].values) / y_test_lag[['rd_expenditure']].values)) * 100

    mape_dict_test[lag] = mape        

#%%--------------------------------------------------------------------------

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/merged_df_test_RV01_batches_all_{model_name}.csv")

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
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/evaluation_batches_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/Macros/evaluation_testAGT1_{model_name}.csv", index=False)

