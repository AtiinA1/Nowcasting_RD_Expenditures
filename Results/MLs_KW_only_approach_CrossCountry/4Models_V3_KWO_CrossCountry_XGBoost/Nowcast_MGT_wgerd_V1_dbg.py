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
from xgboost import XGBRegressor
#%%--------------------------------------------------------------------------


country_cd_df = pd.read_csv('/Users/atin/Nowcasting/data/country_code.csv')


# Read in and filter rd_expenditure_df to only include rows with sector of performance equal to Business enterprise + considering total fundings (millions-LOCAL CURRENCY)
rd_expenditure_df = pd.read_csv('/Users/atin/Nowcasting/data/GERD/DP_LIVE_08052023154811337.csv')
rd_expenditure_df = rd_expenditure_df.rename(columns={'Value': 'rd_expenditure'})  
rd_expenditure_df = rd_expenditure_df[rd_expenditure_df['MEASURE'] == 'MLN_USD'] #USD constant prices using 2015 base year 
print(rd_expenditure_df.columns)


# Merge the two DataFrames
country_cd_df = country_cd_df[['alpha-2', 'alpha-3']]
rd_expenditure_df_rev = pd.merge(rd_expenditure_df, country_cd_df, how='left' , left_on='LOCATION', right_on='alpha-3')
# Rename the columns
rd_expenditure_df_rev = rd_expenditure_df_rev.rename(columns={'alpha-2': 'Country', 'TIME': 'Year'})
rd_expenditure_df_rev = rd_expenditure_df_rev[['Country', 'Year', 'rd_expenditure']]

oecd_df_rev = rd_expenditure_df_rev

# Create a binary column to indicate missing values in 'rd_expenditure'
oecd_df_rev['rd_expenditure_missing'] = oecd_df_rev['rd_expenditure'].isna().astype(int)

# Fill NA values in 'rd_expenditure' with zero
oecd_df_rev['rd_expenditure'].fillna(0, inplace=True)
oecd_df_rev['rd_expenditure_missing'].fillna(0, inplace=True)

rd_expenditure_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'rd_expenditure_missing']]

#%%--------------------------------------------------------------------------


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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/df_trends_monthly_YTD.csv')

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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/df_trends_monthly_YTD_YR.csv')
print(df_trends_monthly_aggregated.columns)


# Generate a list of columns to be dropped
columns_to_drop = [column for column in df_trends_monthly_aggregated.columns if ('_mean_YTD' not in column) and ('_yearly_avg' not in column) and (column not in ['date', 'Country', 'Month', 'Year'])]

# Drop the columns
df_trends_rev = df_trends_monthly_aggregated.drop(columns=columns_to_drop)


#breakpoint()

#%%--------------------------------------------------------------------------

# Merge the two DataFrames
merged_df = pd.merge(oecd_df_rev, df_trends_rev, on=['Country', 'Year'], how='left')
merged_df = merged_df.drop_duplicates()

print(f'Nb. Rows: {merged_df.shape[0]}')
print(merged_df.columns)
print(merged_df.head(5))

merged_df.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/merged_df.csv')

#breakpoint()

#%%--------------------------------------------------------------------------

#Model 3: Annual GT + Monthly GT (if available)

# Initialize the model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
#n_estimators: 100 (lots of observations) to 1000 (few observations)
#The number of trees in our ensemble. Equivalent to the number of boosting rounds.

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


print(merged_df['Country'].unique())

#%%--------------------------------------------------------------------------

# Create the lagged values for all variables (GTs only) first
for lag in range(1, max_lag+1):
    for col in df_trends.columns:
        if col not in ['Country', 'Year']:
            # Create a new lagged column
            lagged_col_name = f'{col}_yearly_avg_lag{lag}'
            merged_df[lagged_col_name] = merged_df.groupby('Country')[f'{col}_yearly_avg'].shift(lag)
            # Fill NA values in the new lagged column with zero
            merged_df[lagged_col_name].fillna(0, inplace=True)

    for col in rd_expenditure_df_rev.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
            # Drop rows with NA in 'rd_expenditure' and its lagged columns
            #merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', 'rd_expenditure', 'rd_expenditure_missing'])

# Fill NaN in columns with keyword 'mean_YTD' 
for col in merged_df.columns[merged_df.columns.str.contains('mean_YTD')]:
    merged_df[col].fillna(0, inplace=True)

# Get unique countries from df_trends
unique_countries = df_trends['Country'].unique()

# Limit merged_df to those countries
merged_df = merged_df[merged_df['Country'].isin(unique_countries)]

merged_df[lagged_col_name] = merged_df[lagged_col_name].fillna(0, inplace=True)

# Define groups of columns
cols_fillna_zero = [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year']]
cols_fillna_zero += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']]

cols_dropna = [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]
cols_dropna += ['Year']

# Fill NaN with 0 in specific columns
for col in cols_fillna_zero:
    if col in merged_df.columns:
        merged_df[col].fillna(0, inplace=True)

# Drop rows with NaN in specific columns
for col in cols_dropna:
    if col in merged_df.columns:
        merged_df.dropna(subset=[col], inplace=True)

merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or ('_lag' in col) or ('_mean_YTD' in col)]]

#%%--------------------------------------------------------------------------

# Create a copy of 'Country' column
#X_all['Country_label'] = X_all['Country']

# One-hot encode the 'Country' column
#X_all = pd.get_dummies(X_all, columns=['Country_label'])

# Print the first few rows to verify
print(X_all.head())

#%%--------------------------------------------------------------------------

y = merged_df[['Country', 'Year', 'rd_expenditure']]

#%%--------------------------------------------------------------------------

print(X_all['Country'].unique())
    
#%%--------------------------------------------------------------------------
    


# Loop over the unique countries - country specific training

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, train_size=0.8, test_size=0.2, random_state=123
)

# Loop over the range of lags
for lag in range(1, max_lag+1):
    
    # Initialize the list outside the loop so you can keep adding to it
    relevant_columns = set(['Year', 'Country'])
    
    # Loop over the range of lags
    for cur_lag in range(1, lag+1):

        # Append the lagged columns for the current lag to the set
        relevant_columns.update([f"{col}_yearly_avg_lag{cur_lag}" for col in df_trends.columns if col not in ['Country', 'Year']])
        relevant_columns.update([f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']])
        relevant_columns.update([f"{col}_lag{cur_lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']])

    # After the loop, relevant_columns will have all the columns for lags from 1 to max_lag
    # Convert the set back to a list and filter the training set
    relevant_columns = list(relevant_columns)

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
    #model_coeff.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/model_coeff_{lag}.csv")
    #print(f'Training Set - Lag: {lag}', model_coeff.head(5))

    #print("Shape of coefficients: ", model_coeff.shape)
    print("Number of columns: ", len(X_train_lag.columns))

    #breakpoint()
    
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
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/model_scatter_train_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_train_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/model_residuals_train_{model_name}_{lag}.png")
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
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/model_residuals_test_{model_name}_{lag}.png")
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

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/merged_df_test_RV01_batches_all_{model_name}.csv")

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
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/evaluation_batches_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V3_KWO_CrossCountry_XGBoost/MGT/MGTwRD/evaluation_testAGT1_{model_name}.csv", index=False)


