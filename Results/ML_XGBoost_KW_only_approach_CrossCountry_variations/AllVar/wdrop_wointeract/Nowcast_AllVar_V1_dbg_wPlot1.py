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
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/df_trends_monthly_YTD.csv')

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

df_trends_monthly_aggregated.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/df_trends_monthly_YTD_YR.csv')
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

merged_df.to_csv('/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/merged_df.csv')

#%%--------------------------------------------------------------------------

#Model 3: Annual GT + Monthly GT (if available)

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

for col in merged_df.columns:
    if col != 'Country':  # Exclude the 'Country' column
        if merged_df[col].dtype == 'object':  # If the column is a string
            merged_df[col] = merged_df[col].str.replace(',', '')  # Remove the commas
            merged_df[col] = merged_df[col].replace('--', np.nan)  # Replace '--' with NaN
            merged_df[col] = merged_df[col].astype(float)  # Convert the column to float
            #merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

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
            #merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', 'rd_expenditure'])

# Fill NaN in columns with keyword 'mean_YTD' 
for col in merged_df.columns[merged_df.columns.str.contains('mean_YTD')]:
    merged_df[col].fillna(0, inplace=True)

# Loop over the range of lags
for lag in range(1, max_lag+1):
    # Create lagged values for macro variables
    for col in oecd_df_rev_4lag.columns:
        if col not in ['Country', 'Year']:
            merged_df[f'{col}_lag{lag}'] = merged_df.groupby('Country')[col].shift(lag)
            print(merged_df.columns) # Debugging line
            #merged_df = merged_df.dropna(subset=[f'{col}_lag{lag}', f'{col}'])


# Get unique countries from df_trends
unique_countries = df_trends['Country'].unique()

# Limit merged_df to those countries
merged_df = merged_df[merged_df['Country'].isin(unique_countries)]

merged_df[lagged_col_name] = merged_df[lagged_col_name].fillna(0, inplace=True)

# Define groups of columns
cols_fillna_zero = [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year']]
cols_fillna_zero += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']]
cols_fillna_zero += [f"{col}_lag{lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']]


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


merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/merged_df_test_RV01_batches_all_{model_name}_0.csv")

merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]

# Define X_all (excluding 'rd_expenditure') and y
X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or ('_lag' in col) or ('_mean_YTD' in col)]]
X_all.fillna(0, inplace=True)


#%%--------------------------------------------------------------------------

# Create a copy of 'Country' column
#X_all['Country_label'] = X_all['Country']

# One-hot encode the 'Country' column
#X_all = pd.get_dummies(X_all, columns=['Country_label'])

# Print the first few rows to verify
print(X_all.head())
print(X_all['Country'].unique())
print(X_all.nunique())
print(X_all.info())

#%%--------------------------------------------------------------------------

y = merged_df[['Country', 'Year', 'rd_expenditure']]

print(y.head())
print(y['Country'].unique())
print(y.nunique())
print(y.info())
    
#%%--------------------------------------------------------------------------

# First, split data into training + validation and test sets (e.g., 80-20 split)
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Then, split the training + validation data into training and validation sets (e.g., 80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


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
        relevant_columns.update([f"{col}_lag{cur_lag}" for col in oecd_df_rev_4lag.columns if col not in ['Country', 'Year']])


    # After the loop, relevant_columns will have all the columns for lags from 1 to max_lag
    # Convert the set back to a list and filter the training set
    relevant_columns = list(relevant_columns)


    # Apply the relevant_columns to the train and test datasets
    X_train_lag = X_train[relevant_columns]
    X_val_lag = X_val[relevant_columns]    
    X_test_lag = X_test[relevant_columns]
    y_train_lag = y_train[['rd_expenditure']]
    y_val_lag = y_val[['rd_expenditure']]
    y_test_lag = y_test[['rd_expenditure']]
    
    # Convert 'Country' column to dummies
    X_train_lag = pd.get_dummies(X_train_lag, columns=['Country'])
    X_val_lag = pd.get_dummies(X_val_lag, columns=['Country'])    
    X_test_lag = pd.get_dummies(X_test_lag, columns=['Country'])

    # Save column names before any operation that changes X_train_lag to a numpy array
    original_columns = X_train_lag.columns.tolist()

    # Adding Polynomial Features
    #poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    #X_train_poly = poly.fit_transform(X_train_lag)
    #X_val_poly = poly.transform(X_val_lag)
    #X_test_poly = poly.transform(X_test_lag)

    #X_train_lag = X_train_poly
    #X_val_lag = X_val_poly
    #X_test_lag = X_test_poly

    #####################

    # Now pass the data to the model
    model_fit = model.fit(X_train_lag, y_train_lag, early_stopping_rounds=10, eval_set=[(X_val_lag, y_val_lag)], verbose=False)

    # Feature Importance
    plot_importance(model_fit, max_num_features=15, importance_type='gain')  # top 105features
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/feature_importance_plt_importance_{lag}.png", dpi=300)    
    plt.close()  # Close the current figure


    feature_names = X_train_lag.columns

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
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/feature_importance_norm_name_{lag}.png", dpi=300)
    plt.close()  # Close the current figure

    # Use model.feature_importances_ directly to filter columns
    importance_threshold = 0.05  # example threshold

    # Filtering important columns
    important_features_mask = model_fit.feature_importances_ > importance_threshold # Create a Boolean Mask for Important Features
    important_cols = [name for i, name in enumerate(feature_names) if important_features_mask[i]]

    # Save this iteration's important columns to a list
    all_important_features.append({"lag": lag, "important_features": ", ".join(important_cols)})
    

    #####################

    # Assuming X_train_lag, X_val_lag, and X_test_lag are numpy arrays:

    # Convert arrays to DataFrames
    X_train_lag_df = pd.DataFrame(X_train_lag, columns=feature_names)
    X_val_lag_df = pd.DataFrame(X_val_lag, columns=feature_names)
    X_test_lag_df = pd.DataFrame(X_test_lag, columns=feature_names)

    # Now, filter the DataFrames using only important columns
    X_train_filtered = X_train_lag_df[important_cols]
    X_val_filtered = X_val_lag_df[important_cols]
    X_test_filtered = X_test_lag_df[important_cols]

    X_train_lag = X_train_filtered
    X_val_lag = X_val_filtered
    X_test_lag = X_test_filtered    

    # Retrain the model using the filtered training set
    model_fit = model.fit(X_train_lag, y_train_lag, early_stopping_rounds=10, eval_set=[(X_val_lag, y_val_lag)], verbose=False)

    #####################

    # Plot the first tree
    xgboost.plot_tree(model_fit, num_trees=0)
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/first_tree_{lag}.png", dpi=300)
    #plt.show()

    # Plot the second tree
    xgboost.plot_tree(model_fit, num_trees=1)
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/second_tree_{lag}.png", dpi=300)
    #plt.show()

    # Plot the third tree
    xgboost.plot_tree(model_fit, num_trees=2)
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/third_tree_{lag}.png", dpi=300)
    #plt.show()

    # Plot the fourth tree
    xgboost.plot_tree(model_fit, num_trees=3)
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/fourth_tree_{lag}.png", dpi=300)
    #plt.show()

    # Plot the fifth tree
    xgboost.plot_tree(model_fit, num_trees=4)
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/fifth_tree_{lag}.png", dpi=300)
    #plt.show()

    # Plot the fifth tree
    xgboost.plot_tree(model_fit, num_trees=99)
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/last_tree_{lag}.png", dpi=300)        

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
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/model_residuals_train_{model_name}_{lag}.png")
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

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.scatter(y_test_lag['rd_expenditure'], y_test_lag['y_pred'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/model_scatter_test_{model_name}_{lag}.png")
    #plt.show()

    sns.set_theme(color_codes=True)
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    plt.hist(y_test_lag['residuals'])
    plt.savefig(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/model_residuals_test_{model_name}_{lag}.png")
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

merged_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/merged_df_test_RV01_batches_all_{model_name}.csv")

# Convert all_important_features to a DataFrame and save
df_all_important_features = pd.DataFrame(all_important_features)
df_all_important_features.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/all_important_features_by_lag_{model_name}.csv", index=False)

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
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/evaluation_train_{model_name}.csv", index=False)
evaluation_test_df.to_csv(f"/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/evaluation_test_{model_name}.csv", index=False)

