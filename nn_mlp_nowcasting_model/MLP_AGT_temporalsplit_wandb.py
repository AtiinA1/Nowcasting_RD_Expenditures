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
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

# Initialize wandb
def init_wandb(config=None):
    # Default configuration (ultra-aggressive training for small dataset)
    default_config = {
        'learning_rate': 0.01,          # Start high
        'batch_size': 32,                # Even smaller batch  
        'hidden1_dim': 200,              # Simpler architecture
        'hidden2_dim': 20,              # Simpler architecture  
        'hidden3_dim': 20,               # Simpler architecture
        'embedding_dim': 4,             # Small embedding
        'size_ensemble': 3,             # Small ensemble
        'num_epochs': 18000,             # EXTREME training time!
        'patience': 8000,               # MASSIVE patience (33% of epochs)!
        'lr_milestone': 300,            # Very late milestone
        'lr_gamma': 0.1,                # Decay to 0.001 at epoch 800
        'optimizer': 'adamw',
        'weight_decay': 0.0001,         # Light regularization
        'dropout_rate': 0.1             # Light dropout
    }
    
    # Use provided config or defaults
    if config is None:
        config = default_config
    else:
        # Fill in any missing values with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

    # Allow overriding config via environment variables (for sweep scripts)
    import json as _json
    fixed_path = os.getenv("FIXED_CONFIG_PATH")
    fixed_json = os.getenv("FIXED_CONFIG_JSON")
    try:
        if fixed_path and os.path.exists(fixed_path):
            with open(fixed_path, "r") as f:
                config.update(_json.load(f))
        elif fixed_json:
            config.update(_json.loads(fixed_json))
    except Exception as e:
        print(f"[WARN] Failed to load fixed config override: {e}")
    
    # Get wandb entity from environment variable or use None
    entity = os.getenv('WANDB_ENTITY', None)
    
    wandb.init(
        project="nowcasting-rd-mlp",
        entity=entity,
        config=config,
        tags=["AGT", "temporal-split", "hyperparameter-sweep"]
    )
    return wandb.config

#---------------------------------------------------------------------------------------------------------------------------
## Pre-Processing Macroeconomic Variables + Target Variable (GERD in billion dollars)
#---------------------------------------------------------------------------------------------------------------------------

def load_and_preprocess_data():
    """Load and preprocess all the data"""
    
    country_cd_df = pd.read_csv('/Users/atin/Nowcasting/Nowcasting_github/data/country_code.csv')

    # Read in and filter rd_expenditure_df to only include rows with sector of performance equal to Business enterprise + considering total fundings (millions-LOCAL CURRENCY)
    rd_expenditure_df = pd.read_csv('/Users/atin/Nowcasting/Nowcasting_github/data/GERD/DP_LIVE_08052023154811337.csv')
    rd_expenditure_df = rd_expenditure_df.rename(columns={'Value': 'rd_expenditure'})  
    rd_expenditure_df = rd_expenditure_df[rd_expenditure_df['MEASURE'] == 'MLN_USD'] #USD constant prices using 2015 base year 
    print(rd_expenditure_df.columns)

    rd_expenditure_df['rd_expenditure'] = rd_expenditure_df['rd_expenditure'] / 1000

    # Read and filter n_patents
    n_patents_df = pd.read_csv('/Users/atin/Nowcasting/Nowcasting_github/data/OECD/PATS_IPC_11062023234902217.csv')
    n_patents_df = n_patents_df[n_patents_df['IPC'] == 'TOTAL']
    n_patents_df = n_patents_df[n_patents_df['KINDDATE'] == 'PRIORITY']
    n_patents_df = n_patents_df[n_patents_df['KINDCOUNTRY'] == 'INVENTORS']
    print(n_patents_df.columns)
    n_patents_df = n_patents_df[['LOCATION','TIME','Value']]
    n_patents_df = n_patents_df.rename(columns={'Value': 'n_patents'})
    print(n_patents_df.columns)

    # Read and filter IMF (various macro variables)
    imf_df = pd.read_csv('/Users/atin/Nowcasting/Nowcasting_github/data/IMF/WEOApr2023all.csv')

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
    rd_expenditure_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'rd_expenditure_missing']]

    print(rd_expenditure_df_rev.head(5))

    #--------------------------------------------------------------------------------------------------------------------------
    ## Pre-Processing Google Trends (GT)
    #--------------------------------------------------------------------------------------------------------------------------

    df_trends= pd.read_csv('/Users/atin/Nowcasting/Nowcasting_github/data/GT/trends_data_by_topic_resampled_filtered.csv')
    df_trends_monthly = pd.read_csv('/Users/atin/Nowcasting/Nowcasting_github/data/GT/trends_data_by_topic_filtered.csv')

    print(df_trends.columns)
    print(df_trends_monthly.columns)

    # Yearly G-Trend data
    df_trends = df_trends[df_trends.columns.drop(list(df_trends.filter(regex='isPartial')))]

    # Melt the DataFrame so that each row is a country-year pair
    df_trends = df_trends.reset_index(drop=True).melt(id_vars='Year', var_name='Country_Keyword', value_name='Value')

    # Extract the country code and keyword from the "Country_Keyword" column
    df_trends[['Country', 'Keyword']] = df_trends['Country_Keyword'].str.split('_', n=1, expand=True)

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
    df_trends_monthly[['Country', 'Keyword']] = df_trends_monthly['Country_Keyword'].str.split('_', n=1, expand=True)

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

    for col in merged_df.columns:
        if col != 'Country':  # Exclude the 'Country' column
            if merged_df[col].dtype == 'object':  # If the column is a string
                merged_df[col] = merged_df[col].str.replace(',', '')  # Remove the commas
                merged_df[col] = merged_df[col].replace('--', np.nan)  # Replace '--' with NaN
                merged_df[col] = merged_df[col].astype(float)  # Convert the column to float

    print(merged_df['Country'].unique())

    # Ensure the DataFrame is sorted by Country, Year, and then by Month
    merged_df.sort_values(by=['Country', 'Year', 'Month'], inplace=True)

    return merged_df, df_trends, rd_expenditure_df_rev, oecd_df_rev_4lag

def preprocess_merged_data(merged_df, df_trends, rd_expenditure_df_rev, oecd_df_rev_4lag):
    """Preprocess the merged data with lagged features"""
    
    # Define a range of lags to consider
    max_lag = 3  # considering up to 3 years of lagged values

    # Apply linear interpolation only where 'rd_expenditure_missing' indicates missing data
    def conditional_interpolate(group):
        # Only interpolate where 'rd_expenditure_missing' equals 1
        mask = group['rd_expenditure_missing'] == 1
        group.loc[mask, 'rd_expenditure'] = group.loc[mask, 'rd_expenditure'].interpolate()
        return group

    # Apply the conditional interpolation to the 'rd_expenditure' column
    merged_df = merged_df.groupby('Country').apply(conditional_interpolate).reset_index(drop=True)

    # Create the lagged values for all variables (GTs only) first
    for lag in range(1, max_lag+1):
        for col in df_trends.columns:
            if col not in ['Country', 'Year', 'Month']:
                # Create a new lagged column
                lagged_col_name = f'{col}_yearly_avg_lag{lag}'
                merged_df[lagged_col_name] = merged_df.groupby('Country')[f'{col}_yearly_avg'].shift(12 * lag)
                # Fill NA values in the new lagged column with zero
                merged_df[lagged_col_name].fillna(0, inplace=True)

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

    # Define groups of columns
    cols_fillna_zero = [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year']]
    cols_fillna_zero += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year']]

    cols_dropna = [f"{col}_lag{lag}" for col in rd_expenditure_df_rev.columns if col not in ['Country', 'Year']]
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

    merged_df = merged_df[merged_df['rd_expenditure_missing'] == 0]
    merged_df = merged_df[(merged_df['Year'] >= 2004)]

    return merged_df, max_lag

def create_train_test_splits(merged_df, df_trends, rd_expenditure_df_rev, max_lag):
    """Create temporal train/validation/test splits"""
    
    # Define X_all (excluding 'rd_expenditure') and y
    X_all = merged_df[[col for col in merged_df.columns if (col == 'Country') or (col == 'Year') or (col == 'Month')  or ('_lag' in col) or ('_mean_YTD' in col)]]
    X_all.fillna(0, inplace=True)

    y = merged_df[['Country', 'Year', 'Month', 'rd_expenditure']]

    # Get unique countries
    countries = X_all['Country'].unique()

    # Initialize empty DataFrames for the splits
    X_train_combined = pd.DataFrame()
    X_val_combined = pd.DataFrame()
    X_test_combined = pd.DataFrame()
    y_train_combined = pd.Series()
    y_val_combined = pd.Series()
    y_test_combined = pd.Series()

    # Define split ratios
    train_ratio = 0.64  # 64% for training
    val_ratio = 0.16    # 16% for validation
    test_ratio = 0.20   # 20% for test

    print(f"Temporal split ratios - Train: {train_ratio:.0%}, Validation: {val_ratio:.0%}, Test: {test_ratio:.0%}")

    # Split the data temporally for each country
    for country in countries:
        # Filter both X and y for the current country
        country_mask = X_all['Country'] == country
        X_country = X_all[country_mask].copy()
        y_country = y[country_mask].copy()
        
        # Sort by Year and Month to ensure temporal order
        sort_order = X_country.sort_values(['Year', 'Month']).index
        X_country = X_country.loc[sort_order].reset_index(drop=True)
        y_country = y_country.loc[sort_order].reset_index(drop=True)
        
        # Calculate split indices
        n_samples = len(X_country)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Create temporal splits
        X_train = X_country.iloc[:train_end]
        X_val = X_country.iloc[train_end:val_end]
        X_test = X_country.iloc[val_end:]
        
        y_train = y_country.iloc[:train_end]
        y_val = y_country.iloc[train_end:val_end]
        y_test = y_country.iloc[val_end:]
        
        # Combine the splits
        X_train_combined = pd.concat([X_train_combined, X_train])
        X_val_combined = pd.concat([X_val_combined, X_val])
        X_test_combined = pd.concat([X_test_combined, X_test])
        y_train_combined = pd.concat([y_train_combined, y_train])
        y_val_combined = pd.concat([y_val_combined, y_val])
        y_test_combined = pd.concat([y_test_combined, y_test])

    # The final combined datasets
    X_train = X_train_combined.reset_index(drop=True)
    X_val = X_val_combined.reset_index(drop=True)
    X_test = X_test_combined.reset_index(drop=True)
    y_train = y_train_combined.reset_index(drop=True)
    y_val = y_val_combined.reset_index(drop=True)
    y_test = y_test_combined.reset_index(drop=True)

    print(f"\nFinal dataset sizes:")
    print(f"Training: {len(X_train)} samples ({len(X_train)/len(X_all):.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X_all):.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(X_all):.1%})")
    print(f"Total: {len(X_all)} samples")

    # Create relevant columns for feature selection
    relevant_columns = ['Year', 'Month','Country']
    relevant_columns += [f"{col}_mean_YTD" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]

    # Loop over the range of lags
    for lag in range(1, max_lag+1):
        # Append the lagged columns for the current lag to the list
        relevant_columns += [f"{col}_yearly_avg_lag{lag}" for col in df_trends.columns if col not in ['Country', 'Year', 'Month']]

    # Apply the relevant_columns to the train and test datasets
    X_train = X_train[relevant_columns].drop_duplicates()
    X_val = X_val[relevant_columns].drop_duplicates()
    X_test = X_test[relevant_columns].drop_duplicates()

    # Filter y datasets based on the indices of the X datasets to maintain alignment
    y_train = y_train.loc[X_train.index][['rd_expenditure']]
    y_val = y_val.loc[X_val.index][['rd_expenditure']]
    y_test = y_test.loc[X_test.index][['rd_expenditure']]

    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_neural_network_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Prepare data for neural network training"""
    
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # One-hot encode 'Month' for the training data
    month_encoded_train = pd.get_dummies(X_train['Month'], prefix='Month')

    # Align the validation and test data with the columns of the training data
    month_encoded_val = pd.get_dummies(X_val['Month'], prefix='Month').reindex(columns=month_encoded_train.columns, fill_value=0)
    month_encoded_test = pd.get_dummies(X_test['Month'], prefix='Month').reindex(columns=month_encoded_train.columns, fill_value=0)

    # Concatenate the one-hot encoded columns with the original data
    X_train = pd.concat([X_train, month_encoded_train], axis=1)
    X_val = pd.concat([X_val, month_encoded_val], axis=1)
    X_test = pd.concat([X_test, month_encoded_test], axis=1)

    # Encode the 'Country' column
    le = LabelEncoder()
    X_train['country_encoded'] = le.fit_transform(X_train['Country'])
    X_val['country_encoded'] = le.transform(X_val['Country'])
    X_test['country_encoded'] = le.transform(X_test['Country'])

    num_countries = X_train['Country'].nunique()

    # Drop the original 'Country' column
    X_train = X_train.drop(columns=['Country'])
    X_val = X_val.drop(columns=['Country'])
    X_test = X_test.drop(columns=['Country'])

    # Standardization of Input for Neural Networks Model
    excluded_columns = ['country_encoded', 'Year', 'Month']

    # Separate the features to be standardized and the excluded columns
    X_train_continuous = X_train.drop(columns=excluded_columns)
    X_val_continuous = X_val.drop(columns=excluded_columns)
    X_test_continuous = X_test.drop(columns=excluded_columns)

    # Apply the standardization
    X_scaler = StandardScaler()

    # Fit the scaler using the training data and transform
    X_train_continuous_standardized = X_scaler.fit_transform(X_train_continuous)
    X_val_continuous_standardized = X_scaler.transform(X_val_continuous)
    X_test_continuous_standardized = X_scaler.transform(X_test_continuous)

    # Merge them back together
    X_train_standardized = np.hstack((X_train_continuous_standardized, 
                                      X_train[excluded_columns].values))
    X_val_standardized = np.hstack((X_val_continuous_standardized, 
                                    X_val[excluded_columns].values))
    X_test_standardized = np.hstack((X_test_continuous_standardized, 
                                     X_test[excluded_columns].values))

    # Convert back to DataFrames
    X_train_cols = X_train_continuous.columns.tolist() + excluded_columns
    X_val_cols = X_val_continuous.columns.tolist() + excluded_columns
    X_test_cols = X_test_continuous.columns.tolist() + excluded_columns

    X_train_standardized = pd.DataFrame(X_train_standardized, columns=X_train_cols)
    X_val_standardized = pd.DataFrame(X_val_standardized, columns=X_val_cols)
    X_test_standardized = pd.DataFrame(X_test_standardized, columns=X_test_cols)

    return X_train_standardized, X_val_standardized, X_test_standardized, y_train, y_val, y_test, num_countries, le

def create_tensors(X_train, X_val, X_test, y_train, y_val, y_test):
    """Create PyTorch tensors"""
    
    # Define the columns to be excluded from the model input
    excluded_columns = ['country_encoded', 'Year', 'Month']
    feature_columns = [col for col in X_train.columns if col not in excluded_columns]

    # For the training set
    X_train_tensor = torch.FloatTensor(X_train[feature_columns].values)
    country_indices_train = torch.LongTensor(X_train['country_encoded'].values)
    y_train_tensor = torch.FloatTensor(y_train.values)

    # For the validation set
    X_val_tensor = torch.FloatTensor(X_val[feature_columns].values)
    country_indices_val = torch.LongTensor(X_val['country_encoded'].values)
    y_val_tensor = torch.FloatTensor(y_val.values)

    # For the test set
    X_test_tensor = torch.FloatTensor(X_test[feature_columns].values)
    country_indices_test = torch.LongTensor(X_test['country_encoded'].values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    return (X_train_tensor, X_val_tensor, X_test_tensor, 
            country_indices_train, country_indices_val, country_indices_test,
            y_train_tensor, y_val_tensor, y_test_tensor, 
            len(feature_columns))

# Define the MLP Model with Variable Architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_countries, embedding_dim, dropout_rate=0.0):
        super(MLP, self).__init__()
        
        # Embedding for countries
        self.country_embedding = nn.Embedding(num_embeddings=num_countries, embedding_dim=embedding_dim)
        
        # Store architecture info
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        
        # Build dynamic layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input dimension includes original features + country embedding
        current_dim = input_dim + embedding_dim
        
        # Create hidden layers dynamically
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim, bias=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim, bias=True)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x, country_indices):
        country_embeds = self.country_embedding(country_indices)
        
        # Concatenate the country embeddings with the input
        x = torch.cat([x, country_embeds], dim=1)
        
        # Pass through hidden layers
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Output layer (no activation/dropout)
        x = self.output_layer(x)
        return x



# Define the Ensemble Neural Network
class Ensemble:
    def __init__(self, model_class, input_dim, hidden_dims, output_dim, size_ensemble, num_countries, embedding_dim, dropout_rate=0.0):
        self.models = []
        for i in range(size_ensemble):
            torch.manual_seed(i)
            self.models.append(model_class(input_dim, hidden_dims, output_dim, num_countries, embedding_dim, dropout_rate))

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
        """Calculate R-squared"""
        ss_total = torch.sum((labels - torch.mean(labels)) ** 2)
        ss_res = torch.sum((labels - outputs) ** 2)
        r2 = 1 - ss_res / ss_total
        return r2        

    def train(self, X_train, y_train, X_val, y_val, country_indices_train, country_indices_val, config):
        
        criterion = nn.MSELoss()
        
        # Get optimizer class from config
        if config.optimizer == "adam":
            optimizer_class = optim.Adam
        elif config.optimizer == "adamw":
            optimizer_class = optim.AdamW
        elif config.optimizer == "sgd":
            optimizer_class = optim.SGD
        elif config.optimizer == "rmsprop":
            optimizer_class = optim.RMSprop
        else:
            optimizer_class = optim.Adam

        model_rmse_train = []
        model_rmse_val = []
        model_mae_train = []
        model_mae_val = []
        model_mape_train = []
        model_mape_val = []

        self.train_losses = [[] for _ in range(len(self.models))]
        self.val_losses = [[] for _ in range(len(self.models))]

        for idx, model in enumerate(self.models):
            model.train()
            
            # Get weight_decay from config, default to 0 if not specified
            weight_decay = getattr(config, 'weight_decay', 0.0)
            optimizer = optimizer_class(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)

            # Learning rate scheduler
            scheduler = MultiStepLR(optimizer, milestones=[config.lr_milestone], gamma=config.lr_gamma)

            best_val_loss = float('inf')
            no_improve_count = 0

            for epoch in range(config.num_epochs):
                train_loss_epoch = 0.0
                num_batches = 0

                for i in range(0, len(X_train), config.batch_size):
                    batch_X = X_train[i:i+config.batch_size]
                    batch_country_indices = country_indices_train[i:i+config.batch_size]
                    batch_y = y_train[i:i+config.batch_size]

                    outputs = model(batch_X, batch_country_indices)
                    loss = criterion(outputs, batch_y)

                    self.train_losses[idx].append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss_epoch += loss.item()
                    num_batches += 1

                    scheduler.step()

                # Validation
                with torch.no_grad():
                    model.eval()
                    val_outputs = model(X_val, country_indices_val)
                    val_loss = criterion(val_outputs, y_val)
                    self.val_losses[idx].append(val_loss.item())

                    # Log metrics to wandb
                    if idx == 0:  # Only log once per epoch
                        wandb.log({
                            'epoch': epoch,
                            'train_loss': train_loss_epoch / num_batches,
                            'val_loss': val_loss.item(),
                            'model_idx': idx
                        })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if no_improve_count == config.patience:
                        print(f"Model {idx+1} early stopped after {epoch} epochs")
                        break

                model.train()

            # After training, compute metrics
            with torch.no_grad():
                model.eval()
                
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

        # Ensemble validation metrics
        ensemble_val_predictions = self.predict(X_val, country_indices_val)
        ensemble_val_rmse = self.rmse(ensemble_val_predictions, y_val).item()
        ensemble_val_mae = self.mae(ensemble_val_predictions, y_val).item()
        ensemble_val_mape = self.mape(ensemble_val_predictions, y_val).item()

        # Log final ensemble metrics to wandb
        wandb.log({
            'final_ensemble_val_rmse': ensemble_val_rmse,
            'final_ensemble_val_mae': ensemble_val_mae,
            'final_ensemble_val_mape': ensemble_val_mape,
            'final_avg_train_rmse': np.mean(model_rmse_train),
            'final_avg_val_rmse': np.mean(model_rmse_val)
        })

        return model_rmse_train, model_rmse_val, model_mae_train, model_mae_val, model_mape_train, model_mape_val, ensemble_val_rmse, ensemble_val_mae, ensemble_val_mape

    def predict(self, X, country_indices):
        with torch.no_grad():
            for model in self.models:
                model.eval()
            
            predictions = [model(X, country_indices) for model in self.models]
            mean_predictions = torch.mean(torch.stack(predictions), dim=0)
            
            return mean_predictions

def adjusted_r_squared(outputs, labels, input_dim):
    """Calculate adjusted R-squared"""
    ss_res = torch.sum((labels - outputs) ** 2)
    ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = labels.size(0)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - input_dim - 1)
    return adj_r2

def main():
    """Main training function for wandb sweep"""
    
    try:
        # Initialize wandb with default config
        config = init_wandb()
        
        print(f"Running with config: {config}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        merged_df, df_trends, rd_expenditure_df_rev, oecd_df_rev_4lag = load_and_preprocess_data()
        merged_df, max_lag = preprocess_merged_data(merged_df, df_trends, rd_expenditure_df_rev, oecd_df_rev_4lag)
        
        # Create train/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_splits(merged_df, df_trends, rd_expenditure_df_rev, max_lag)
        
        # Prepare neural network data
        X_train, X_val, X_test, y_train, y_val, y_test, num_countries, le = prepare_neural_network_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Create tensors
        (X_train_tensor, X_val_tensor, X_test_tensor, 
         country_indices_train, country_indices_val, country_indices_test,
         y_train_tensor, y_val_tensor, y_test_tensor, 
         base_input_dim) = create_tensors(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # The MLP class will add embedding dimension internally
        input_dim = base_input_dim  # DON'T add embedding_dim here!
        
        print(f"Base input dimension: {input_dim}")
        print(f"Embedding dimension: {config.embedding_dim}")
        print(f"Total input dimension (after embedding): {input_dim + config.embedding_dim}")
        print(f"Number of countries: {num_countries}")
        
        # Create ensemble with dynamic architecture
        dropout_rate = getattr(config, 'dropout_rate', 0.0)
        
        # Build hidden dimensions list based on config
        hidden_dims = []
        if hasattr(config, 'hidden1_dim'):
            hidden_dims.append(config.hidden1_dim)
        if hasattr(config, 'hidden2_dim') and config.hidden2_dim > 0:
            hidden_dims.append(config.hidden2_dim)
        if hasattr(config, 'hidden3_dim') and config.hidden3_dim > 0:
            hidden_dims.append(config.hidden3_dim)
        
        # Fallback for new format
        if hasattr(config, 'hidden_dims'):
            hidden_dims = config.hidden_dims
            
        print(f"Network architecture: {len(hidden_dims)} layers with sizes {hidden_dims}")
        
        ensemble = Ensemble(
            MLP, 
            input_dim,  # Pass base_input_dim, MLP will add embedding internally
            hidden_dims, 
            1,  # output_dim
            config.size_ensemble, 
            num_countries, 
            config.embedding_dim,
            dropout_rate
        )
        
        # Train ensemble
        model_rmse_train, model_rmse_val, model_mae_train, model_mae_val, model_mape_train, model_mape_val, ensemble_val_rmse, ensemble_val_mae, ensemble_val_mape = ensemble.train(
            X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
            country_indices_train, country_indices_val, config
        )
        
        # Test set evaluation
        test_predictions = ensemble.predict(X_test_tensor, country_indices_test)
        test_rmse = ensemble.rmse(test_predictions, y_test_tensor).item()
        test_mae = ensemble.mae(test_predictions, y_test_tensor).item()
        test_mape = ensemble.mape(test_predictions, y_test_tensor).item()
        test_r2 = adjusted_r_squared(test_predictions, y_test_tensor, input_dim + config.embedding_dim).item()
        
        # Log final test metrics
        wandb.log({
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'test_r2': test_r2
        })
        
        print(f"Final Test RMSE: {test_rmse:.4f}")
        print(f"Final Test MAE: {test_mae:.4f}")
        print(f"Final Test MAPE: {test_mape:.4f}")
        print(f"Final Test R²: {test_r2:.4f}")
        
        # Save model if this is the best run
        wandb.save("model.pt")
        
    except Exception as e:
        print(f"Error during training: {e}")
        wandb.log({"error": str(e)})
        raise e
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 