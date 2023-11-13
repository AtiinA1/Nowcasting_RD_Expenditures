import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


for col in rd_expenditure_df_rev.columns:
    if col != 'Country':  # Exclude the 'Country' column
        if rd_expenditure_df_rev[col].dtype == 'object':  # If the column is a string
            rd_expenditure_df_rev[col] = rd_expenditure_df_rev[col].str.replace(',', '')  # Remove the commas
            rd_expenditure_df_rev[col] = rd_expenditure_df_rev[col].replace('--', np.nan)  # Replace '--' with NaN
            rd_expenditure_df_rev[col] = rd_expenditure_df_rev[col].astype(float)  # Convert the column to float
            #rd_expenditure_df_rev[col] = pd.to_numeric(rd_expenditure_df_rev[col], errors='coerce')


for col in rd_expenditure_df_rev.columns:
    if col not in ['Country', 'Year', 'Month']:
        rd_expenditure_df_rev = rd_expenditure_df_rev.dropna(subset=['rd_expenditure'])

#########################################################################################################################################################


df_trends= pd.read_csv('/Users/atin/Nowcasting/data/GT/kw_only_approach/trends_data_resampled.csv')
df_trends = df_trends[df_trends.columns.drop(list(df_trends.filter(regex='isPartial')))]

# Melt the DataFrame so that each row is a country-year pair
df_trends = df_trends.reset_index(drop=True).melt(id_vars='Year', var_name='Country_Keyword', value_name='Value')

# Extract the country code and keyword from the "Country_Keyword" column
df_trends[['Country', 'Keyword']] = df_trends['Country_Keyword'].str.split('_', 1, expand=True)

# Pivot the DataFrame so that each keyword becomes a column
df_trends = df_trends.pivot(index=['Year', 'Country'], columns='Keyword', values='Value')

print(df_trends.columns)
# Get unique countries from df_trends
#unique_countries = df_trends['Country'].unique()
unique_countries = df_trends.index.get_level_values('Country').unique()

#########################################################################################################################################################

#rd_expenditure_df_rev = oecd_df_rev[['Country', 'Year', 'rd_expenditure', 'rd_expenditure_missing']]

# Assuming df is your DataFrame
df = rd_expenditure_df_rev

# Limit rd_expenditure_df_rev to those countries
df = df[df['Country'].isin(unique_countries)]

df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_arima/df.csv")

breakpoint()

#########################################################################################################################################################

#ARIMA

results_country = []
all_countries_forecasts = []  # List to store all forecasts

# Group the dataframe by 'Country' and iterate over each group
for country, group in df.groupby('Country'):
    
    # Ensure there's enough data for the country
    if len(group) < 10:  # Adjust this threshold as needed
        print(f"Skipping {country} due to insufficient data.")
        continue

    # Splitting dataset into train and test set for the current country
    train_size = int(len(group) * 0.8)
    train, test = group['rd_expenditure'][0:train_size], group['rd_expenditure'][train_size:]

    # Fit ARIMA model
    model = ARIMA(train, order=(1,1,0))  # Using ARIMA(3,1,0) as an example
    model_fit = model.fit()
    
    # Predict using the ARIMA model
    start_index = len(train)
    end_index = start_index + len(test) - 1
    forecast = model_fit.predict(start=start_index, end=end_index, typ='levels')
    
	    # Calculate RMSE instead of MSE
    # Calculate metrics on the test set
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    mape = np.mean(np.abs((test - forecast) / (test + 1e-10))) * 100
    r2 = r2_score(test, forecast)  # Make sure the indentation here is consistent

    results_country.append({'Country': country, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2})

    forecast = forecast.reset_index(drop=True)  # Reset forecast index
    test = test.reset_index(drop=True)  # Reset test index to align with forecast

    # Store true values and forecasts in a DataFrame
    country_forecast_df = pd.DataFrame({'Country': country, 'True': test, 'Forecast': forecast})
    all_countries_forecasts.append(country_forecast_df)

    # Create scatterplot for test vs forecast
    plt.scatter(test, forecast)
    plt.xlabel('True Values')
    plt.ylabel('Forecasted Values')
    plt.title(f'Test vs Forecast Scatterplot for {country}')
    plt.savefig(f'/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_arima//scatterplot_{country}.png')
    plt.clf()  # Clear the figure for the next plot


# Concatenate all country forecasts and save to CSV
all_forecasts_df = pd.concat(all_countries_forecasts)
all_forecasts_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_arima/true_vs_forecast_all_countries.csv", index=False)

# Convert results to a DataFrame and save to CSV
results_df_country = pd.DataFrame(results_country)
results_df_country.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_arima/arima_results_country.csv", index=False)

#########################################################################################################################################################

results = []

# Select the column for analysis (replace 'rd_expenditure' with your column name)
timeseries_data = df['rd_expenditure']

# Splitting dataset into train and test set
train_size = int(len(timeseries_data) * 0.8)
train, test = timeseries_data[0:train_size], timeseries_data[train_size:]

# Fit ARIMA model (adjust order as per your analysis)
model = ARIMA(train, order=(1,1,0))
model_fit = model.fit()

# Predict using the ARIMA model
start_index = len(train)
end_index = start_index + len(test) - 1
forecast = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Calculate metrics on the test set
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / (test + 1e-10))) * 100
r2 = r2_score(test, forecast)  # Make sure the indentation here is consistent

results.append({'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2})


# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/baseline_arima/arima_results.csv", index=False)


