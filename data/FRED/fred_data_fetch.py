from fredapi import Fred
import pandas as pd

# Initialize the API with your key
fred = Fred(api_key='3f8f5bfac0b4b96244183fe0f2fd6616')

# Fetch GDP per capita (series ID is GDPCA)
gdp_per_capita = fred.get_series('GDPCA')

gdp_per_capita.to_csv('./data/gdp_per_capita.csv')

breakpoint()

# Resample to yearly frequency, aggregating with mean
gdp_per_capita_yearly = gdp_per_capita.resample('Y').mean()

# Save the yearly data
gdp_per_capita_yearly.to_csv('./data/GDP_per_capita_yearly.csv')

#----------------------------------------------------------------
