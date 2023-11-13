import pandas as pd

# Load the data
df = pd.read_csv('./final_df_BusinessRnDEcosystem_merged.csv', index_col=0)

# Ensure the index is a datetime index
df.index = pd.to_datetime(df.index)

# Resample to yearly frequency, aggregating with mean
df_yearly = df.resample('Y').mean()

df_yearly.reset_index(inplace=True)
df_yearly['Year'] = pd.to_datetime(df_yearly['date']).dt.year
df_yearly.set_index('Year', inplace=True)
df_yearly.drop(columns=['date'], inplace=True)   

# Save the yearly data
df_yearly.to_csv('./final_df_BusinessRnDEcosystem_merged_resampled.csv')
