import pandas as pd

# Load the topic metadata CSV file
df_topic_metadata = pd.read_csv('/Nowcasting/data/GT/topic_metadata_BusinessRnDEcosystem1_merged_df.csv')

# Filter rows where Binary equals 1
filtered_topic_metadata = df_topic_metadata[df_topic_metadata['Binary'] == 1]

filtered_topic_metadata = filtered_topic_metadata.drop(columns=['Topic Title','MID','Type'])

# Get the list of descriptions from the filtered metadata
descriptions = filtered_topic_metadata['description'].tolist()

##########################################################################################################################

# Load your df_trends and df_trends_monthly dataframes
df_trends_sum= pd.read_csv('/Nowcasting/data/GT/trends_data_by_topic_resampled_sum.csv')
df_trends= pd.read_csv('/Nowcasting/data/GT/trends_data_by_topic_resampled.csv')
df_trends_monthly = pd.read_csv('/Nowcasting/data/GT/trends_data_by_topic.csv')

df_trends_sum = df_trends_sum[df_trends_sum.columns.drop(list(df_trends_sum.filter(regex='isPartial')))]

# Yearly G-Trend data
df_trends = df_trends[df_trends.columns.drop(list(df_trends.filter(regex='isPartial')))]

# Monthly G-Trend data
df_trends_monthly = df_trends_monthly[df_trends_monthly.columns.drop(list(df_trends_monthly.filter(regex='isPartial')))]


##########################################################################################################################

# Function to check if any description is a substring of the column name
def filter_columns(column_name, descriptions):
    # Split the column name by '_' and check if any part matches the descriptions
    return any(desc in column_name.split('_') for desc in descriptions)

# Filter columns in df_trends based on descriptions, and keep 'Year'
df_trends_filtered_sum = df_trends_sum[['Year'] + [col for col in df_trends_sum.columns if filter_columns(col, descriptions)]]

# Filter columns in df_trends based on descriptions, and keep 'Year'
df_trends_filtered = df_trends[['Year'] + [col for col in df_trends.columns if filter_columns(col, descriptions)]]

# Filter columns in df_trends_monthly based on descriptions, and keep 'date'
df_trends_monthly_filtered = df_trends_monthly[['date'] + [col for col in df_trends_monthly.columns if filter_columns(col, descriptions)]]

# Now, df_trends_filtered and df_trends_monthly_filtered have columns 
# that match the descriptions in the filtered topic metadata, along with 'Year' and 'date'

# Save the filtered df_trends to a CSV file
csv_path_trends_sum_filtered = '/Nowcasting/data/GT/trends_data_by_topic_resampled_sum_filtered.csv'
df_trends_filtered_sum.to_csv(csv_path_trends_sum_filtered, index=False)

# Save the filtered df_trends to a CSV file
csv_path_trends_filtered = '/Nowcasting/data/GT/trends_data_by_topic_resampled_filtered.csv'
df_trends_filtered.to_csv(csv_path_trends_filtered, index=False)

# Save the filtered df_trends_monthly to a CSV file
csv_path_trends_monthly_filtered = '/Nowcasting/data/GT/trends_data_by_topic_filtered.csv'
df_trends_monthly_filtered.to_csv(csv_path_trends_monthly_filtered, index=False)

print(f"Filtered df_trends saved to {csv_path_trends_filtered}")
print(f"Filtered df_trends_monthly saved to {csv_path_trends_monthly_filtered}")



