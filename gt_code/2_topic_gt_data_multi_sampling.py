import time
from pytrends.request import TrendReq
import pandas as pd

# Set up pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# Read the CSV file to get topic IDs and descriptions
file_path = '/Nowcasting/data/GT/topic_metadata_BusinessRnDEcosystem1_merged_df.csv'

topics_df = pd.read_csv(file_path)
topics_df = topics_df[topics_df['type'] == 'Topic']
topics_df = topics_df.drop_duplicates(subset='mid', keep='first')
topic_id_to_description = dict(zip(topics_df['mid'], topics_df['description']))
from pytrends.request import TrendReq
import pandas as pd
import time
from pytrends.exceptions import ResponseError

# Set up PyTrends
pytrends = TrendReq(hl='en-US', tz=360)


# Define list of keywords associated with each stakeholder
keywords = {
    'Businesses' : ['R&D expenditure', 'product development'],
    'VCs' : ['startup funding', 'VC investment'],
    'Financial Institutions' : ['R&D loans', 'R&D financing'],
    'Research Institutions' : ['research grant', 'collaboration with industry'],
    'Government agencies' : ['research funding', 'government grants'],
    'R&D Employees' : ['R&D jobs', 'R&D collaboration tools'],
    'Tax Authorities' : ['R&D tax credit', 'R&D tax incentives'],
    'Consulting Firms' : ['innovation strategy', 'innovation management'],
    'Innovation Hubs' : ['startup incubation', 'technology park'],
    'Patent Attorneys' : ['patent registration', 'patent attorney'],
    'Tax Consultants' : ['tax advice for R&D', 'accounting for R&D']
}


# Define selected countries (ISO 3166-1 alpha-2 codes)
countries = ['US', 'CN', 'JP', 'DE', 'KR', 'CA', 'CH', 'GB']


# Create an empty list to store topic metadata
topic_metadata = []

for i, country in enumerate(countries, start=1):
    for j, (stakeholder, keywords_list) in enumerate(keywords.items(), start=1):
        for keyword in keywords_list:
            attempts = 0
            max_attempts = 5  # Set a max number of attempts to avoid infinite loop
            while attempts < max_attempts:
                try:
                    pytrends.build_payload([keyword], cat=0, timeframe='all', geo=country, gprop='')
                    rt = pytrends.related_topics()
                    if rt[keyword]['top'] is not None:
                        related_topics_top = rt[keyword]['top']
                        if 'topic_title' in related_topics_top.columns:
                            for _, row in related_topics_top.iterrows():
                                topic_metadata.append({
                                    'Keyword': keyword,
                                    'Topic Title': row['topic_title'],
                                    'MID': row['topic_mid'],
                                    'Type': row['topic_type']
                                })
                    break  # Exit the loop after successful data retrieval
                except ResponseError:
                    attempts += 1
                    print(f"Attempt {attempts}: Hit the rate limit. Waiting for 60 seconds.")
                    time.sleep(60)

print("Fetching data completed.")
pd.DataFrame(topic_metadata).to_csv('/Nowcasting/data/GT/topic_metadata_BusinessRnDEcosystem1_wkw.csv')

#################################################################################################

# Number of samples to draw
num_samples = 5

# Initialize an empty DataFrame to store the results
final_df = pd.DataFrame()

# Loop over each country and topic ID
for country in countries:
    country_df = pd.DataFrame()
    for topic_id, description in topic_id_to_description.items():
        all_samples = []
        for sample_num in range(num_samples):
            while True:
                try:
                    pytrends.build_payload([topic_id], cat=0, timeframe='all', geo=country, gprop='')
                    trends_data = pytrends.interest_over_time()
                    if not trends_data.empty:
                        trends_data = trends_data.drop(columns=['isPartial'])
                        column_name = f'{country}_{description.replace(" ", "_")}_sample{sample_num+1}'
                        trends_data.rename(columns={topic_id: column_name}, inplace=True)
                        all_samples.append(trends_data)

                    print(f'Finished fetching data for {description} ({topic_id}) in {country}, sample {sample_num+1}.')
                    break  # Break the while loop if the request is successful

                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Sleeping for 60 seconds before retrying...")
                    time.sleep(60)  # Delay for 60 seconds
                    continue  # Retry the failed request

            # Sleep to avoid hitting the rate limit
            time.sleep(10)  # Delay for 10 seconds

        # Average the samples
        if all_samples:
            avg_df = pd.concat(all_samples, axis=1).mean(axis=1)
            avg_df = pd.DataFrame(avg_df, columns=[f'{country}_{description.replace(" ", "_")}_average'])
            all_samples.append(avg_df)

            # Concatenate all samples and the average
            combined_df = pd.concat(all_samples, axis=1)
            country_df = pd.concat([country_df, combined_df], axis=1)
            final_df = pd.concat([final_df, combined_df], axis=1)

    # Save the country-specific DataFrame to a CSV file
    country_output_csv_path = f'/Nowcasting/data/GT/December/trends_data_by_topic_{country}.csv'
    country_df.to_csv(country_output_csv_path)
    print(f"Data for {country} saved to {country_output_csv_path}")

print("Fetching data completed.")

# Save the final DataFrame to a combined CSV file
output_csv_path = '/Nowcasting/data/GT/December/trends_data_by_topic_multi_sampling.csv'
final_df.to_csv(output_csv_path)

print(f"Combined data saved to {output_csv_path}")

#################################################################################################
# Read the CSV file
file_path = '/Nowcasting/data/GT/December/trends_data_by_topic_multi_sampling.csv'
df = pd.read_csv(file_path)

# Keep only the columns with '_average' suffix and the 'date' column
columns_to_keep = ['date'] + [col for col in df.columns if col.endswith('_average')]

# Select the relevant columns
df_filtered = df[columns_to_keep]

# Rename the columns by dropping the '_average' suffix
df_filtered.columns = ['date'] + [col.replace('_average', '') for col in df_filtered.columns if col != 'date']

# Save the result to a new CSV file
output_file_path = '/Users/atin/Nowcasting/data/GT/December/trends_data_by_topic.csv'
df_filtered.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")

#################################################################################################

# Load the data
df = pd.read_csv('/Nowcasting/data/GT/trends_data_by_topic.csv', index_col=0) 

# Ensure the index is a datetime index
df.index = pd.to_datetime(df.index)

# Resample to yearly frequency, aggregating with mean
df_yearly = df.resample('Y').mean()

df_yearly.reset_index(inplace=True)
df_yearly['Year'] = pd.to_datetime(df_yearly['date']).dt.year
df_yearly.set_index('Year', inplace=True)
df_yearly.drop(columns=['date'], inplace=True)   

# Save the yearly data
df_yearly.to_csv('/Nowcasting/data/GT/December/trends_data_by_topic_resampled.csv')

#################################################################################################

# Ensure the index is a datetime index
df.index = pd.to_datetime(df.index)

# Resample to yearly frequency, aggregating with sum
df_yearly = df.resample('Y').sum()

df_yearly.reset_index(inplace=True)
df_yearly['Year'] = pd.to_datetime(df_yearly['date']).dt.year
df_yearly.set_index('Year', inplace=True)
df_yearly.drop(columns=['date'], inplace=True)   

# Save the yearly data
df_yearly.to_csv('/Nowcasting/data/GT/December/trends_data_by_topic_resampled_sum.csv')

