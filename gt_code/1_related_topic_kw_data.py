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


# Create an empty dictionary to store topic metadata
topic_metadata = {}

for i, country in enumerate(countries, start=1):
    for j, (stakeholder, keywords_list) in enumerate(keywords.items(), start=1):
        for keyword in keywords_list:
            # Handle the rate limit error
            while True:
                try:
                    # Build the payload
                    pytrends.build_payload([keyword], cat=0, timeframe = 'all', geo=country, gprop='')

                    # Get the interest over time data
                    iot_df = pytrends.interest_over_time()

                    # Get related queries
                    rq = pytrends.related_queries()
                    if rq[keyword]['top'] is not None:
                        related_keywords = rq[keyword]['top']['query'].tolist()  # Top related queries
                    else:
                        related_keywords = []

                    # Get related topics
                    rt = pytrends.related_topics()
                    if rt[keyword]['top'] is not None:
                        related_topics_top = rt[keyword]['top']
                        # Store topic metadata
                        if 'topic_title' in related_topics_top.columns:
                            for _, row in related_topics_top.iterrows():
                                topic_metadata[row['topic_title']] = {'mid': row['topic_mid'], 'type': row['topic_type']}
                        # Get topic titles
                        related_topics = related_topics_top['topic_title'].tolist() if 'topic_title' in related_topics_top.columns else []

                    else:
                        related_topics = []

                    # Concatenate all related keywords and topics into a single list
                    all_related = related_keywords + related_topics

                    for related in all_related:
                        # Apply a delay before each request
                        time.sleep(5)
                        pytrends.build_payload([related], cat=0, timeframe='all', geo=country, gprop='')
                        related_df = pytrends.interest_over_time()

                        # Concatenate the country, stakeholder, and related keyword/topic to create a unique column name
                        related_df.rename(columns={related: f'{country}_{stakeholder}: {related}'}, inplace=True)

                        final_df = pd.concat([final_df, related_df], axis=1)

                    print(f'Finished fetching data for {stakeholder} in {country}. ({j}/{len(keywords)} stakeholders, {i}/{len(countries)} countries)')
                    break  # Break the loop if the code executed successfully
                except ResponseError:#pytrends.exceptions.TooManyRequestsError:
                    print("Hit the rate limit. Waiting for 60 seconds.")
                    time.sleep(60)  # Wait for 60 seconds before the next attempt

print("Fetching data completed.")
print(final_df)
print(topic_metadata)

final_df.to_csv('/Nowcasting/data/GT/final_df_BusinessRnDEcosystem1.csv')
pd.DataFrame.from_dict(topic_metadata, orient='index').to_csv('/Nowcasting/data/GT/topic_metadata_BusinessRnDEcosystem1.csv')

