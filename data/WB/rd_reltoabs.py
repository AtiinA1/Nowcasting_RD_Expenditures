import pandas as pd


country_cd_df = pd.read_csv('/Users/atin/Nowcasting/data/country_code.csv')


# Read in relative rd_expenditure (%GDP)
df = pd.read_csv('/Users/atin/Nowcasting/data/WB/API_GB.XPD.RSDV.GD.ZS_DS2_en_csv_v2_5551515.csv')

# First, melt the DataFrame to have years as a column
df_melted = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', value_name='rd_expenditure_ratio')

# Make sure the year column is of integer data type
df_melted['Year'] = df_melted['Year'].astype(int)
df_melted = df_melted[['Country Code','Year','rd_expenditure_ratio']]
df_melted = df_melted.rename(columns={'Year': 'TIME', 'Country Code':'LOCATION'})   


# Read and filter IMF
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


gdp_df = imf_df_rev[imf_df_rev['WEO Subject Code'] == 'PPPGDP'] #GDP, current prices, PPP; international dollars
gdp_df = gdp_df[['ISO','TIME','Value']]
gdp_df = gdp_df.rename(columns={'Value': 'gdp', 'ISO':'LOCATION'})   
# Convert Year column in gdp_df to integer
gdp_df['TIME'] = gdp_df['TIME'].astype(int)

# Merge both DataFrames on country and year
merged_df = pd.merge(df_melted, gdp_df,  how='left', 
                     left_on=['LOCATION','TIME'], 
                     right_on = ['LOCATION','TIME'])


merged_df['rd_expenditure_ratio'] = merged_df['rd_expenditure_ratio'].astype(str).str.replace(',', '')
merged_df['gdp'] = merged_df['gdp'].astype(str).str.replace(',', '')


merged_df['rd_expenditure_ratio'] = pd.to_numeric(merged_df['rd_expenditure_ratio'], errors='coerce')
merged_df['gdp'] = pd.to_numeric(merged_df['gdp'], errors='coerce')


# Create a new column for the absolute value of research and development expenditure
# Here we assume that 'GDP' is in the same units as you want for the 'rd_expenditure' (e.g., million USD)
merged_df['rd_expenditure'] = merged_df['rd_expenditure_ratio'] * merged_df['gdp']

# Let's remove the rows with missing 'rd_expenditure' or 'GDP' values
merged_df = merged_df.dropna(subset=['rd_expenditure'])
merged_df.to_csv('gedr_absolute_wb.csv')