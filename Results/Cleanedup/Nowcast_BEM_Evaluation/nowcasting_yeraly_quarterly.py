# Yearly data
import pandas as pd
import os
import matplotlib.pyplot as plt

save_path = "/Users/atin/Nowcasting/data/Results/Cleanedup/ML_DL_Models_Results/MGTwRD/bem/test10-mlp-pytorch/Plots"

# Ensure the directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Read the dataset
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/Cleanedup/ML_DL_Models_Results/MGTwRD/bem/test10-mlp-pytorch/df_trends_monthly_estimate.csv")
df = df[(df['Year'] < 2021)]


# Group by Country and Year to aggregate the estimated_rd_expenditure
yearly_estimate = df.groupby(['Country', 'Year'])['estimated_rd_expenditure'].sum().reset_index()

# Merge with the original data to get rd_expenditure
yearly_comparison = pd.merge(yearly_estimate, df[['Country', 'Year', 'rd_expenditure']].drop_duplicates(), on=['Country', 'Year'])
yearly_comparison = yearly_comparison[yearly_comparison['Year'] < 2023]

print(yearly_comparison)

countries = df['Country'].unique()

for country in countries:
    subset = yearly_comparison[yearly_comparison['Country'] == country]
    plt.figure(figsize=(10,5))
    
    # Plotting even if there are NaN values for 'True Value'
    if not subset['rd_expenditure'].isna().all():  # Plot only if there are some non-NaN values
        plt.plot(subset['Year'], subset['rd_expenditure'], label='True Value', linestyle='-', marker='o')
    
    plt.plot(subset['Year'], subset['estimated_rd_expenditure'], label='Estimated Value', linestyle='--', marker='x')
    plt.title(f'Yearly R&D Expenditure for {country}')
    plt.xlabel('Year')
    plt.ylabel('Expenditure (MLN_USD)')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'Yearly_RD_Expenditure_{country}.png'))
    plt.close()  # Close the plot to free up memory

# for country in countries:
#     subset = yearly_comparison[yearly_comparison['Country'] == country]
#     plt.figure(figsize=(10,5))
#     plt.plot(subset['Year'], subset['rd_expenditure'], label='True Value')
#     plt.plot(subset['Year'], subset['estimated_rd_expenditure'], label='Estimated Value', linestyle='--')
#     plt.title(f'Yearly R&D Expenditure for {country}')
#     plt.xlabel('Year')
#     plt.ylabel('Expenditure')
#     plt.legend()
#     plt.savefig(os.path.join(save_path, f'Yearly_RD_Expenditure_{country}.png'))    
#     plt.show()

# Quarterly data

# Convert month to quarters
df['Quarter'] = df['Month'].apply(lambda x: (x-1)//3 + 1)

# Group by Country, Year, and Quarter to aggregate the data
quarterly_true = df.groupby(['Country', 'Year', 'Quarter'])['monthly_rd_expenditure'].sum().reset_index()
quarterly_estimate = df.groupby(['Country', 'Year', 'Quarter'])['estimated_rd_expenditure'].sum().reset_index()

# Merge to get a comparison table
quarterly_comparison = pd.merge(quarterly_true, quarterly_estimate, on=['Country', 'Year', 'Quarter'])

print(quarterly_comparison)

for country in countries:
    subset = quarterly_comparison[quarterly_comparison['Country'] == country]
    plt.figure(figsize=(10,5))
    plt.plot(subset['Year'] + (subset['Quarter'] - 1) * 0.25, subset['monthly_rd_expenditure'], label='Artificially-Distributed True Value')
    plt.plot(subset['Year'] + (subset['Quarter'] - 1) * 0.25, subset['estimated_rd_expenditure'], label='Estimated Value', linestyle='--')
    plt.title(f'Quarterly R&D Expenditure for {country}')
    plt.xlabel('Time (in quarters)')
    plt.ylabel('Expenditure (MLN_USD)')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'Quarterly_RD_Expenditure_{country}.png'))    
    plt.show()


# for country in countries:
#     subset = quarterly_comparison[quarterly_comparison['Country'] == country]
#     plt.figure(figsize=(10,5))
    
#     # Create a list of strings for the x-axis ticks in the format "Year-Q"
#     x_ticks_labels = [f"{year}-Q{quarter}" for year, quarter in zip(subset['Year'], subset['Quarter'])]
    
#     plt.plot(range(len(subset)), subset['monthly_rd_expenditure'], label='True Value')
#     plt.plot(range(len(subset)), subset['estimated_rd_expenditure'], label='Estimated Value', linestyle='--')
    
#     # Setting the x-ticks to show the "Year-Q" labels
#     plt.xticks(range(len(subset)), x_ticks_labels, rotation=45)  # rotation for better visibility
    
#     plt.title(f'Quarterly R&D Expenditure for {country}')
#     plt.xlabel('Time (Year-Quarter)')
#     plt.ylabel('Expenditure')
#     plt.legend()
#     plt.tight_layout()  # To ensure everything fits properly
#     plt.savefig(os.path.join(save_path, f'Quarterly_RD_Expenditure_{country}.png'))    
#     plt.show()

