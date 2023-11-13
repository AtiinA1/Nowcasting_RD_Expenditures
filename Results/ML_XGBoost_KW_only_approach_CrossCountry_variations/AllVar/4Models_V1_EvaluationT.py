import os
import pandas as pd

# Set up root directory
root_dir = "/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar"

# Dynamically generate the list of sub-models based on sub-directories in the root directory
sub_models = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

# Initialize a DataFrame to store all results
all_results_df = pd.DataFrame()

# Iterate over sub-models
for sub_model in sub_models:
    # Construct directory path
    directory_path = os.path.join(root_dir, sub_model)
    
    # Ensure the directory exists
    if os.path.exists(directory_path):
        # Iterate over files in directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv') and filename.startswith('evaluation_test'):
                # Construct file path
                file_path = os.path.join(directory_path, filename)
                
                # Parse model name from filename
                model = filename.replace('evaluation_test', '').replace('.csv', '')
                
                # Load the data, add model and sub-model columns, and append to all_results_df
                df = pd.read_csv(file_path)
                df['Model'] = model
                df['Sub-Model'] = sub_model
                all_results_df = all_results_df.append(df, ignore_index=True)

# Save to a CSV file
all_results_df.to_csv("/Users/atin/Nowcasting/data/Results/4Models_V4_KWO_CrossCountry_XGBoost/AllVar/combined_results_V4_KWO.csv", index=False)
