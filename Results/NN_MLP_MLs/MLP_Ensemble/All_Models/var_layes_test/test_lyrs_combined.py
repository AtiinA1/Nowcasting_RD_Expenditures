import pandas as pd
import os

# The root directory where your folders are
root_dir = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/var_ly/"

# An empty list to hold all the dataframes
all_data = []

# Loop through all the folders in the root directory
for folder_name in os.listdir(root_dir):
    # Construct the full path to the folder
    folder_path = os.path.join(root_dir, folder_name)
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Construct the full file path to the metrics CSV
        file_path = os.path.join(folder_path, "metrics_df_minibatchGD.csv")
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Add a new column with the folder name
            df['Model_Type'] = folder_name
            # Append the dataframe to the list
            all_data.append(df)

# Concatenate all dataframes into one
final_df = pd.concat(all_data, ignore_index=True)

# Output the combined dataframe to a new CSV file
final_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/var_ly/combined_metrics.csv", index=False)
