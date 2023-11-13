import pandas as pd
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/test10_ext_r2/df_combined_pred_vs_true.csv")


# Compute R-squared for train, validation, and test
types = df['Type'].unique()
r2_scores_general = []

for t in types:
    df_type = df[df['Type'] == t]
    r2 = r2_score(df_type['True_Values'], df_type['Predicted_Values'])
    r2_scores_general.append({
        'Type': t,
        'R_squared': r2
    })

# Save the general R-squared values to a CSV file
r2_scores_general_df = pd.DataFrame(r2_scores_general)
r2_scores_general_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/test10_ext_r2/r2_scores_general.csv", index=False)

# Compute R-squared per country and type
r2_scores_per_country = []

for country in df['Country'].unique():
    for t in types:
        df_country_type = df[(df['Country'] == country) & (df['Type'] == t)]
        r2 = r2_score(df_country_type['True_Values'], df_country_type['Predicted_Values'])
        r2_scores_per_country.append({
            'Country': country,
            'Type': t,
            'R_squared': r2
        })

# Convert the results to a DataFrame
r2_scores_per_country_df = pd.DataFrame(r2_scores_per_country)

# Save the per-country R-squared values to a CSV file
r2_scores_per_country_df.to_csv("/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/test10_ext_r2/r2_scores_per_country_and_type.csv", index=False)

# Output the general R-squared scores
print("General R-squared scores for each type:")
print(r2_scores_general_df)

