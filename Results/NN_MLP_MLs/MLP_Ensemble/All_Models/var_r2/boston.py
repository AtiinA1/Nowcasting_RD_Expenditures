
import numpy as np
import pandas as pd

# For plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn libraries for dataset, model, and metrics
#from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#boston_dataset = load_boston()

from sklearn.datasets import fetch_california_housing
housing_dataset = fetch_california_housing()
housing = pd.DataFrame(housing_dataset.data, columns=housing_dataset.feature_names)
housing['Target'] = housing_dataset.target

features = ['MedInc', 'HouseAge']
target = housing['Target']

X_train = housing[features]
y_train = target

###########################################################################

# Initialize the model
model = LinearRegression()

model_name = type(model).__name__

###########################################################################


# Now pass the data to the model
model_fit = model.fit(X_train, y_train)

# Get the model parameters
model_params = model_fit.get_params()

# Calculate the accuracy score on the training set (R2?)
train_score = model_fit.score(X_train, y_train)

#print(f'Training Set, model_fit.coef_, model_fit.intercept_)

# Perform cross-validation using the model
cv_scores = cross_val_score(model_fit, X_train, y_train, cv=4)

# Make predictions (Training set)
y_pred_train = model_fit.predict(X_train)

# Create a new DataFrame for storing predictions and residuals
predictions_df = pd.DataFrame({'True_Values': y_train, 'Predicted_Values': y_pred_train})
predictions_df['Residuals'] = predictions_df['True_Values'] - predictions_df['Predicted_Values']

print(predictions_df.head(5))
print(predictions_df.info())

train_r2 = r2_score(y_train, y_pred_train)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(predictions_df['True_Values'], predictions_df['Predicted_Values']))
mae = mean_absolute_error(predictions_df['True_Values'], predictions_df['Predicted_Values'])

# Compute Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((predictions_df['True_Values'] - predictions_df['Predicted_Values']) / predictions_df['True_Values'])) * 100

combined_df = y_train

# Select relevant columns and rename as needed
df_combined_pred_vs_true = predictions_df[['True_Values', 'Predicted_Values']].copy()
df_combined_pred_vs_true.rename(columns={'True_Values': 'Target', 'Predicted_Values': 'y_pred_train'}, inplace=True)

# Save to CSV
csv_file_path = "/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/boston/df_combined_pred_vs_true.csv"
df_combined_pred_vs_true.to_csv(csv_file_path, index=False)


#%%--------------------------------------------------------------------------

# Create DataFrames to store the MAE and RMSE results for each lag
evaluation_train_df = pd.DataFrame(columns=['MAE', 'RMSE', 'MAPE', 'R^2'])

train_row = {
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'R^2': train_r2
}

evaluation_train_df = evaluation_train_df.append(train_row, ignore_index=True)

# Save the DataFrames to CSV files
evaluation_train_df.to_csv(f"/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/boston/evaluation_batches_train_{model_name}.csv", index=False)
