import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

# Load the data
df = pd.read_csv('/Users/atin/Nowcasting/data/Results/AggMGT_Evolution/MLP_Ensemble_10_Final/XGBoost_MLP_Meta_Model/AllVar/test10_ext_ensemblesize10/df_combined_pred_vs_true.csv')

# Compute the residuals
df['Residuals'] = df['True_Values'] - df['Predicted_Values']

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of residuals
sns.histplot(df['Residuals'], kde=True)
plt.title('Histogram of Residuals')
plt.savefig('/Users/atin/Nowcasting/data/Results/Cleanedup/ML_DL_Models_Results/AllVar/test10_ext/Histogram_residuals.png')
plt.show()

# Residuals vs. Fitted values
plt.scatter(df['Predicted_Values'], df['Residuals'])
plt.axhline(0, color='red', linestyle='--')  # Zero line for reference
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('/Users/atin/Nowcasting/data/Results/Cleanedup/ML_DL_Models_Results/AllVar/test10_ext/scatter.png')
plt.show()

# Mean of residuals
print("Mean of Residuals:", df['Residuals'].mean())

from scipy.stats import shapiro, jarque_bera

# Shapiro-Wilk Test
stat, p = shapiro(df['Residuals'])
print("Shapiro-Wilk Test p-value:", p)

# Jarque-Bera Test
jb_stat, jb_p = jarque_bera(df['Residuals'])
print("Jarque-Bera Test p-value:", jb_p)

from statsmodels.graphics.tsaplots import plot_acf

# ACF plot
plot_acf(df['Residuals'].dropna())
plt.title('Autocorrelation Plot of Residuals')
plt.savefig('/Users/atin/Nowcasting/data/Results/Cleanedup/ML_DL_Models_Results/AllVar/test10_ext/autocorrelation.png')

plt.show()



# 1. Mean of residuals
mean_residuals = df['Residuals'].mean()

# 2. Shapiro-Wilk Test for normality
shapiro_stat, shapiro_p = shapiro(df['Residuals'])

# 3. Breusch-Pagan Test for heteroscedasticity
_, bp_p, _, _ = het_breuschpagan(df['Residuals'], df[['Predicted_Values']])

# 4. Durbin-Watson Test for autocorrelation
dw_stat = durbin_watson(df['Residuals'])

# Saving the results
results = {
    'Mean of Residuals': mean_residuals,
    'Shapiro-Wilk Test Stat': shapiro_stat,
    'Shapiro-Wilk Test p-value': shapiro_p,
    'Breusch-Pagan Test p-value': bp_p,
    'Durbin-Watson Stat': dw_stat
}

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
results_df.to_csv('/Users/atin/Nowcasting/data/Results/Cleanedup/ErrorTest/statistical_test_results_errorterm_allvar_alltype.csv')


######################################################################

df_train = df[df['Type'] == 'Train']

# Now, you can apply any of the above methods separately on these filtered residuals.
# Histogram of residuals
sns.histplot(df_train['Residuals'], kde=True)
plt.title('Histogram of Residuals')
plt.savefig('/Users/atin/Nowcasting/data/Results/Cleanedup/ErrorTest/Histogram_residuals.png')
plt.show()

# Residuals vs. Fitted values
plt.scatter(df_train['Predicted_Values'], df_train['Residuals'])
plt.axhline(0, color='red', linestyle='--')  # Zero line for reference
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('/Users/atin/Nowcasting/data/Results/Cleanedup/ErrorTest/scatter.png')
plt.show()

# Mean of residuals
print("Mean of Residuals:", df_train['Residuals'].mean())

from scipy.stats import shapiro, jarque_bera

# Shapiro-Wilk Test
stat, p = shapiro(df_train['Residuals'])
print("Shapiro-Wilk Test p-value:", p)

# Jarque-Bera Test
jb_stat, jb_p = jarque_bera(df_train['Residuals'])
print("Jarque-Bera Test p-value:", jb_p)

from statsmodels.graphics.tsaplots import plot_acf

# ACF plot
plot_acf(df_train['Residuals'].dropna())
plt.title('Autocorrelation Plot of Residuals')
plt.savefig('/Users/atin/Nowcasting/data/Results//Cleanedup/ErrorTest/autocorrelation.png')

plt.show()



# 1. Mean of residuals
mean_residuals = df_train['Residuals'].mean()

# 2. Shapiro-Wilk Test for normality
shapiro_stat, shapiro_p = shapiro(df_train['Residuals'])

# 3. Breusch-Pagan Test for heteroscedasticity
_, bp_p, _, _ = het_breuschpagan(df_train['Residuals'], df_train[['Predicted_Values']])

# 4. Durbin-Watson Test for autocorrelation
dw_stat = durbin_watson(df_train['Residuals'])

# Saving the results
results = {
    'Mean of Residuals': mean_residuals,
    'Shapiro-Wilk Test Stat': shapiro_stat,
    'Shapiro-Wilk Test p-value': shapiro_p,
    'Breusch-Pagan Test p-value': bp_p,
    'Durbin-Watson Stat': dw_stat
}

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])

results_df.to_csv('/Users/atin/Nowcasting/data/Results/Cleanedup/ErrorTest/statistical_test_results_errorterm_allvar_train_residuals.csv')

