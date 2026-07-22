import importlib.util, sys, os
import numpy as np, pandas as pd

MOD = "/Users/atin/Nowcasting/Nowcasting_github/nn_mlp_nowcasting_model/MLP_AGT_temporalsplit_wandb.py"
spec = importlib.util.spec_from_file_location("mlpmod", MOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

merged_df, df_trends, rd_df, oecd4lag = m.load_and_preprocess_data()
merged_df, max_lag = m.preprocess_merged_data(merged_df, df_trends, rd_df, oecd4lag)

cols = list(merged_df.columns)
def grp(pred):
    return [c for c in cols if pred(c)]

print("TOTAL cols:", len(cols))
print("\n-- rd_expenditure lags:", grp(lambda c: c.startswith("rd_expenditure_lag")))
print("\n-- macro lag examples:", grp(lambda c: ("gdpca_lag" in c or "unemp_rate_lag" in c or "population_lag" in c or "inflation_lag" in c or "export_vol_lag" in c or "import_vol_lag" in c))[:10], "...count=", len(grp(lambda c: any(k in c for k in ["gdpca_lag","unemp_rate_lag","population_lag","inflation_lag","export_vol_lag","import_vol_lag"]))))
print("\n-- GT yearly_avg_lag count:", len(grp(lambda c: "_yearly_avg_lag" in c)))
print("-- GT mean_YTD count:", len(grp(lambda c: c.endswith("_mean_YTD") or "_mean_YTD" in c)))
print("\n-- sample yearly_avg_lag:", grp(lambda c: "_yearly_avg_lag" in c)[:3])
print("-- sample mean_YTD:", grp(lambda c: "_mean_YTD" in c)[:3])
print("\nCountries:", sorted(merged_df.Country.unique()))
print("Rows:", len(merged_df), "unique (country,year):", merged_df.groupby(['Country','Year']).ngroups)
print("Year range:", merged_df.Year.min(), merged_df.Year.max())
merged_df.to_parquet("/Users/atin/Nowcasting/Nowcasting_github/additional_analysis/out/merged_features.parquet") if False else merged_df.to_csv("/Users/atin/Nowcasting/Nowcasting_github/additional_analysis/out/merged_features.csv", index=False)
print("saved merged_features.csv")
