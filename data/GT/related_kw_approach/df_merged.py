import pandas as pd
import numpy as np

# Load the data
df1 = pd.read_csv('./final_df_BusinessRnDEcosystem1.csv', index_col=0)
df2 = pd.read_csv('./final_df_BusinessRnDEcosystem2.csv', index_col=0)
df3 = pd.read_csv('./final_df_BusinessRnDEcosystem3.csv', index_col=0)


df1 = df1[df1.columns.drop(list(df1.filter(regex='isPartial')))]
df2 = df2[df2.columns.drop(list(df2.filter(regex='isPartial')))]
df3 = df3[df3.columns.drop(list(df3.filter(regex='isPartial')))]

# Merge the two DataFrames
cols_to_drop1 = df1.filter(regex='^DE_').columns
df1 = df1.drop(cols_to_drop1, axis=1)

cols_to_drop2 = df2.filter(regex='^GB_').columns
df2 = df2.drop(cols_to_drop2, axis=1)

df_merged1 = pd.merge(df1, df2, how='outer' , left_on='date', right_on='date')

df_merged2 = pd.merge(df_merged1, df3, how='outer' , left_on='date', right_on='date')


df_merged2.to_csv('final_df_BusinessRnDEcosystem_merged.csv')
