#%%
import pandas as pd
import os


print ('all modules loaded')

df = pd.read_csv('products.csv',lineterminator='\n').reset_index(drop=True)
df['price'] = df['price'].replace('[\£,]', '', regex=True).astype(float)

df.head(10)

# %%
