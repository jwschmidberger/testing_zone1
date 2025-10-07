#%%
import pypropel as pp
import tmkit as tmk
import numpy as np
import pandas as pd
# import sys
# import sqlite3



# %%
datapath = '/Users/jasonschmidberger/DataScience/data'

df_old = pp.io.read(
    df_fpn=f'{datapath}/pdbtm_alpha.txt',
    # df_fpn=pdb_fpn_dict['10.02.2023'],
    # df_fpn=pdb_fpn_dict['12.06.2024'],
    df_sep='\t',
    header=None,
)
df_old

# %%
df_short = df_old.head(200)
df_short
# %%
df = pd.DataFrame()
df['prot'] = df_short[0].astype(str).str[:4]
df['chain'] = df_short[0].astype(str).str[-1:]
df['ij'] = df.apply(lambda x: x['prot'] + '.' + x['chain'], axis=1)
print(df)
prot_series = pd.Series(df['prot'].unique())
print(prot_series)

# %%
tmk.seq.retrieve_pdb_from_pdbtm(
    prot_series=prot_series,
    kind='tr',
    sv_fp=f'{datapath}/cplx/',
)
# %%
