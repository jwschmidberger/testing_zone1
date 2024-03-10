#%%

# This is looking at a mortgage structure for a house we are interested in.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





# %%
P = 316000
r = 0.0449

b = P * (1 + (r/365))**1
b

# %%
values = np.array([])
b = 316000
for i in range(365*20):
    b = b * (1 + (r/365))
    if (i != 0) & (i%30 == 0):
        b = b - 2200
    values = np.append(values, b)
values

# %%
sns.lineplot(values)
# %%
30%30
# %%
