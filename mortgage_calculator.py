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
values = np.array([])
b = 316000
r = 0.0449

years = np.arange(0, 20, 1/365)  # Create an array of years

for i in range(365*20):
    b = b * (1 + (r/365))
    if (i != 0) & (i % 30 == 0):
        b = b - 2200
    values = np.append(values, b)

year_below_zero = None
for i in range(len(values)):
    if values[i] < 0:
        year_below_zero = years[i]
        break

# Plot
sns.lineplot(x=years, y=values)
plt.xlabel('Years')
plt.ylabel('Mortgage Amount')
plt.title('Mortgage Amount Over 20 Years')
if year_below_zero is not None:
    plt.axvline(x=year_below_zero, color='r', linestyle='--', label=f'Mortgage Below 0 ({year_below_zero:.2f} years)')
plt.legend()
plt.show()


# %%
