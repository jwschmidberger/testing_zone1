#%%

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
#%%

df = pd.read_csv('example.csv')


df.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigree",
    "Age",
    "Outcome",
]
df
# %%
# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('example.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
y.shape

# %%
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
X


# %%
 