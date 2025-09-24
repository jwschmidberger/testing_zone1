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
