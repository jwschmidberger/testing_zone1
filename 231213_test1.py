# %%
import pandas as pd
import numpy as np
import sketch

from sklearn.datasets import load_iris

iris = load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
data1

# %%
data1.sketch.ask("What is this dataframe about?")
# %%
data1.sketch.howto("How would I find the max value for each column along with its index")
# %%

# Get the max value for each column along with its index
max_values = data1.max()
max_indexes = data1.idxmax()

# Print the results
for col, val, idx in zip(data1.columns, max_values, max_indexes):
    print(f"The max value of {col} is {val} at index {idx}")
# %%
