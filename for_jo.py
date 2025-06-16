#%%
import chemdraw
import numpy as np

mol = "O=C(C)Oc1ccccc1C(=O)O"
drawer = chemdraw.Drawer(mol, title=mol)
fig = drawer.draw()
fig.show()


# %%
import numpy as np
# Generate a random dataset with 100 samples and 2 features
data = np.random.rand(100, 2)
print(data)
# %%
