#%%
import chemdraw

mol = "O=C(C)Oc1ccccc1C(=O)O"
drawer = chemdraw.Drawer(mol, title=mol)
fig = drawer.draw()
fig.show()


# %%
