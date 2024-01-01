'''
Here I am working with a DataCamp Course that intruduces the concept of Deep Learning.
'''

#%%
import numpy as np
input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]), 
           'output': np.array([2, -1])}
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()
hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)
# %%
hidden_layer_values.shape
# %%
output = (hidden_layer_values * weights['output']).sum()
print(output)
# %%
input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]), 
           'output': np.array([2, -1])}
node_0_value = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_value)
node_1_value = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_value)
hidden_layer_values = np.array([node_0_output, node_1_output])
output = (hidden_layer_values * weights['output']).sum()
print(output)

# %%
'''
Forward Propagation using Relu activation function
'''

def relu(input_data):
    result = max(input_data, 0)
    return(result)


input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]), 
           'output': np.array([2, -1])}
node_0_value = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_value)
node_1_value = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_value)
hidden_layer_values = np.array([node_0_output, node_1_output])
output = (hidden_layer_values * weights['output']).sum()
print(output)
# %%


input_data = np.array([np.array([3, 5]), np.array([1, -1]), 
                       np.array([0, 0]), np.array([8, 4])])
weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 'output': np.array([2, 7])}
weights
# %%
for item in input_data:
    print(item)
# %%
def forward_propagation(input_data_row, weights):
    node_0_value = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_value)
    node_1_value = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_value)
    hidden_layer_values = np.array([node_0_output, node_1_output])
    output = (hidden_layer_values * weights['output']).sum()
    return(output)

results = []

for row in input_data:
    results.append(forward_propagation(row, weights))

results

# %%
