#%%
'''I am looking into multivariate interpolation of a irregular grid of points.
This is challenging for being both multivariate and irregular.
I have found a few references to methods that might work, but nothing definitive.
One reference is to a paper by Renka (1988) on multivariate interpolation of scattered data points.
Another reference is to the work by Shewchuk (1997) on adaptive triangulation and interpolation.
I am particularly interested in methods that can handle the irregularity of the grid and provide accurate interpolation results.

From the wiki page on Multivariate interpolation: https://en.wikipedia.org/wiki/Multivariate_interpolation
* In numerical analysis, multivariate interpolation is the problem of interpolating functions of more than one variable.

Regular grids
* The values are already on a regular grid, and the problem is to interpolate at points in between the grid points. 
* This is a common problem in image processing, where the image pixels form a regular grid, and one wants to interpolate the image at non-pixel locations. 
* Common methods for this problem include bilinear interpolation, bicubic interpolation, and spline interpolation.

Irregular grids
* The values are given at scattered points in the domain, and the problem is to interpolate at other points in the domain. 
* This is a more challenging problem, as there is no regular structure to the data. 
* Common methods for this problem include nearest neighbor interpolation, inverse distance weighting, radial basis functions, and kriging.


I have already looked into the use of nearest neighbor interpolation and it did not provide satisfactory results.
I am now looking into the use of radial basis functions (RBF) for multivariate interpolation on irregular grids.


I have found a few references to the use of RBF for this problem, including the following:
* Fasshauer, G. E. (2007). Meshfree approximation methods with MATLAB. World Scientific Publishing Company.
* Buhmann, M. D. (2003). Radial basis functions: theory and implementations. Cambridge University Press.
* Wendland, H. (2004). Scattered data approximation. Cambridge University Press.


Recommended (perplexity) reading:
from scipy.interpolate import Rbf, griddata
Moved to RBFInterpolator.

https://www.tutorialspoint.com/scipy/scipy_interpolate_rbfinterpolator_function.htm

'''

#%%
import numpy as np
from scipy.interpolate import RBFInterpolator

# Define known points (x, y) and their values
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
values = np.array([0, 1, 1, 0])

# Create the RBF interpolator
interpolator = RBFInterpolator(points, values)

# Interpolate at a new point
new_point = np.array([[0.5, 0.5]])
interpolated_value = interpolator(new_point)
print("Interpolated value at (0.5, 0.5):", interpolated_value[0])



#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

# Define known data points (x, y) and their corresponding values (z)
x = np.random.rand(10) * 10  # Random x-coordinates
y = np.random.rand(10) * 10  # Random y-coordinates
z = np.sin(x) * np.cos(y)    # Some values based on a function

# Create a meshgrid for interpolation
grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]

# Create a plot to compare the kernels
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
kernels = ['linear', 'cubic', 'quintic', 'gaussian', 'thin_plate_spline']
epsilon_value = 1.0  # Set a value for epsilon for the Gaussian kernel

# Interpolate and plot for each kernel
for ax, kernel in zip(axs.flatten(), kernels):
    # Set epsilon only for Gaussian kernel
    if kernel == 'gaussian':
        rbf = RBFInterpolator(np.column_stack((x, y)), z, kernel=kernel, epsilon=epsilon_value)
    else:
        rbf = RBFInterpolator(np.column_stack((x, y)), z, kernel=kernel)

    # Perform interpolation on the grid (stacking grid_x and grid_y)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))  # Combine into (n_points, 2)
    grid_z = rbf(points)  # Perform interpolation

    # Reshape grid_z back to the original grid shape
    grid_z = grid_z.reshape(grid_x.shape)

    # Plot the result
    img = ax.imshow(grid_z.T, extent=(0, 10, 0, 10), origin='lower', cmap='viridis', alpha=0.8)
    ax.scatter(x, y, c=z, edgecolor='k', label='Data Points')
    ax.set_title(f'RBF Interpolation with Kernel: {kernel}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(img, ax=ax, label='Interpolated Values')

# Show the plot
plt.tight_layout()
plt.show()




# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D

# Define known data points (x, y) and their corresponding values (z)
x = np.random.rand(10) * 10  # Random x-coordinates
y = np.random.rand(10) * 10  # Random y-coordinates
z = np.sin(x) * np.cos(y)    # Some values based on a function

# Create a meshgrid for interpolation
grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]

# Set epsilon for the Gaussian kernel
epsilon_value = 1.0

# Create RBF interpolator with Gaussian kernel
rbf = RBFInterpolator(np.column_stack((x, y)), z, kernel='gaussian', epsilon=epsilon_value)

# Perform interpolation on the grid (stacking grid_x and grid_y)
points = np.column_stack((grid_x.ravel(), grid_y.ravel()))  # Combine into (n_points, 2)
grid_z = rbf(points)  # Perform interpolation
grid_z = grid_z.reshape(grid_x.shape)  # Reshape grid_z back to the original grid shape

# Create 2D contour plot
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
contour = plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Interpolated Values')
plt.scatter(x, y, c=z, edgecolor='k', label='Data Points')
plt.title('RBF Interpolation - Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Create 3D surface plot
ax = plt.subplot(1, 2, 2, projection='3d')
ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.scatter(x, y, z, color='r', label='Data Points')
ax.set_title('RBF Interpolation - 3D Surface Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=30)  # Adjust the viewing angle
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
# %%
