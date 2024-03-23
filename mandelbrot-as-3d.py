import numpy as np
import matplotlib.cm as cm
import pyvista as pv
from vtk import vtkOctreePointLocator


# Set the boundaries of the complex plane
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

# Other position levels
#x_min, x_max = -1.0, 0.5
#y_min, y_max = -0.75, 0.75

#x_min, x_max = -0.3+0.2, 0.45+0.2
#y_min, y_max = -0.375, 0.375

#x_min, x_max = (-0.3+0.2)/2+0.2, (0.45+0.2)/2+0.2
#y_min, y_max = -0.375/2, 0.375/2

#x_min, x_max = ((-0.3+0.2)/2+0.2)/2+0.1, ((0.45+0.2)/2+0.2)/2+0.1
#y_min, y_max = -0.375/2/2, 0.375/2/2

#x_min, x_max = (((-0.3+0.2)/2+0.2)/2+0.1)/2+0.15, (((0.45+0.2)/2+0.2)/2+0.1)/2+0.15
#y_min, y_max = -0.375/2/2/2, 0.375/2/2/2

#x_min, x_max = ((((-0.3+0.2)/2+0.2)/2+0.1)/2+0.12)/2+0.15, ((((0.45+0.2)/2+0.2)/2+0.1)/2+0.12)/2+0.15
#y_min, y_max = -0.375/2/2/2/2, 0.375/2/2/2/2

width, height = 2000, 2000  # Resolution of the image - 20000, 20000 was used to create the screen graps
max_iterations = 1000  # Maximum number of iterations
zoom_level = 100000000.0;
escape_radius = 2; # shouldn't need to change
threshold = 1  # Adjust the threshold value based on your preference - Typically 2 or 1

def mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, escape_radius):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    z = np.zeros_like(c, dtype=np.complex128)
    iterations = np.zeros((height, width), dtype=int)
    
    for i in range(max_iter):
        z = z ** 2 + c
        mask = (np.abs(z) > escape_radius)
        iterations[mask & (iterations == 0)] = i
        z[mask] = 2
    
    return iterations


def colorize(iterations, max_iter, brightness_threshold=0.5, brightness_boost=0.5):
    norm = iterations.astype(float) / max_iter
    colors = cm.viridis(norm)
    
    # Calculate the luminance of each color
    luminance = np.dot(colors[..., :3], [0.299, 0.587, 0.114])
    
    # Create a mask for colors below the brightness threshold
    mask = luminance < brightness_threshold
    
    # Apply the brightness boost to the colors below the threshold
    colors[mask, :] = np.power(colors[mask, :], brightness_boost)
    
    return colors[..., :3]


def calculate_depth(colors, zoom_level):
    depth_scale = 1.0 + zoom_level * 0.1
    depth = 1.0 - np.linalg.norm(colors - [0, 0, 0], axis=-1) / np.sqrt(3)
    depth *= depth_scale
    return depth


def adaptive_sampling(iterations, threshold):
    mask = iterations > threshold
    return mask


def generate_3d_points(iterations, depth, threshold, zoom_level):
    height, width = iterations.shape
    
    scaling_factor = zoom_level / 10
    
    # Adjust the range of x and y coordinates based on the scaling factor
    x_range = (x_max - x_min) * scaling_factor
    y_range = (y_max - y_min) * scaling_factor
    x_min_scaled = -x_range / 2
    x_max_scaled = x_range / 2
    y_min_scaled = -y_range / 2
    y_max_scaled = y_range / 2
    
    x = np.linspace(x_min_scaled, x_max_scaled, width)
    y = np.linspace(y_min_scaled, y_max_scaled, height)
    xx, yy = np.meshgrid(x, y)
    
    mask = adaptive_sampling(iterations, threshold)
    xx_sampled = xx[mask]
    yy_sampled = yy[mask]
    depth_sampled = depth[mask]
    
    points = np.column_stack((xx_sampled.flatten(), yy_sampled.flatten(), depth_sampled.flatten()))
    return points



def render_point_cloud(points, colors, iterations, threshold):
    cloud = pv.PolyData(points)
    
    # Apply adaptive sampling to colors
    colors_sampled = colors[adaptive_sampling(iterations, threshold)]
    colors_flat = colors_sampled.reshape(-1, 3)
    
    cloud['colors'] = colors_flat

    # Create an octree for level-of-detail rendering
    octree = vtkOctreePointLocator()
    octree.SetDataSet(cloud)
    octree.BuildLocator()

    # Print the number of points being displayed
    num_points = cloud.n_points
    print(f"Number of points being displayed: {num_points}")

    plotter = pv.Plotter()
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=5.0, scalars='colors', rgb=True)
    plotter.show()


# Main program
iterations = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iterations, escape_radius)
colors = colorize(iterations, max_iterations)
depth = calculate_depth(colors, zoom_level=zoom_level)

points = generate_3d_points(iterations, depth, threshold, zoom_level)
render_point_cloud(points, colors, iterations, threshold)
