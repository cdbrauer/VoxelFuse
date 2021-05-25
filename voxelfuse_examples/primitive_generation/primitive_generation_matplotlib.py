"""
Generate a series of primitive solids and display them using matplotlib

----

Copyright 2021 - Cole Brauer, Dan Aukes
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import voxelfuse as vf

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__=='__main__':
    # Save start time
    t1 = time.time()

    model1 = vf.cube(11, (0, 0, 0), 1)
    model2 = vf.cuboid((11, 20, 6), (13, 0, 0), 2)
    model3 = vf.sphere(5, (31, 5, 5), 3)
    model4 = vf.cylinder(5, 15, (44, 5, 0), 4)
    model5 = vf.cone(1, 5, 15, (57, 5, 0), 5)
    model6 = vf.pyramid(0, 5, 15, (70, 5, 0), 6)

    model_result = model1 | model2 | model3 | model4 | model5 | model6

    # Find exterior voxels
    interiorVoxels = model_result.erode(radius=1, connectivity=1)
    exteriorVoxels = model_result.difference(interiorVoxels)
    voxels = exteriorVoxels.voxels

    # Get colors
    colors = np.empty(voxels.shape, dtype=object)
    colors[:] = 'white'
    colors[voxels == 1] = 'red'
    colors[voxels == 2] = 'green'
    colors[voxels == 3] = 'blue'
    colors[voxels == 4] = 'gray'
    colors[voxels == 5] = 'black'
    colors[voxels == 6] = 'cyan'

    # Plot
    fig = plt.figure()
    ax3d = fig.add_subplot(projection='3d')
    ax3d.voxels(voxels, facecolors=colors)
    set_axes_equal(ax3d)
    plt.show()

    # Get elapsed time
    t2 = time.time()
    time_matplotlib = t2 - t1
    print('Time to generate plot: ' + str(time_matplotlib) + ' sec')