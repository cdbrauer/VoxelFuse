"""
Generate a series of primitive solids.

----

Copyright 2021 - Cole Brauer, Dan Aukes
"""

import time
import voxelfuse as vf

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

    # Create mesh data
    # mesh1 = vf.Mesh.fromVoxelModel(model_result)
    mesh2 = vf.Mesh.simpleSquares(model_result)
    # mesh3 = vf.Mesh.marchingCubes(model_result, smooth=False)

    # Get elapsed time
    t2 = time.time()
    time_mesh = t2 - t1
    print('Time to generate mesh: ' + str(time_mesh) + ' sec')

    # Save mesh
    # mesh1.export('primitives.stl')
    mesh2.export('primitives-simple.stl')
    # mesh3.export('primitives-marchingcubes.stl')

    # Create plot
    # mesh1.viewer(grids=True, name='primitives')
    mesh2.viewer(grids=True, name='primitives-simple')
    # mesh3.viewer(grids=True, name='primitives-marchingcubes') # TODO: Creating a second window causes an OpenGL error and displays a blank plot
