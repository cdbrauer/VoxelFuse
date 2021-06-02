"""
Generate a series of prism solids.

----

Copyright 2021 - Cole Brauer, Dan Aukes
"""

import time
import voxelfuse as vf

if __name__=='__main__':
    # Save start time
    t1 = time.time()

    model1 = vf.prism((50, 25, 80), 0)
    model2 = vf.prism((50, 20, 40), -40, (80, 0, 0))
    model3 = vf.prism((80, 15, 50), 40, (160, 0, 0))

    model_result = model1 | model2 | model3

    # Create mesh data
    mesh1 = vf.Mesh.marchingCubes(model_result)

    # Get elapsed time
    t2 = time.time()
    time_mesh = t2 - t1
    print('Time to generate mesh: ' + str(time_mesh) + ' sec')

    # Save mesh
    mesh1.export('prism-set.stl')

    # Create plot
    mesh1.viewer(grids=True, name='prism set')