"""
Generate a series of primitive solids.

----

Copyright 2021 - Cole Brauer, Dan Aukes
"""

import sys
import time
import PyQt5.QtGui as qg
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.plot import Plot
from voxelfuse.primitives import *

if __name__=='__main__':
    # Save start time
    t1 = time.time()

    # app1 = qg.QApplication(sys.argv)

    model1 = cube(11, (0, 0, 0), 1)
    model2 = cuboid((11, 20, 6), (13, 0, 0), 2)
    model3 = sphere(5, (31, 5, 5), 3)
    model4 = cylinder(5, 15, (44, 5, 0), 4)
    model5 = cone(1, 5, 15, (57, 5, 0), 5)
    model6 = pyramid(0, 5, 15, (70, 5, 0), 6)

    model_result = model1 | model2 | model3 | model4 | model5 | model6

    # Save mesh data
    # model_result.marchingCubes('primitives', smooth=False)
    model_result.marchingCubes(None, smooth=False, display=True)
    model_result.marchingCubes(None, smooth=True, display=True)

    # Create plot
    # plot1 = Plot(mesh1, grids=True)
    # plot1.show()
    # app1.processEvents()

    # Get elapsed time
    # t2 = time.time()
    # time_pyqt5 = t2 - t1
    # print('Time to generate plot: ' + str(time_pyqt5) + ' sec')
    #
    # app1.exec()