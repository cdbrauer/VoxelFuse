"""
Export a VoxelFuse model as a .vxc file that can be imported into VoxCad.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import *

if __name__=='__main__':
    # app1 = qg.QApplication(sys.argv)

    model1 = cube(5, (0, 0, 0), material=1)
    model2 = cube(5, (4, 0, 3), material=8)
    model3 = cube(5, (8, 0, 6), material=3)

    modelResult = (model1 + model2).scaleValues() | model3

    # Save VXC file
    modelResult.saveVXC("test-file", compression=False)
    modelResult.saveVF("test-file")

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(modelResult)

    # Create plot
    mesh1.viewer()
