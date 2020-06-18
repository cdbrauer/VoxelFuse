"""
Import mesh data.

Import a mesh file and convert it to voxels.

----

Copyright 2019 - Cole Brauer, Dan Aukes
"""

import PyQt5.QtGui as qg
import sys
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.plot import Plot

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    model = VoxelModel.fromMeshFile('axes.stl', (0, 0, 0))

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(model)

    # Create plot
    plot1 = Plot(mesh1, grids=True)
    plot1.show()

    app1.processEvents()
    app1.exec_()