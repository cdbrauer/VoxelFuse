"""
Copyright 2018
Dan Aukes, Cole Brauer

Mesh data import

Program imports a mesh file and converts it to voxels
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