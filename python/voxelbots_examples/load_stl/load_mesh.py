# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:57:14 2019

@author: daukes
"""
import PyQt5.QtGui as qg
import sys
from voxelbots.voxel_model import VoxelModel
from voxelbots.mesh import Mesh
from voxelbots.plot import Plot

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    model = VoxelModel.fromMeshFile('axes.stl', 0, 0, 0)
    model = VoxelModel.fromVoxFile('cylinder-blue.vox', 0, 0, 0).union(model)

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(model)

    # Create plot
    plot1 = Plot(mesh1, grids=True)
    plot1.show()

    app1.processEvents()
    app1.exec_()