"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import PyQt5.QtGui as qg
import sys

from VoxelModel import VoxelModel
from Mesh import Mesh
from Plot import Plot

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    # Import models ##############################################################
    model1 = VoxelModel.fromFile('sphere-blue.vox', 0, 0, 0)
    model2 = VoxelModel.fromFile('sphere-red.vox', -4, 4, 4)
    model3 = VoxelModel.fromFile('sphere-blue.vox', 0, 4, 4)

    # Perform Boolean Operations ###############################################
    modelAdd = model1 + model2 + model3
    modelSub = model1 - model2
    modelInt = model1.intersection(model2)

    # Create mesh data
    mesh1 = Mesh(modelAdd)
    mesh2 = Mesh(modelSub)
    mesh3 = Mesh(modelInt)

    # Create plot
    plot1 = Plot(mesh1)

    # Process all events
    app1.processEvents()

    # Save screenshot of plot 1
    plot1.export('voxel-tools-bool-fig1.png')

    # Export mesh
    #mesh1.export('result.stl')

    app1.exec_()
