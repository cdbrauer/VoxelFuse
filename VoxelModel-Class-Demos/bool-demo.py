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
    model1 = VoxelModel.fromFile('square-blue-L.vox')
    model2 = VoxelModel.fromFile('square-red-R.vox')

    # Perform Boolean Operations ###############################################
    modelAdd = model1 + model2
    modelSub = model1 - model2

    # Create mesh data
    mesh1 = Mesh(modelAdd)
    mesh2 = Mesh(modelSub)

    # Create plot
    plot1 = Plot(mesh1)
    plot2 = Plot(mesh2)

    # Process all events
    app1.processEvents()

    # Save screenshot of plot 1
    plot1.export('voxel-tools-bool-fig1.png')

    # Export mesh
    #mesh1.export('result.stl')

    app1.exec_()
