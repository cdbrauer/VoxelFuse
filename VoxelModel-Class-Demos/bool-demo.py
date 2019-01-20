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
    model2 = VoxelModel.fromFile('sphere-red.vox', 4, 4, 4)
    model3 = VoxelModel.fromFile('sphere-blue.vox', 4, 4, 4)
    model4 = model2 + model3

    # Perform Boolean Operations ###############################################
    #modelResult = model3.isolateLayer(2)
    #modelResult = model1 + model2 + model3
    #modelResult = model1.subtractVolume(model2)
    modelResult = model4.intersectVolume(model1)
    #modelResult = model1.addVolume(model2)
    #modelResult = modelResult.isolateMaterial(0)

    # Create mesh data
    mesh1 = Mesh(modelResult)

    # Create plot
    plot1 = Plot(mesh1)

    # Process all events
    app1.processEvents()

    # Save screenshot of plot 1
    #plot1.export('voxel-tools-bool-fig1.png')

    # Export mesh
    #mesh1.export('result.stl')

    app1.exec_()
