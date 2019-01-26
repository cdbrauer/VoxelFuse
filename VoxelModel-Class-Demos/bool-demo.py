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
    model3 = VoxelModel.fromFile('sphere-blue.vox', -4, 4, 4)
    model4 = model2 + model3

    #model1 = VoxelModel.fromFile('square-blue-L.vox', 0, 0, 0)
    #model2 = VoxelModel.fromFile('square-red-R.vox', 0, 0, 0)

    # Perform Boolean Operations ###############################################
    #modelResult = model3.isolateLayer(2)
    modelResult = model2 + model1
    #modelResult = model4 - model1
    #modelResult = model4.subtractVolume(model1)
    #modelResult = model4.intersectVolume(model1)
    #modelResult = model4.intersectMaterial(model1)
    #modelResult = model1.addVolume(model2)
    #modelResult = modelResult.isolateMaterial(2)
    #modelResult = model4.invert()
    #modelResult = model4.xor(model1)
    #modelResult = model4.dilate()
    #modelResult = model4.erode()
    modelResult = modelResult.blur(region='all', threshold=0.5)

    normalizedResult = modelResult.normalize()

    # Create mesh data
    mesh1 = Mesh(modelResult)
    mesh2 = Mesh(normalizedResult)

    # Create plot
    plot1 = Plot(mesh1)
    plot2 = Plot(mesh2)

    # Process all events
    app1.processEvents()

    # Save screenshot of plot 1
    plot1.export('voxel-tools-bool-fig1.png')
    plot2.export('voxel-tools-bool-fig2.png')

    # Export mesh
    #mesh1.export('result.stl')

    app1.exec_()
