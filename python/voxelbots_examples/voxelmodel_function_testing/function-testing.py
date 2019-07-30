"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import PyQt5.QtGui as qg
import sys
import numpy as np

from voxelbots.voxel_model import VoxelModel
from voxelbots.mesh import Mesh
from voxelbots.plot import Plot
from voxelbots.materials import materials

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    # Basic Operations ##########################################################
    model1 = VoxelModel.fromVoxFile('cylinder-blue.vox', 0, 0, 0)
    #model2 = VoxelModel.fromVoxFile('cylinder-red.vox', 0, 5, 0)
    #model3 = VoxelModel.fromVoxFile('cylinder-red.vox', 0, 0, 0)
    #model4 = model1+model3
    #model4 = model4.scaleValues()
    #model5 = VoxelModel.fromFile('blurregion.vox', 0, 0, 0)

    modelResult = model1
    #modelResult = model3.getUnoccupied().setMaterial(0) + model3
    #modelResult = model3.isolateLayer(2)
    #modelResult = model4 - model2
    #modelResult = modelResult.removeNegatives()
    #modelResult = model1.difference(model2)
    #region = model1.intersection(model2)
    #modelResult = model1.union(model2)
    #modelResult = model1.add(model2)
    #modelResult = modelResult.isolateMaterial(2)
    #modelResult = model1.dilate(radius=3, connectivity=2)
    #modelResult = model1.erode(radius=2, connectivity=1)
    #modelResult = modelResult.blur(1)
    #modelResult = modelResult.blurRegion(3, region)
    #modelResult = model4.rotate(90, 'x')

    interiorVoxels = model1.erode(radius=1, connectivity=1)
    exteriorVoxels = model1.difference(interiorVoxels)
    modelResult = exteriorVoxels.isolateLayer(6)

    # Manufacturing Feature Generation ##########################################
    #model1 = VoxelModel.fromVoxFile('sample-object-2.vox', 0, 0, 0)
    #model2 = VoxelModel.fromVoxFile('user-support-2.vox', 0, 0, 0)

    #modelResult = model1.keepout(method='mill')
    #modelResult = model1.clearance(method='3dp')
    #modelResult = model1.support('laser')

    #support = model1.userSupport(model2, 'laser')
    #web = model1.web('laser', 1, 5)
    #modelResult = support.union(web)
    #modelResult = modelResult.setMaterial(2)
    #modelResult = modelResult.isolateLayer(8)
    #modelResult = model1.union(modelResult)

    #modelResult = modelResult.normalize()

    # Create mesh data
    #mesh1 = Mesh(model1)
    mesh2 = Mesh.fromVoxelModel(modelResult)

    # Create plot
    #plot1 = Plot(mesh1)
    plot2 = Plot(mesh2)
    plot2.show()
    app1.processEvents()

    #plot2.export('voxel-tools-bool-fig1.png')

    # Export mesh
    #mesh1.export('result.stl')

    app1.exec_()
