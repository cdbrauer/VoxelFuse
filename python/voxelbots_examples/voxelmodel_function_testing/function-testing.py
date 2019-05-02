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

    model1 = VoxelModel.fromFile('cylinder-blue.vox', 0, 0, 0)
    model2 = VoxelModel.fromFile('cylinder-red.vox', 0, 5, 0)
    model3 = VoxelModel.fromFile('cylinder-red.vox', 0, 0, 0)
    #model4 = model1.addMaterial(model3)
    #model4 = model4.normalize()
    model5 = VoxelModel.fromFile('blurregion.vox', 0, 0, 0)
    #modelResult = model3.isolateLayer(2)
    #modelResult = model4 - model2
    #modelResult = model1.addVolume(model2)
    #modelResult = model1.intersectVolume(model2)
    #modelResult = model4.intersectMaterial(model1)
    #modelResult = model4.addMaterial(model2)
    #modelResult = modelResult.isolateMaterial(2)
    #modelResult = model4.invert()
    #modelResult = model4.xor(model1)
    #modelResult = model1.dilate()
    #modelResult = model1.erode()

    #model1 = VoxelModel.fromFile('square-blue-L.vox', 0, 0, 0)
    #model2 = VoxelModel.fromFile('square-red-R.vox', 0, 0, 0)
    modelCombined = model1.addMaterial(model2)
    modelCombined = modelCombined.normalize()
    modelBlur = modelCombined.blur(radius=1, region='all')
    #modelBlur = modelBlur.intersectVolume(model5)
    modelResult = modelBlur#.addVolume(modelCombined)
    modelResult = modelResult.normalize()

    #model1 = VoxelModel.fromFile('sample-object-2.vox', 0, 0, 0)
    #model2 = VoxelModel.fromFile('user-support-2.vox', 0, 0, 0)
    #modelResult = model1.keepout(method='laser')
    #modelResult = model1.clearance(method='laser')
    #modelResult = model1.web('mill', 9, 1, 5)
    #modelResult = model1.support('laser')
    #modelResult = model1.mergeSupport(model2, 'laser')
    #modelResult = modelResult.setMaterial(0)
    #modelResult = model1.fixture(12, 1, 5)
    #modelResult = model1.addVolume(modelResult)

    #modelResult = modelResult.normalize()

    # Create mesh data
    #mesh1 = Mesh(model1)
    mesh2 = Mesh(modelResult)

    # Create plot
    #plot1 = Plot(mesh1)
    plot2 = Plot(mesh2)
    app1.processEvents()

    plot2.export('voxel-tools-bool-fig1.png')

    # Export mesh
    #mesh1.export('result.stl')

    app1.exec_()
