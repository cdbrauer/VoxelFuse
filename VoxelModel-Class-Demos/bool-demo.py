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

    #model1 = VoxelModel.fromFile('sphere-blue.vox', 0, 0, 0)
    #model2 = VoxelModel.fromFile('sphere-red.vox', -4, 4, 4)
    #model3 = VoxelModel.fromFile('sphere-blue.vox', -4, 4, 4)
    #model4 = model2 + model3
    #modelResult = model3.isolateLayer(2)
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

    #model1 = VoxelModel.fromFile('square-blue-L.vox', 0, 0, 0)
    #model2 = VoxelModel.fromFile('square-red-R.vox', 0, 0, 0)
    #modelResult = model1 + model2
    #modelResult = modelResult.blur(region='object', threshold=0)

    model1 = VoxelModel.fromFile('sample-object-1.vox', 0, 0, 0)
    #model2 = VoxelModel.fromFile('user-support-1.vox', 0, 0, 0)
    #modelResult = model1.keepout(method='laser')
    #modelResult = model1.clearance(method='mill')
    modelResult = model1.web('laser', 25, 1, 5)
    #modelResult = model1.support('laser')
    #modelResult = model1.mergeSupport(model2, 'laser')
    modelResult = modelResult.setMaterial(0)
    modelResult = model1.addVolume(modelResult)
    
    #modelResult = modelResult.normalize()

    # Create mesh data
    mesh1 = Mesh(modelResult)

    # Create plot
    plot1 = Plot(mesh1)
    app1.processEvents()
    #plot1.export('voxel-tools-bool-fig1.png')

    # Export mesh
    #mesh1.export('result.stl')

    app1.exec_()
