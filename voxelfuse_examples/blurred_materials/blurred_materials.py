"""
Copyright 2019
Dan Aukes, Cole Brauer

Multimaterial 3D printing with blurring/dithering

User inputs a model and the program will apply blurring to requested materials

boxes.vox demonstrates two blurred materials and one non-blurred material
joint2.1.vox demonstrates a dog bone joint with blurring
"""

import PyQt5.QtGui as qg
import sys

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.plot import Plot

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    # User preferences
    modelName = 'boxes.vox'
    #modelName = 'joint2.1.vox'
    blur = [1, 2] # materials to blur
    blurRadius = 3

    # Import model
    modelIn = VoxelModel.fromVoxFile(modelName)

    # Isolate materials with blurring requested
    modelBlur = VoxelModel.emptyLike(modelIn)
    for i in blur:
        modelBlur = modelBlur + modelIn.isolateMaterial(i)

    # Blur compatible materials
    # modelBlur = modelBlur.dither(blurRadius)
    modelBlur = modelBlur.blur(blurRadius)

    # Add unmodified voxels to result
    modelResult = modelBlur.union(modelIn)

    # Clean up result
    modelResult = modelResult.scaleValues()

    # TODO: Create list of all materials present in model

    # TODO: Sort by material percentages (posterize)

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(modelIn)
    mesh2 = Mesh.fromVoxelModel(modelResult)

    # TODO: Export .stl file for each material

    # Create plots
    plot1 = Plot(mesh1)
    plot1.show()
    app1.processEvents()
    #plot1.export('input.png')

    plot2 = Plot(mesh2)
    plot2.show()
    app1.processEvents()
    #plot2.export('output.png')

    app1.exec_()