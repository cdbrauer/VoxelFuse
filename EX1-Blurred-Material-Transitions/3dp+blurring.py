"""
Copyright 2018
Dan Aukes, Cole Brauer

Example 1 - Multimaterial 3D printing with blurring

User inputs a model and the program will automatically
apply blurring to materials with blurring defined in
the materials file

three-boxes.vox demonstrates two blurred materials and one non-blurred material
joint2.1.vox demonstrates a dog bone joint with blurring

"""

import PyQt5.QtGui as qg
import sys
import numpy as np

from VoxelModel import VoxelModel
from Mesh import Mesh
from Plot import Plot

from materials import materials

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    # User preferences
    # modelName = 'three-boxes.vox'
    modelName = 'joint2.1.vox'
    blur = True
    blurRadius = 3

    # Import model
    modelIn = VoxelModel.fromFile(modelName)

    # Initialize object to hold result
    modelResult = VoxelModel.emptyLike(modelIn)

    # If blurring was requested
    if blur:
        # Isolate materials compatible with blurring
        modelBlur = VoxelModel.emptyLike(modelIn)
        for i in range(len(materials)):
            if materials[i]['blur']:
                modelBlur = modelBlur + modelIn.isolateMaterial(i)

        # Blur compatible materials
        modelBlur = modelBlur.blur(blurRadius)

        # Add to result
        modelResult = modelBlur + modelResult

    # Add unmodified voxels to result
    modelResult = modelResult + modelIn

    # Create list of all materials present in model

    # Sort by material percentages

    # Create mesh data for each material
    mesh1 = Mesh(modelResult)

    # Export .stl file for each material

    # Create plot
    plot1 = Plot(mesh1)
    app1.processEvents()
    plot1.export('fig1.png')

    app1.exec_()