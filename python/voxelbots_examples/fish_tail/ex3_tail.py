"""
Copyright 2018
Dan Aukes, Cole Brauer

Example 3 - Multimaterial fish tail

User inputs a model and the program will apply blurring to requested materials
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

    # User preferences
    modelName = 'tail-20.vox'
    blur = [0, 2] # materials to blur
    blurRadius = 5

    # Import model
    modelIn = VoxelModel.fromFile(modelName)

    # Rotate to best orientation for printing ... need to figure out the 91 thing
    modelIn = modelIn.rotate(91, 'y')

    # Initialize object to hold result
    modelResult = VoxelModel.emptyLike(modelIn)

    # Isolate materials with blurring requested
    modelBlur = VoxelModel.emptyLike(modelIn)
    for i in range(len(blur)):
        modelBlur = modelBlur + modelIn.isolateMaterial(blur[i])

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