"""
Copyright 2018
Dan Aukes, Cole Brauer

Example 3 - Joint with Embedded Servo

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
    modelName = 'joint-2.vox'
#    modelName = 'tail-holder-1r.vox'
    clearance = [1, 3] # materials to leave clearance around

    # Import model
    modelIn = VoxelModel.fromFile(modelName)

    # Rotate to best orientation for printing
    modelIn = modelIn.rotate(90, 'y')

    # Initialize object to hold result
    modelResult = VoxelModel.copy(modelIn)

    # Subtract clearance shell around specified materials
    for i in range(len(clearance)):
        modelResult = modelResult - (modelIn.isolateMaterial(clearance[i]).dilate() - modelIn.isolateMaterial(clearance[i]))

    # Initialize object to hold inserted components
    insertedComponents = VoxelModel.emptyLike(modelResult)

    # Find inserted components
    for m in range(len(materials)):
        if materials[m]['process'] == 'ins':
            insertedComponents = insertedComponents + modelIn.isolateMaterial(m).dilate()

    # Find clearance for inserted components
    insertedComponentsClearance = insertedComponents.clearance('mill')

    # Find pause layers at top of each inserted component
    for z in range(1, len(insertedComponents.model[0, :, 0, 0])):
        if np.sum(insertedComponents.model[:, z, :, :]) == 0:

            # Remove clearance between parts
            insertedComponentsClearance.model[:, z, :, :].fill(0)

            # Identify tops of parts
            if np.sum(insertedComponents.model[:, z-1, :, :]) > 0:
                print(z-1) # replace with save to array

    insertedComponents = insertedComponents + insertedComponentsClearance

    # Remove inserted components
    modelResult = modelResult - insertedComponents

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(modelResult)

    # Export .stl file
    #mesh1.export('joint-2.stl')
    mesh1.export('tail-holder.stl')

    # Slice .stl file

    # Insert pauses at top of each inserted component

    # Create plot
    plot1 = Plot(mesh1)
    plot1.show()
    app1.processEvents()
    plot1.export('fig1.png')

    app1.exec_()