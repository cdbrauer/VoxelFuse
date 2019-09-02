"""
Copyright 2019
Dan Aukes, Cole Brauer

Joint with Embedded Servo

Program adds clearances for inserted components, determines where pauses must be inserted in the gcode,
and exports an stl file of the modified part
"""

import PyQt5.QtGui as qg
import sys
import numpy as np
import time

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.plot import Plot
from voxelfuse.materials import materials

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    # User preferences
    modelName = 'joint-2.vox'
    # modelName = 'tail-holder-1r.vox'
    clearance = [1, 3] # materials to leave clearance around

    # Import model
    start = time.time()
    modelIn = VoxelModel.fromVoxFile(modelName)
    end = time.time()
    importTime = (end - start)

    start = time.time()

    # Rotate to best orientation for printing
    modelIn = modelIn.rotate90(axis='y')

    # Initialize object to hold result
    modelResult = VoxelModel.copy(modelIn)

    # Subtract clearance shell around specified materials
    for i in range(len(clearance)):
        modelResult = modelResult.difference(modelIn.isolateMaterial(clearance[i]).dilate().difference(modelIn.isolateMaterial(clearance[i])))

    # Initialize object to hold inserted components
    insertedComponents = VoxelModel.emptyLike(modelResult)

    # Find inserted components
    for m in range(len(materials)):
        if materials[m]['process'] == 'ins':
            insertedComponents = insertedComponents.union(modelIn.isolateMaterial(m).dilate())

    # Find clearance for inserted components
    insertedComponentsClearance = insertedComponents.clearance('mill')

    # Find pause layers at top of each inserted component
    for z in range(1, len(insertedComponents.model[0, :, 0, 0])):
        if np.sum(insertedComponents.model[:, z, :, 0]) == 0:

            # Remove clearance between parts
            insertedComponentsClearance.model[:, z, :, :].fill(0)

            # Identify tops of parts
            if np.sum(insertedComponents.model[:, z-1, :, 0]) > 0:
                print(z-1) # TODO: replace with save to array

    insertedComponents = insertedComponents.union(insertedComponentsClearance)

    # Remove inserted components
    modelResult = modelResult.difference(insertedComponents)

    end = time.time()
    processingTime = (end - start)

    # Create mesh data
    start = time.time()
    mesh1 = Mesh.fromVoxelModel(modelIn)
    mesh2 = Mesh.fromVoxelModel(modelResult)
    end = time.time()
    meshingTime = (end - start)

    # Export .stl file
    #mesh1.export('joint-2.stl')
    #mesh1.export('tail-holder.stl')

    # TODO: Slice .stl file

    # Insert pauses at top of each inserted component
    # TODO: Integrate gcode_pause.py

    # Print elapsed times
    print("Import time = %s" % importTime)
    print("Processing time = %s" % processingTime)
    print("Meshing time = %s" % meshingTime)

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