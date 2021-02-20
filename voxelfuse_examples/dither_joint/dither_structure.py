"""
Material Dithering.

Use dithering on an entire multi-material part.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import numpy as np
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh

if __name__=='__main__':
    # User preferences
    modelName = 'joint2.2.vox'

    # Import model
    modelIn = VoxelModel.fromVoxFile(modelName)

    # Define the material vector for the model
    materialVector = np.zeros(len(modelIn.materials[0]))
    materialVector[0] = 1
    materialVector[2] = 0.3
    materialVector[3] = 0.7

    # Apply material
    model = modelIn.setMaterialVector(materialVector)

    # Apply dither
    modelDither = model.dither()

    # Create mesh data
    for m in range(1, len(modelDither.materials)):
        currentMaterial = modelDither.isolateMaterial(m)
        currentMesh = Mesh.fromVoxelModel(currentMaterial)
        currentMesh.export('full_dither_' + str(m) + '.stl')