"""
Material Dithering.

Use dithering to add graded material transitions to a multi-material part.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh

if __name__=='__main__':
    # User preferences
    modelName = 'joint2.2.vox'
    blurRadius = 10

    # Import model
    modelIn = VoxelModel.fromVoxFile(modelName)

    # Apply blur
    modelBlur = modelIn.blur(blurRadius)

    # Apply dither
    modelDither = modelBlur.dither(blur=False) # Blur operation separated for clarity

    # Create mesh data
    for m in range(1, len(modelDither.materials)):
        currentMaterial = modelDither.isolateMaterial(m)
        currentMesh = Mesh.fromVoxelModel(currentMaterial)
        currentMesh.export('output_' + str(m) + '.stl')