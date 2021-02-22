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
    modelBlur = modelBlur.scaleValues()

    # Apply dither
    modelDither = modelBlur.dither()

    # Create mesh data
    for m in range(1, len(modelIn.materials)):
        currentMaterial = modelIn.isolateMaterial(m)
        currentMesh = Mesh.fromVoxelModel(currentMaterial)
        currentMesh.export('input_' + str(m) + '.stl')

    modelBlur = modelBlur.round(0.1).removeDuplicateMaterials()
    for m in range(1, len(modelBlur.materials)):
        currentMaterial = modelBlur.isolateMaterial(m)
        currentMesh = Mesh.fromVoxelModel(currentMaterial)
        currentMesh.export('blur_' + str(m) + '.stl')

    for m in range(1, len(modelDither.materials)):
        currentMaterial = modelDither.isolateMaterial(m)
        currentMesh = Mesh.fromVoxelModel(currentMaterial)
        currentMesh.export('dither_' + str(m) + '.stl')