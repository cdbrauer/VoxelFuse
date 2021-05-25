"""
Multimaterial 3D printing with blurring/dithering.

User inputs a model and the program will apply blurring to requested materials.

- boxes.vox demonstrates two blurred materials and one non-blurred material.
- joint2.1.vox demonstrates a dog bone joint with blurring.

----

Copyright 2019 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh

if __name__=='__main__':
    # User preferences
    modelName = 'boxes.vox'
    #modelName = 'joint2.1.vox'
    blurMaterials = [1, 2] # materials to blur
    blurRadius = 3

    # Import model
    modelIn = VoxelModel.fromVoxFile(modelName)

    # Isolate materials with blurring requested
    modelBlur = VoxelModel.emptyLike(modelIn)
    for m in blurMaterials:
        modelBlur = modelBlur + modelIn.isolateMaterial(m)

    # Blur compatible materials
    # modelBlur = modelBlur.dither(blurRadius)
    modelBlur = modelBlur.blur(blurRadius)

    # Add unmodified voxels to result
    modelResult = modelBlur.union(modelIn)

    # Clean up result
    modelResult = modelResult.scaleValues()

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(modelIn)
    mesh2 = Mesh.fromVoxelModel(modelResult)

    # Create plots
    # mesh1.viewer(name='Input')
    mesh2.viewer(name='Output')