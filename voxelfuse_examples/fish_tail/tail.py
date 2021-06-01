"""
Multimaterial fish tail.

Apply blurring to the specified materials.

----

Copyright 2019 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel, Axes
from voxelfuse.mesh import Mesh

if __name__=='__main__':
    # User preferences
    modelName = 'tail-20.vox'
    blurMaterials = [1, 2] # materials to blur
    blurRadius = 5

    # Import model
    modelIn = VoxelModel.fromVoxFile(modelName)

    # Rotate to best orientation for printing
    modelIn = modelIn.rotate90(axis=Axes.Y)

    # Isolate materials with blurring requested
    modelBlur = VoxelModel.emptyLike(modelIn)
    for m in blurMaterials:
        modelBlur = modelBlur.union(modelIn.isolateMaterial(m))

    # Blur compatible materials
    modelBlur = modelBlur.blur(blurRadius)

    # Add unmodified voxels to result
    modelResult = modelBlur.union(modelIn)

    # Clean up result
    modelResult = modelResult.scaleValues()
    modelResult = modelResult.round() # "posterize" result
    modelResult = modelResult.removeDuplicateMaterials()

    # Create mesh files
    # for m in range(1, len(modelResult.materials)):
    #     currentMaterial = modelResult.isolateMaterial(m)
    #     currentMesh = Mesh.fromVoxelModel(currentMaterial)
    #     currentMesh.export('output_' + str(m) + '.stl')

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(modelIn)
    mesh2 = Mesh.fromVoxelModel(modelResult)

    # Create plots
    # mesh1.viewer(name='Input')
    mesh2.viewer(name='Output')