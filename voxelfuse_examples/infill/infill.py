"""
Infill Generation.

Generate a part containing a shell of one material, and infill of another material.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import sphere
from voxelfuse.periodic import gyroid
from voxelfuse.mesh import Mesh

if __name__=='__main__':
    # User preferences
    modelRadius = 60
    infillScale = 30
    infillThickness = 1
    shellThickness = 1

    # Create volume model
    volume = sphere(radius=modelRadius, material=3)
    volume = volume.setCoords((0,0,0))

    # Create infill structure
    infillN, infillP = gyroid(size=(modelRadius*2, modelRadius*2, modelRadius*2), scale=infillScale)
    infillN = infillN.dilate(infillThickness)
    infillP = infillP.dilate(infillThickness)
    infill = infillN & infillP

    # Hollow out volume model
    hollow = volume.erode(shellThickness)
    shell = volume.difference(hollow)

    # Trim infill model
    infill = infill & hollow

    # Combine infill and shell
    result = shell | infill

    # Create mesh data
    for m in range(1, len(result.materials)):
        currentMaterial = result.isolateMaterial(m)
        currentMesh = Mesh.fromVoxelModel(currentMaterial)
        currentMesh.export('output_' + str(m) + '.stl')