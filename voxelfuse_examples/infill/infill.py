"""
Infill Generation.

Generate a part containing a shell of one material, and infill of another material.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import sphere
from voxelfuse.periodic import gyroid

if __name__=='__main__':
    # User preferences
    modelRadius = 60
    infillScale = 30
    infillThickness = 1
    shellThickness = 1

    # Create volume model
    volume = sphere(radius=modelRadius, material=1)
    volume = volume.setCoords((0,0,0))

    # Display
    mesh1 = Mesh.fromVoxelModel(volume)
    mesh1.viewer(name='Input Model')

    # Create infill structure
    infillN, infillP = gyroid(size=(modelRadius*2, modelRadius*2, modelRadius*2), scale=infillScale, material1=2, material2=2)
    infillN = infillN.dilate(infillThickness)
    infillP = infillP.dilate(infillThickness)
    infill = infillN & infillP

    # Display
    mesh2 = Mesh.fromVoxelModel(infillN.setMaterial(3) | infillP)
    mesh2.viewer(name='Infill Volume Halves')

    # Display
    mesh3 = Mesh.fromVoxelModel(infill)
    mesh3.viewer(name='Infill Surface')

    # Hollow out volume model
    hollow = volume.erode(shellThickness)
    shell = volume.difference(hollow)

    # Trim infill model
    infill = infill & hollow

    # Display
    mesh4 = Mesh.fromVoxelModel(infill)
    mesh4.viewer(name='Trimmed Infill')

    # Combine infill and shell
    result = shell | infill

    # Create mesh data
    # for m in range(1, len(result.materials)):
    #     currentMaterial = result.isolateMaterial(m)
    #     currentMesh = Mesh.fromVoxelModel(currentMaterial)
    #     currentMesh.export('output_' + str(m) + '.stl')