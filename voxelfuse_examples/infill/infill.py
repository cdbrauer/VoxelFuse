"""
Infill Generation.

Generate a part containing a shell of one material, and infill of another material.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import PyQt5.QtGui as qg
import sys
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.plot import Plot
from voxelfuse.primitives import sphere
from voxelfuse.periodic import gyroid
from voxelfuse.mesh import Mesh

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    # User preferences
    modelRadius = 60
    infillScale = 30
    infillThickness = 1
    shellThickness = 1

    # Create volume model
    volume = sphere(radius=modelRadius, material=1)
    volume = volume.setCoords((0,0,0))

    # Display
    # mesh1 = Mesh.fromVoxelModel(volume)
    # plot1 = Plot(mesh1, name='Input Model')
    # plot1.show()
    # app1.processEvents()

    # Create infill structure
    infillN, infillP = gyroid(size=(modelRadius*2, modelRadius*2, modelRadius*2), scale=infillScale, material1=2, material2=2)
    infillN = infillN.dilate(infillThickness)
    infillP = infillP.dilate(infillThickness)
    infill = infillN & infillP

    # Display
    # mesh2 = Mesh.fromVoxelModel(infillN.setMaterial(3) | infillP)
    # plot2 = Plot(mesh2, name='Infill Volume Halves')
    # plot2.show()
    # app1.processEvents()

    # Display
    # mesh3 = Mesh.fromVoxelModel(infill)
    # plot3 = Plot(mesh3, name='Infill Surface')
    # plot3.show()
    # app1.processEvents()

    # Hollow out volume model
    hollow = volume.erode(shellThickness)
    shell = volume.difference(hollow)

    # Trim infill model
    infill = infill & hollow

    # Display
    # mesh4 = Mesh.fromVoxelModel(infill)
    # plot4 = Plot(mesh4, name='Trimmed Infill')
    # plot4.show()
    # app1.processEvents()

    # Combine infill and shell
    result = shell | infill

    # Create mesh data
    for m in range(1, len(result.materials)):
        currentMaterial = result.isolateMaterial(m)
        currentMesh = Mesh.fromVoxelModel(currentMaterial)
        currentMesh.export('output_' + str(m) + '.stl')

    # app1.exec_()