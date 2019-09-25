"""
Copyright 2018
Dan Aukes, Cole Brauer

Generate coupon for tensile testing
"""

import PyQt5.QtGui as qg
import sys
import time
import numpy as np
from voxelfuse.voxel_model import VoxelModel, Axes
from voxelfuse.mesh import Mesh
from voxelfuse.plot import Plot
from voxelfuse.materials import material_properties

if __name__=='__main__':
    # Settings
    stl = True
    highRes = False

    blur = False
    blurRadius = 2

    mold = True
    moldWallThickness = 2
    moldGap = 1

    fixture = True
    fixtureWallThickness = 5
    fixtureGap = 1

    export = False

    app1 = qg.QApplication(sys.argv)

    # Import coupon components
    print('Importing Files')
    if stl:
        # TODO: Improve dimensional accuracy of stl model import and use these files instead of vox file
        if highRes:
            end1 = VoxelModel.fromMeshFile('end1x10.stl', (0, 0, 0))
            center = VoxelModel.fromMeshFile('centerx10.stl', (670, 30, 0))
            end2 = VoxelModel.fromMeshFile('end2x10.stl', (980, 0, 0))
        else:
            end1 = VoxelModel.fromMeshFile('end1.stl', (0, 0, 0))
            center = VoxelModel.fromMeshFile('center.stl', (67, 3, 0))
            end2 = VoxelModel.fromMeshFile('end2.stl', (98, 0, 0))

        # Set materials
        end1 = end1.setMaterial(1)
        end2 = end2.setMaterial(1)
        center = center.setMaterial(2)

        # Combine components
        coupon = VoxelModel.copy(end1.union(center.union(end2)))
    else: # use vox file
        coupon = VoxelModel.fromVoxFile('coupon.vox') # Should use materials 1 and 2 (red and green)

    start = time.time()

    if blur: # Blur materials
        print('Blurring')
        coupon = coupon.blur(blurRadius)
        coupon = coupon.scaleValues()

    if mold: # Generate mold feature around material 2
        print('Generating Mold')

        # Find all voxels containing <50% material 2
        material_vector = np.zeros(len(material_properties) + 1)
        material_vector[0] = 1
        material_vector[3] = 0.5
        printed_components = coupon - coupon.setMaterialVector(material_vector)
        printed_components.materials = np.around(printed_components.materials, 0)
        printed_components = printed_components.scaleValues()

        # Generate mold body
        mold_model = coupon.difference(printed_components).dilate(moldWallThickness+1, plane=Axes.XY)

        # Find clearance to prevent mold from sticking to model and apply clearance to body
        mold_model = mold_model.difference(printed_components.dilate(moldGap, plane=Axes.XY))

        if fixture: # Generate a fixture around the full part to support mold
            print('Generating Fixture')
            coupon = coupon.union(coupon.web('laser', 1, 5).setMaterial(3))

        # Add mold to coupon model
        coupon = coupon.union(mold_model.setMaterial(3))

    end = time.time()
    processingTime = (end - start)
    print("Processing time = %s" % processingTime)

    # Create mesh data
    print('Meshing')
    mesh1 = Mesh.fromVoxelModel(coupon)

    # Create plot
    print('Plotting')
    plot1 = Plot(mesh1, grids=True)
    plot1.show()
    app1.processEvents()

    if export:
        print('Exporting')

        # Get non-cast components
        # Find all voxels containing <50% material 2
        material_vector = np.zeros(len(material_properties) + 1)
        material_vector[0] = 1
        material_vector[3] = 0.5
        printed_components = coupon - coupon.setMaterialVector(material_vector)
        printed_components.materials = np.around(printed_components.materials, 0)
        printed_components = printed_components.scaleValues()
        printed_components = printed_components.setMaterial(1)

        mesh2 = Mesh.fromVoxelModel(printed_components)
        plot2 = Plot(mesh2, grids=True)
        plot2.show()
        app1.processEvents()
        mesh2.export('modified-coupon.stl')

    app1.exec_()