"""
Copyright 2018
Dan Aukes, Cole Brauer

Generate coupon for tensile testing
"""

import PyQt5.QtGui as qg
import sys
import numpy as np
from voxelbots.voxel_model import VoxelModel
from voxelbots.mesh import Mesh
from voxelbots.plot import Plot
from voxelbots.linkage import Linkage
from voxelbots.materials import materials

if __name__=='__main__':
    # Settings
    stl = False
    tabs = True
    blur = True
    blurRadius = 2
    mold = True
    simpleMold = False
    moldWallThickness = 2
    moldGap = 1

    app1 = qg.QApplication(sys.argv)

    # Import coupon components
    if stl:
        # TODO: Improve dimensional accuracy of stl model import and use these files instead of vox file
        end1 = VoxelModel.fromMeshFile('end1.stl', 0, 0, 0)
        center = VoxelModel.fromMeshFile('center.stl', 67, 3, 0)
        end2 = VoxelModel.fromMeshFile('end2.stl', 98, 0, 0)

        # Set materials
        end1 = end1.setMaterial(1)
        end2 = end2.setMaterial(1)
        center = center.setMaterial(2)

        # Combine components
        coupon = Linkage.copy(end1.union(center.union(end2)))
    else: # use vox file
        coupon = Linkage.fromVoxFile('coupon.vox') # Should use materials 1 and 2 (red and green)

    # Apply effects to material interface
    if tabs: # Add connecting tabs
        # Load tab template
        tab_template = VoxelModel.fromVoxFile('tab_puzzle.vox')

        # Set tab locations
        xVals = [79, 46]
        yVals = [9, 9]
        rVals = [0, 2]

        # Generate tabs
        coupon = coupon.insertTabs(tab_template, xVals, yVals, rVals)

    if blur: # Blur materials
        coupon = coupon.blur(blurRadius)
        coupon = coupon.scaleValues()

    if mold and not simpleMold: # Generate mold feature around material 2
        # Find all voxels containing <50% material 2
        material_vector = np.zeros(len(materials) + 1)
        material_vector[0] = 1
        material_vector[3] = 0.5
        printed_components = coupon - coupon.setMaterialVector(material_vector)
        printed_components.model = np.around(printed_components.model, 0)
        printed_components = printed_components.scaleValues()

        # Find voxels containing >50% material 2
        cast_components = coupon.difference(printed_components)

        # Generate mold body
        mold_model = cast_components.dilate(moldWallThickness+1, plane='xy')
        mold_model = mold_model.difference(cast_components)

        # Find clearance to prevent mold from sicking to model
        mold_clearance = printed_components.dilate(moldGap, plane='xy')

        # Apply clearance to body
        mold_model = mold_model.difference(mold_clearance)

        # Add mold to coupon model
        mold_model = mold_model.setMaterial(3)
        coupon = coupon.union(mold_model)

    if simpleMold: # Generate a basic mold feature around material 2
        printed_components = coupon.isolateMaterial(1)
        cast_components = coupon.isolateMaterial(2)

        # Generate mold body
        mold_model = cast_components.dilate(moldWallThickness + 1, plane='y')
        mold_model = mold_model.difference(cast_components)

        # Find clearance to prevent mold from sicking to model
        mold_clearance = printed_components.dilate(moldGap, plane='y')

        # Apply clearance to body
        mold_model = mold_model.difference(mold_clearance)

        # Add mold to coupon model
        mold_model = mold_model.setMaterial(3)
        coupon = coupon.union(mold_model)

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(coupon)

    # Create plot
    plot1 = Plot(mesh1, grids=True)
    plot1.show()

    app1.processEvents()
    app1.exec_()