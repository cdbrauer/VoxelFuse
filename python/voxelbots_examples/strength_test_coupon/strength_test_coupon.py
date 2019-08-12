"""
Copyright 2018
Dan Aukes, Cole Brauer

Generate coupon for tensile testing
"""

import PyQt5.QtGui as qg
import sys
from voxelbots.voxel_model import VoxelModel
from voxelbots.mesh import Mesh
from voxelbots.plot import Plot
from voxelbots.linkage import Linkage

if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    # Import coupon components
    # TODO: Improve dimensional accuracy of stl model import and use these files instead of vox file
    #end1 = VoxelModel.fromMeshFile('end1.stl', 0, 0, 0)
    #center = VoxelModel.fromMeshFile('center.stl', 67, 3, 0)
    #end2 = VoxelModel.fromMeshFile('end2.stl', 98, 0, 0)
    coupon = Linkage.fromVoxFile('coupon.vox')
    tab = VoxelModel.fromVoxFile('tab_puzzle.vox')

    # Set center material
    # Enable when using stl files
    #center = center.setMaterial(2)

    # Combine components
    # Enable when using stl files
    #coupon = end1.union(center.union(end2))

    # Apply effect to material interface
    xVals = [79, 46]
    yVals = [9, 9]
    rVals = [0, 2]
    coupon = coupon.insertTabs(tab, xVals, yVals, rVals)
    coupon = coupon.blur(2)

    # Clean up model
    coupon = coupon.scaleValues()

    # Create mesh data
    mesh1 = Mesh.fromVoxelModel(coupon)

    # Create plot
    plot1 = Plot(mesh1, grids=True)
    plot1.show()

    app1.processEvents()
    app1.exec_()