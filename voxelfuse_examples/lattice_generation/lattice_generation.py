"""
Lattice structure generation.

Generate a lattice structure over a cylindrical region based on a template element file.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

# Import Libraries
import PyQt5.QtGui as qg
import sys
import numpy as np
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import cylinder
from voxelfuse.mesh import Mesh
from voxelfuse.plot import Plot

# Start Application
if __name__=='__main__':
    app1 = qg.QApplication(sys.argv)

    dilate_radius = 2

    # Import element template
    T = VoxelModel.fromVoxFile('lattice_element_1m.vox')
    T = T.dilateBounded(dilate_radius)

    # Create target volume model
    M = cylinder(30, 60).translate((30,30,0))
    M = M.setCoords((0,0,0))

    t = T.voxels.shape[0]
    size = M.voxels.shape

    # Create empty model to hold result
    V = VoxelModel.empty(size)

    # Add tiled lattice elements
    for x in range(int(np.ceil(size[0]/t))):
        for y in range(int(np.ceil(size[1] / t))):
            for z in range(int(np.ceil(size[2]/t))):
                T = T.setCoords((M.coords[0]+(x*t), M.coords[1]+(y*t), M.coords[2]+(z*t)))
                V = V | T

    # Intersect lattice with target volume
    V = M & V

    # Create Mesh
    mesh1 = Mesh.fromVoxelModel(T)
    mesh2 = Mesh.fromVoxelModel(M)
    mesh3 = Mesh.fromVoxelModel(V)

    mesh1.export('T.stl')
    mesh2.export('M.stl')
    mesh3.export('V.stl')

    # Create Plot
    plot3 = Plot(mesh3)
    plot3.show()
    app1.processEvents()
    app1.exec_()