"""
Import a mesh file and convert it to voxels.

----

Copyright 2019 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh

if __name__=='__main__':
    # Import model file
    model = VoxelModel.fromMeshFile('axes.stl', (0, 0, 0))

    # To use a copy of gmsh on the system path, replace line 21 with the following:
    # model = VoxelModel.fromMeshFile('axes.stl', (0, 0, 0), gmsh_on_path=True)

    # Create new mesh data for plotting
    mesh1 = Mesh.fromVoxelModel(model)

    # Create plot
    mesh1.viewer(grids=True)