"""
Create a VoxelModel from a binary array and use it to generate an STL file

Based on an answer to this StackExchange question: https://3dprinting.stackexchange.com/questions/10205/convert-a-3d-numpy-array-of-voxels-to-an-stl-file
"""

import voxelfuse as vf
import numpy as np

if __name__ == '__main__':
    # Define the voxel array
    sponge = np.array([
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ],
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
    ])

    model = vf.VoxelModel(sponge) # Create a VoxelModel object

    mesh = vf.Mesh.fromVoxelModel(model) # Convert VoxelModel to a Mesh
    mesh = mesh.setResolution(0.1)  # Change the resolution (in vx/mm) to make the output model larger
    mesh.export('mesh.stl') # Save the mesh to an stl file
