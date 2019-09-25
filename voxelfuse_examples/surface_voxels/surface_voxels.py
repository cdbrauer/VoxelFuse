"""
Copyright 2019
Dan Aukes, Cole Brauer

Finds the surface voxels of a model and outputs a list of their coordinates

Used by the Mesh class to improve meshing speed
"""

from voxelfuse.voxel_model import VoxelModel

if __name__=='__main__':
    # User preferences
    modelName = 'cylinder-blue.vox'

    # Import model
    model1 = VoxelModel.fromVoxFile(modelName, (0, 0, 0))

    # Find exterior voxels
    interiorVoxels = model1.erode(radius=1, connectivity=1)
    exteriorVoxels = model1.difference(interiorVoxels)

    x_len = len(exteriorVoxels.voxels[:, 0, 0])
    y_len = len(exteriorVoxels.voxels[0, :, 0])
    z_len = len(exteriorVoxels.voxels[0, 0, :])

    # Create list of exterior voxel coordinates
    exterior_voxels = []
    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                if exteriorVoxels.voxels[x, y, z] != 0:
                    exterior_voxels.append([x, y, z])

    print(exterior_voxels)