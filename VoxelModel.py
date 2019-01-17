"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
from pyvox.parser import VoxParser
from materials import materials

"""
Function to make object dimensions compatible for solid body operations.  Takes location coordinates into account.
"""
def alignDims(modelA, modelB):
    xMaxA = modelA.x + len(modelA.model[0, 0, :, 0])
    yMaxA = modelA.y + len(modelA.model[:, 0, 0, 0])
    zMaxA = modelA.z + len(modelA.model[0, :, 0, 0])

    xMaxB = modelB.x + len(modelB.model[0, 0, :, 0])
    yMaxB = modelB.y + len(modelB.model[:, 0, 0, 0])
    zMaxB = modelB.z + len(modelB.model[0, :, 0, 0])

    xNew = min(modelA.x, modelB.x)
    yNew = min(modelA.y, modelB.y)
    zNew = min(modelA.z, modelB.z)

    xMaxNew = max(xMaxA, xMaxB)
    yMaxNew = max(yMaxA, yMaxB)
    zMaxNew = max(zMaxA, zMaxB)

    modelANew = np.zeros((yMaxNew - yNew, zMaxNew - zNew, xMaxNew - xNew, len(modelA.model[0, 0, 0, :])))
    modelBNew = np.zeros((yMaxNew - yNew, zMaxNew - zNew, xMaxNew - xNew, len(modelB.model[0, 0, 0, :])))

    modelANew[(modelA.y - yNew):(yMaxA - yNew), (modelA.z - zNew):(zMaxA - zNew), (modelA.x - xNew):(xMaxA - xNew), :] = modelA.model
    modelBNew[(modelB.y - yNew):(yMaxB - yNew), (modelB.z - zNew):(zMaxB - zNew), (modelB.x - xNew):(xMaxB - xNew), :] = modelB.model

    return VoxelModel(modelANew, xNew, yNew, zNew), VoxelModel(modelBNew, xNew, yNew, zNew)


"""
VoxelModel Class

Initialized from a model array or file and position coordinates

Properties:
  model - array storing the material type present at each voxel
  x_coord, y_coord, z_coord - position of model origin
"""
class VoxelModel:
    def __init__(self, model, x_coord = 0, y_coord = 0, z_coord = 0):
        self.model = model
        self.x = x_coord
        self.y = y_coord
        self.z = z_coord

    @classmethod
    def fromFile(cls, filename, x_coord = 0, y_coord = 0, z_coord = 0):
        m1 = VoxParser(filename).parse()
        m2 = m1.to_dense()
        m2 = np.flip(m2, 1)

        x_len = len(m2[0, 0, :])
        y_len = len(m2[:, 0, 0])
        z_len = len(m2[0, :, 0])

        new_model = np.zeros((y_len, z_len, x_len, len(materials)))

        # Loop through input_model data
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    color_index = m2[y, z, x] - 1
                    if (color_index >= 0) and (color_index < len(materials)):
                        new_model[y, z, x, color_index] = 1
                    elif color_index >= len(materials):
                        print('Unrecognized material index: ' + str(color_index))

        return cls(new_model, x_coord, y_coord, z_coord)

    def union(self, model_to_add):
        a, b = alignDims(self, model_to_add)
        new_model = a.model + b.model
        #new_model[new_model > 1] = 1 # move to mesh code
        return VoxelModel(new_model, a.x, a.y, a.z)

    def __add__(self, other):
        return self.union(other)

    def difference(self, model_to_sub):
        a, b = alignDims(self, model_to_sub)
        new_model = a.model - b.model
        #new_model[new_model < 0] = 0 # same here?
        return VoxelModel(new_model, a.x, a.y, a.z)

    def __sub__(self, other):
        return self.difference(other)

    def intersection(self, model_2):
        a, b = alignDims(self, model_2)

        new_model = np.zeros_like(a.model)

        x_len = len(a.model[0, 0, :, 0])
        y_len = len(a.model[:, 0, 0, 0])
        z_len = len(a.model[0, :, 0, 0])

        # Loop through input_model data
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    material_count_a = sum(a.model[y, z, x, :])
                    material_count_b = sum(b.model[y, z, x, :])

                    if (material_count_a > 0) and (material_count_b > 0):
                        new_model[y, z, x, :] = a.model[y, z, x, :] + b.model[y, z, x, :]

        new_model[new_model > 1] = 1
        return VoxelModel(new_model, a.x, a.y, a.z)
