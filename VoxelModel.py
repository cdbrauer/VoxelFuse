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

    # Selection operations #############################################################
    # Get all voxels with a specified material
    def isolateMaterial(self, material):
        material_vector = np.zeros(len(materials))
        material_vector[material] = 1
        mask = (self.model == material_vector).all(3)
        mask = np.repeat(mask[..., None], len(materials), axis=3)
        new_model = np.multiply(self.model, mask)
        return VoxelModel(new_model, self.x, self.y, self.z)

    # Get all voxels in a specified layer
    def isolateLayer(self, layer):
        new_model = np.zeros_like(self.model)
        new_model[:, layer, :, :] = self.model[:, layer, :, :]
        return VoxelModel(new_model, self.x, self.y, self.z)

    # Boolean operations ###############################################################
    # Material from base model takes priority in volume operations
    def addVolume(self, model_to_add):
        a, b = alignDims(self, model_to_add)
        mask = (a.model == np.zeros(len(materials))).all(3)
        mask = np.repeat(mask[..., None], len(materials), axis=3)
        new_model = np.multiply(b.model, mask)
        new_model = new_model + a.model
        return VoxelModel(new_model, a.x, a.y, a.z)

    def addMaterial(self, model_to_add):
        a, b = alignDims(self, model_to_add)
        new_model = a.model + b.model
        return VoxelModel(new_model, a.x, a.y, a.z)

    def __add__(self, other):
        return self.addMaterial(other)

    def subtractVolume(self, model_to_sub):
        a, b = alignDims(self, model_to_sub)
        mask = (b.model == np.zeros(len(materials))).all(3)
        mask = np.repeat(mask[..., None], len(materials), axis=3)
        new_model = np.multiply(a.model, mask)
        return VoxelModel(new_model, a.x, a.y, a.z)

    def subtractMaterial(self, model_to_sub):
        a, b = alignDims(self, model_to_sub)
        new_model = a.model - b.model
        new_model[new_model < 0] = 0
        return VoxelModel(new_model, a.x, a.y, a.z)

    def __sub__(self, other):
        return self.subtractMaterial(other)

    def intersectVolume(self, model_2):
        a, b = alignDims(self, model_2)
        mask1 = (a.model != np.zeros(len(materials))).any(3)
        mask2 = (b.model != np.zeros(len(materials))).any(3)
        mask = np.multiply(mask1, mask2)
        mask = np.repeat(mask[..., None], len(materials), axis=3)
        new_model = np.multiply(a.model, mask)
        return VoxelModel(new_model, a.x, a.y, a.z)

    def intersectMaterial(self, model_2):
        a, b = alignDims(self, model_2)
        overlap = np.multiply(a.model, b.model)
        overlap[overlap > 0] = 1
        new_model = np.multiply(a.model, overlap) + np.multiply(b.model, overlap)
        return VoxelModel(new_model, a.x, a.y, a.z)

    def xor(self, model_2):
        a, b = alignDims(self, model_2)
        mask1 = (a.model == np.zeros(len(materials))).all(3)
        mask2 = (b.model == np.zeros(len(materials))).all(3)
        mask1 = np.repeat(mask1[..., None], len(materials), axis=3)
        mask2 = np.repeat(mask2[..., None], len(materials), axis=3)
        a.model = np.multiply(a.model, mask2)
        b.model = np.multiply(b.model, mask1)
        new_model = a.model + b.model
        return VoxelModel(new_model, a.x, a.y, a.z)