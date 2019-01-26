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

    def nor(self, model_2):
        a, b = alignDims(self, model_2)
        mask1 = (a.model == np.zeros(len(materials))).all(3)
        mask2 = (b.model == np.zeros(len(materials))).all(3)
        mask = np.multiply(mask1, mask2)
        mask = np.repeat(mask[..., None], len(materials), axis=3)
        return VoxelModel(mask, a.x, a.y, a.z)

    def invert(self):
        mask = (self.model == np.zeros(len(materials))).all(3)
        mask = np.repeat(mask[..., None], len(materials), axis=3)
        return VoxelModel(mask, self.x, self.y, self.z)

    # Dilate and Erode #################################################################
    def dilate(self, radius = 1, plane = 'xyz'):
        x_len = len(self.model[0, 0, :, 0]) + (radius*2) + 2
        y_len = len(self.model[:, 0, 0, 0]) + (radius*2) + 2
        z_len = len(self.model[0, :, 0, 0]) + (radius*2) + 2

        current_model = np.zeros((y_len, z_len, x_len, len(materials)))
        current_model[radius+1:-(radius+1), radius+1:-(radius+1), radius+1:-(radius+1), :] = self.model

        new_model = np.zeros((y_len, z_len, x_len, len(materials)))

        for i in range(radius):
            for x in range(1, x_len-1):
                for y in range(1, y_len-1):
                    for z in range(1, z_len-1):
                        for m in range(len(materials)):
                            if plane == 'xy':
                                new_model[y, z, x, m] = np.max(current_model[y-1:y+2, z, x-1:x+2, m])

                            elif plane == 'xz':
                                new_model[y, z, x, m] = np.max(current_model[y, z-1:z+2, x-1:x+2, m])

                            elif plane == 'yz':
                                new_model[y, z, x, m] = np.max(current_model[y-1:y+2, z-1:z+2, x, m])

                            else:
                                new_model[y, z, x, m] = np.max(current_model[y-1:y+2, z-1:z+2, x-1:x+2, m])

            current_model = np.copy(new_model)

        return VoxelModel(current_model, self.x, self.y, self.z)

    def erode(self, radius = 1, plane = 'xyz'):
        x_len = len(self.model[0, 0, :, 0]) + 2
        y_len = len(self.model[:, 0, 0, 0]) + 2
        z_len = len(self.model[0, :, 0, 0]) + 2

        current_model = np.zeros((y_len, z_len, x_len, len(materials)))
        current_model[1:-1, 1:-1, 1:-1, :] = self.model

        new_model = np.zeros((y_len, z_len, x_len, len(materials)))

        for i in range(radius):
            for x in range(1, x_len-1):
                for y in range(1, y_len-1):
                    for z in range(1, z_len-1):
                        for m in range(len(materials)):
                            if plane == 'xy':
                                new_model[y, z, x, m] = np.min(current_model[y-1:y+2, z, x-1:x+2, m])

                            elif plane == 'xz':
                                new_model[y, z, x, m] = np.min(current_model[y, z-1:z+2, x-1:x+2, m])

                            elif plane == 'yz':
                                new_model[y, z, x, m] = np.min(current_model[y-1:y+2, z-1:z+2, x, m])

                            else:
                                new_model[y, z, x, m] = np.min(current_model[y-1:y+2, z-1:z+2, x-1:x+2, m])

            current_model = np.copy(new_model)

        return VoxelModel(current_model, self.x, self.y, self.z)

    # Cleanup ##########################################################################
    def normalize(self):
        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])

        new_model = np.zeros_like(self.model)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    m_sum = np.sum(self.model[y, z, x, :])
                    if m_sum > 0:
                        for m in range(len(materials)):
                            new_model[y, z, x, m] = self.model[y, z, x, m] / m_sum

        new_model = np.around(new_model, 3)

        return VoxelModel(new_model, self.x, self.y, self.z)

    def blur(self, region = 'all', threshold = 0.5):
        x_len = len(self.model[0, 0, :, 0]) + 2
        y_len = len(self.model[:, 0, 0, 0]) + 2
        z_len = len(self.model[0, :, 0, 0]) + 2

        current_model = np.zeros((y_len, z_len, x_len, len(materials)))
        current_model[1:-1, 1:-1, 1:-1, :] = self.model

        new_model = np.zeros((y_len, z_len, x_len, len(materials)))

        kernel = np.array([[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                           [[2.0, 4.0, 2.0], [4.0, 8.0, 4.0], [2.0, 4.0, 2.0]],
                           [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]) * (1.0 / 64.0)

        for x in range(1, x_len - 1):
            for y in range(1, y_len - 1):
                for z in range(1, z_len - 1):
                    if region == 'overlap':
                        if np.sum(current_model[y, z, x, :]) > 1:
                            for m in range(len(materials)):
                                new_model[y, z, x, m] = np.sum(np.multiply(current_model[y - 1:y + 2, z - 1:z + 2, x - 1:x + 2, m], kernel))
                        else:
                            new_model[y, z, x, :] = current_model[y, z, x, :]
                    else:
                        for m in range(len(materials)):
                            new_model[y, z, x, m] = np.sum(np.multiply(current_model[y - 1:y + 2, z - 1:z + 2, x - 1:x + 2, m], kernel))

                    if np.sum(new_model[y, z, x, :]) < threshold:
                        new_model[y, z, x, :] = np.zeros(len(materials))

        return VoxelModel(new_model, self.x, self.y, self.z).normalize()

    # Manufacturing Features ###########################################################
    def bounding_box(self):
        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])

        new_model = np.zeros_like(self.model)

        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1
        z_min = -1
        z_max = -1

        for x in range(x_len):
            if x_min == -1:
                if np.sum(self.model[:, :, x, :]) > 0:
                    x_min = x
            else:
                if np.sum(self.model[:, :, x, :]) == 0:
                    x_max = x
                    break

        for y in range(y_len):
            if y_min == -1:
                if np.sum(self.model[y, :, :, :]) > 0:
                    y_min = y
            else:
                if np.sum(self.model[y, :, :, :]) == 0:
                    y_max = y
                    break

        for z in range(z_len):
            if z_min == -1:
                if np.sum(self.model[:, z, :, :]) > 0:
                    z_min = z
            else:
                if np.sum(self.model[:, z, :, :]) == 0:
                    z_max = z
                    break

        x_max = x_len if x_max == -1 else x_max
        y_max = y_len if y_max == -1 else y_max
        z_max = z_len if z_max == -1 else z_max

        new_model[y_min:y_max, z_min:z_max, x_min:x_max, :].fill(1)

        return VoxelModel(new_model, self.x, self.y, self.z)

    def keepout(self, method):
        new_model = np.zeros_like(self.model)

        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])

        if method == 'laser':
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    if np.sum(self.model[y, :, x, :]) > 0:
                        new_model[y, :, x, :] = np.ones((z_len, len(materials)))

        elif method == 'mill':
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if np.sum(self.model[y, z:, x, :]) > 0:
                            new_model[y, z, x, :] = np.ones(len(materials))
                        elif np.sum(self.model[y, z:, x, :]) == 0:
                            break

        return VoxelModel(new_model, self.x, self.y, self.z)