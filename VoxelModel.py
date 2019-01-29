"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
from pyvox.parser import VoxParser
from materials import materials
from scipy import ndimage

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

    @classmethod
    def emptyLike(cls, voxel_model):
        new_model = np.zeros_like(voxel_model.model)
        return cls(new_model, voxel_model.x, voxel_model.y, voxel_model.z)

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
        new_model[:, layer - self.z, :, :] = self.model[:, layer - self.z, :, :]
        return VoxelModel(new_model, self.x, self.y, self.z)

    # Mask operations ##################################################################
    def clearMaterial(self):
        mask = (self.model != np.zeros(len(materials))).any(3)
        mask = np.repeat(mask[..., None], len(materials), axis=3)
        return VoxelModel(mask, self.x, self.y, self.z)

    def setMaterial(self, material):
        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])
        new_model = self.clearMaterial().model
        material_vector = np.zeros(len(materials))
        material_vector[material] = 1
        material_array = np.tile(material_vector[None, None, None, :], (y_len, z_len, x_len, 1))
        new_model = np.multiply(new_model, material_array)
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
        return self.addVolume(other)

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
        return self.subtractVolume(other)

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
        x_len = len(self.model[0, 0, :, 0]) + (radius * 2)
        y_len = len(self.model[:, 0, 0, 0]) + (radius * 2)
        z_len = len(self.model[0, :, 0, 0]) + (radius * 2)

        new_model = np.zeros((y_len, z_len, x_len, len(materials)))
        new_model[radius:-radius, radius:-radius, radius:-radius, :] = self.model

        diameter = (radius * 2) + 1

        if plane == 'xy':
            size = (diameter, 1, diameter)
        elif plane == 'xz':
            size = (1, diameter, diameter)
        elif plane == 'yz':
            size = (diameter, diameter, 1)
        else:
            size = (diameter, diameter, diameter)

        for m in range(len(materials)):
            new_model[:, :, :, m] = ndimage.grey_dilation(new_model[:, :, :, m], size)

        return VoxelModel(new_model, self.x - radius, self.y - radius, self.z - radius)

    def erode(self, radius = 1, plane = 'xyz'):
        new_model = np.copy(self.model)
        diameter = (radius * 2) + 1

        if plane == 'xy':
            size = (diameter, 1, diameter)
        elif plane == 'xz':
            size = (1, diameter, diameter)
        elif plane == 'yz':
            size = (diameter, diameter, 1)
        else:
            size = (diameter, diameter, diameter)

        for m in range(len(materials)):
            new_model[:, :, :, m] = ndimage.grey_erosion(new_model[:, :, :, m], size)

        return VoxelModel(new_model, self.x, self.y, self.z)

    # Cleanup ##########################################################################
    def normalize(self):
        material_sums = np.sum(self.model, 3)
        material_sums[material_sums == 0] = 1
        material_sums = np.repeat(material_sums[..., None], len(materials), axis=3)
        new_model = np.divide(self.model, material_sums)
        return VoxelModel(new_model, self.x, self.y, self.z)

    def blur(self, radius = 1, region = 'all'):
        if region == 'overlap':
            material_sums = np.sum(self.model, 3)
            mask = np.zeros_like(material_sums)
            mask[material_sums > 1] = 1
            mask = np.repeat(mask[..., None], len(materials), axis=3)
        else:
            mask = self.clearMaterial().model

        new_model = np.zeros_like(self.model)
        for m in range(len(materials)):
            new_model[:, :, : , m] = ndimage.gaussian_filter(self.model[:, :, : , m], sigma=radius)

        new_model = np.multiply(new_model, mask)
        new_model = VoxelModel(new_model, self.x, self.y, self.z).normalize()

        new_model = new_model + self

        return new_model

    # Manufacturing Features ###########################################################
    def boundingBox(self):
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

    def clearance(self, method):
        new_model = VoxelModel(np.zeros_like(self.model), self.x, self.y, self.z)

        Kl = self.keepout('laser')

        if method == 'laser':
            new_model = Kl.subtractVolume(self)
        elif method == 'mill':
            Km = self.keepout('mill')
            new_model = Kl.subtractVolume(Km)

        return new_model

    def web(self, method, layer, r1=1, r2=1):
        model_A = self.keepout(method)
        model_A = model_A.isolateLayer(layer)
        model_B = model_A.dilate(r1, 'xy')
        model_C = model_B.dilate(r2, 'xy')
        model_D = model_C.boundingBox()
        new_model = model_D.subtractVolume(model_B)
        return new_model

    def support(self, method, r1=1, r2=1, plane='xy'):
        model_A = self.keepout(method)
        model_B = model_A.subtractVolume(self)
        model_C = model_B.dilate(r1, plane)
        model_D = model_A.dilate(r2, plane)
        model_E = model_D.subtractVolume(model_A)
        new_model = model_E.subtractVolume(model_C)
        return new_model

    def mergeSupport(self, support_model, method, r1=1, r2=1, plane='xy'):
        model_A = self.support(method, r1, r2, plane)
        new_model = support_model.intersectVolume(model_A)
        return new_model