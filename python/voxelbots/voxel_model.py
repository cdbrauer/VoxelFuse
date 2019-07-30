"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
from pyvox.parser import VoxParser
from voxelbots.materials import materials
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
  model: array storing the material type present at each voxel
         voxel format: <a, n, m0, m1, ... mn>
         
  x_coord, y_coord, z_coord: position of model origin
"""
class VoxelModel:
    def __init__(self, model, x_coord = 0, y_coord = 0, z_coord = 0):
        self.model = model
        self.x = x_coord
        self.y = y_coord
        self.z = z_coord

    @classmethod
    def fromVoxFile(cls, filename, x_coord = 0, y_coord = 0, z_coord = 0):
        m1 = VoxParser(filename).parse()
        m2 = m1.to_dense()
        m2 = np.flip(m2, 1)

        x_len = len(m2[0, 0, :])
        y_len = len(m2[:, 0, 0])
        z_len = len(m2[0, :, 0])

        new_model = np.zeros((y_len, z_len, x_len, len(materials)+1))

        # Loop through input_model data
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    color_index = m2[y, z, x]
                    if (color_index > 0) and (color_index < len(materials)):
                        new_model[y, z, x, 0] = 1
                        new_model[y, z, x, color_index+1] = 1
                    elif color_index >= len(materials):
                        print('Unrecognized material index: ' + str(color_index) + '. Setting to null')
                        new_model[y, z, x, 0] = 1
                        new_model[y, z, x, 1] = 1

        return cls(new_model, x_coord, y_coord, z_coord)

    @classmethod
    def emptyLike(cls, voxel_model):
        new_model = np.zeros_like(voxel_model.model)
        return cls(new_model, voxel_model.x, voxel_model.y, voxel_model.z)

    @classmethod
    def copy(cls, voxel_model):
        new_model = np.copy(voxel_model.model)
        return cls(new_model, voxel_model.x, voxel_model.y, voxel_model.z)

    # Selection operations #############################################################
    # Get all voxels with a specified material
    def isolateMaterial(self, material): # material input is an index corresponding to the materials table
        material_vector = np.zeros(len(materials)+1)
        material_vector[0] = 1
        material_vector[material+1] = 1
        mask = (self.model == material_vector).all(3)
        mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
        new_model = np.multiply(self.model, mask)
        return VoxelModel(new_model, self.x, self.y, self.z)

    # Get all voxels in a specified layer
    def isolateLayer(self, layer):
        new_model = np.zeros_like(self.model)
        new_model[:, layer - self.z, :, :] = self.model[:, layer - self.z, :, :]
        return VoxelModel(new_model, self.x, self.y, self.z)

    # Mask operations ##################################################################
    # Return a mask of all voxels not occupied by the input model
    def getUnoccupied(self):
        mask = np.logical_not(self.model[:, :, :, 0])
        mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
        return VoxelModel(mask, self.x, self.y, self.z)

    # Return a mask of all voxels occupied by the input model
    def getOccupied(self):
        mask = self.model[:, :, :, 0]
        mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
        return VoxelModel(mask, self.x, self.y, self.z)

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
                if np.sum(self.model[:, :, x, 0]) > 0:
                    x_min = x
            else:
                if np.sum(self.model[:, :, x, 0]) == 0:
                    x_max = x
                    break

        for y in range(y_len):
            if y_min == -1:
                if np.sum(self.model[y, :, :, 0]) > 0:
                    y_min = y
            else:
                if np.sum(self.model[y, :, :, 0]) == 0:
                    y_max = y
                    break

        for z in range(z_len):
            if z_min == -1:
                if np.sum(self.model[:, z, :, 0]) > 0:
                    z_min = z
            else:
                if np.sum(self.model[:, z, :, 0]) == 0:
                    z_max = z
                    break

        x_max = x_len if x_max == -1 else x_max
        y_max = y_len if y_max == -1 else y_max
        z_max = z_len if z_max == -1 else z_max

        new_model[y_min:y_max, z_min:z_max, x_min:x_max, :].fill(1)

        return VoxelModel(new_model, self.x, self.y, self.z)

    # Set the material of a mask or model
    def setMaterial(self, material): # material input is an index corresponding to the materials table
        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])
        new_model = self.getOccupied().model # Converts input model to a mask, no effect if input is already a mask
        material_vector = np.zeros(len(materials)+1)
        material_vector[0] = 1
        material_vector[material+1] = 1
        material_array = np.tile(material_vector[None, None, None, :], (y_len, z_len, x_len, 1))
        new_model = np.multiply(new_model, material_array)
        return VoxelModel(new_model, self.x, self.y, self.z)

    # Boolean operations ###############################################################
    # Material from base model takes priority
    def difference(self, model_to_sub):
        a, b = alignDims(self, model_to_sub)
        # mask = np.logical_and(a.model[:, :, :, 0], np.logical_not(b.model[:, :, :, 0])) # Full comparison as defined in paper
        mask = np.logical_not(b.model[:, :, :, 0]) # Shortened version, gives same result due to multiplication step
        mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
        new_model = np.multiply(a.model, mask)
        return VoxelModel(new_model, a.x, a.y, a.z)

    def intersection(self, model_2, material_priority = 'l'):
        a, b = alignDims(self, model_2)
        mask = np.logical_and(a.model[:, :, :, 0], b.model[:, :, :, 0])
        mask = np.repeat(mask[..., None], len(materials)+1, axis=3)

        if material_priority == 'r':
            new_model = np.multiply(b.model, mask) # material from right model takes priority
        else:
            new_model = np.multiply(a.model, mask) # material from left model takes priority

        return VoxelModel(new_model, a.x, a.y, a.z)

    def union(self, model_to_add, material_priority = 'l'):
        a, b = alignDims(self, model_to_add)

        # Paper uses a symmetric difference operation combined with the left/right intersection
        # A condensed version of this operation is used here for code simplicity
        if material_priority == 'r':
            mask = np.logical_not(b.model[:, :, :, 0])
            mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
            new_model = np.multiply(a.model, mask)
            new_model = new_model + b.model # material from right model takes priority
        else:
            mask = np.logical_not(a.model[:, :, :, 0])
            mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
            new_model = np.multiply(b.model, mask)
            new_model = new_model + a.model # material from left model takes priority

        return VoxelModel(new_model, a.x, a.y, a.z)

    # Material is computed
    def add(self, model_to_add):
        a, b = alignDims(self, model_to_add)
        mask = np.logical_or(a.model[:, :, :, 0], b.model[:, :, :, 0])
        new_model = a.model + b.model
        new_model[:, :, :, 0] = mask
        return VoxelModel(new_model, a.x, a.y, a.z)

    def __add__(self, other):
        return self.add(other)

    def subtract(self, model_to_sub):
        a, b = alignDims(self, model_to_sub)
        mask = np.logical_or(a.model[:, :, :, 0], b.model[:, :, :, 0]) # Note that negative material values are retained
        new_model = a.model - b.model
        new_model[:, :, :, 0] = mask
        return VoxelModel(new_model, a.x, a.y, a.z)

    def __sub__(self, other):
        return self.subtract(other)

    # Dilate and Erode #################################################################
    def dilate(self, radius = 1, plane = 'xyz', connectivity = 3):
        x_len = len(self.model[0, 0, :, 0]) + (radius * 2)
        y_len = len(self.model[:, 0, 0, 0]) + (radius * 2)
        z_len = len(self.model[0, :, 0, 0]) + (radius * 2)

        new_model = np.zeros((y_len, z_len, x_len, len(materials)+1))
        new_model[radius:-radius, radius:-radius, radius:-radius, :] = self.model

        struct = ndimage.generate_binary_structure(3, connectivity)

        if plane == 'xy':
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
        elif plane == 'xz':
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
        elif plane == 'yz':
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)

        for i in range(radius):
            new_model[:, :, :, 0] = ndimage.binary_dilation(new_model[:, :, :, 0], structure=struct)
            for m in range(len(materials)):
                new_model[:, :, :, m+1] = ndimage.grey_dilation(new_model[:, :, :, m+1], footprint=struct)

        return VoxelModel(new_model, self.x - radius, self.y - radius, self.z - radius)

    def erode(self, radius = 1, plane = 'xyz', connectivity = 3):
        x_len = len(self.model[0, 0, :, 0]) + (radius * 2)
        y_len = len(self.model[:, 0, 0, 0]) + (radius * 2)
        z_len = len(self.model[0, :, 0, 0]) + (radius * 2)

        new_model = np.zeros((y_len, z_len, x_len, len(materials) + 1))
        new_model[radius:-radius, radius:-radius, radius:-radius, :] = self.model

        struct = ndimage.generate_binary_structure(3, connectivity)

        if plane == 'xy':
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
        elif plane == 'xz':
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
        elif plane == 'yz':
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)

        for i in range(radius):
            new_model[:, :, :, 0] = ndimage.binary_erosion(new_model[:, :, :, 0], structure=struct)
            for m in range(len(materials)):
                new_model[:, :, :, m+1] = ndimage.grey_erosion(new_model[:, :, :, m+1], footprint=struct)

        return VoxelModel(new_model, self.x - radius, self.y - radius, self.z - radius)

    # Material Interface Modification ##################################################
    def blur(self, radius=1):
        new_model = np.zeros_like(self.model)

        new_model[:, :, :, 0] = self.model[:, :, :, 0]
        for m in range(len(materials)):
            new_model[:, :, :, m+1] = ndimage.gaussian_filter(self.model[:, :, :, m+1], sigma=radius)

        new_model = np.multiply(new_model, self.getOccupied().model)

        return VoxelModel(new_model, self.x, self.y, self.z)

    def blurRegion(self, radius, region):
        new_model = self.blur(radius)
        new_model = new_model.intersection(region)
        new_model = new_model.union(self)
        return new_model

    # Add dithering options

    # Cleanup ##########################################################################
    def removeNegatives(self): # Remove negative material values (which have no physical meaning)
        new_model = np.copy(self.model)
        new_model[new_model < 0] = 0
        material_sums = np.sum(new_model[:,:,:,1:], 3) # This and following update the a values
        material_sums[material_sums > 0] = 1
        new_model[:, :, :, 0] = material_sums
        return VoxelModel(new_model, self.x, self.y, self.z)

    def scaleValues(self): # Scale material values while maintaining the ratio between all materials
        new_model = np.copy(self.removeNegatives().model)
        material_sums = np.sum(new_model[:,:,:,1:], 3)
        material_sums[material_sums == 0] = 1
        material_sums = np.repeat(material_sums[..., None], len(materials), axis=3)
        new_model[:,:,:,1:] = np.divide(new_model[:,:,:,1:], material_sums)
        return VoxelModel(new_model, self.x, self.y, self.z)

    def scaleNull(self): # Scale null values to make all voxels contain 100% material
        new_model = np.copy(self.removeNegatives().model)
        material_sums = np.sum(new_model[:,:,:,2:], 3)
        material_sums = np.ones(np.shape(material_sums)) - material_sums
        material_sums[material_sums < 0] = 0
        new_model[:,:,:,1] = material_sums
        return VoxelModel(new_model, self.x, self.y, self.z)

    def rotate(self, angle, axis):
        if axis == 'x':
            plane = (0, 1)
        elif axis == 'y':
            plane = (1, 2)
        else: # axis == 'z'
            plane = (0, 2)

        new_model = ndimage.rotate(self.model, angle, plane, order=0)

        return VoxelModel(new_model, self.x, self.y, self.z)

    # Manufacturing Features ###########################################################
    def projection(self, direction):
        new_model = np.zeros_like(self.model)

        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])

        if direction == 'both':
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    if np.sum(self.model[y, :, x, 0]) > 0:
                        new_model[y, :, x, :].fill(1)

        elif direction == 'down':
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if np.sum(self.model[y, z:, x, 0]) > 0:
                            new_model[y, z, x, :].fill(1)
                        elif np.sum(self.model[y, z:, x, 0]) == 0:
                            break

        elif direction == 'up':
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if np.sum(self.model[y, :z, x, 0]) > 0:
                            new_model[y, z, x, :].fill(1)

        return VoxelModel(new_model, self.x, self.y, self.z)

    def keepout(self, method):
        if method == 'laser':
            new_model = self.projection('both')
        elif method == 'mill':
            new_model = self.projection('down')
        elif method == 'ins':
            new_model = self.projection('up')
        else:
            new_model = self

        return new_model

    def clearance(self, method):
        if method == 'laser':
            new_model = self.projection('both').difference(self)
        elif method == 'mill':
            new_model = self.projection('both').difference(self.projection('down'))
        elif (method == 'ins') or (method == '3dp'):
            new_model = self.projection('up')
        else:
            new_model = self

        return new_model

    def support(self, method, r1=1, r2=1, plane='xy'):
        model_A = self.keepout(method)
        model_B = model_A.difference(self)
        model_C = model_B.dilate(r1, plane) # Regions where support is ineffective due to proximity to inaccessible workspace regions
        model_D = model_A.dilate(r2, plane)
        model_E = model_D.difference(model_A) # Accesible region around part of width r2
        new_model = model_E.difference(model_C) # Valid support regions
        return new_model

    def userSupport(self, support_model, method, r1=1, r2=1, plane='xy'):
        model_A = self.support(method, r1, r2, plane)
        new_model = support_model.intersection(model_A)
        return new_model

    def web(self, method, r1=1, r2=1, layer=-1):
        model_A = self.keepout(method)
        if layer != -1:
            model_A = model_A.isolateLayer(layer)
        model_B = model_A.dilate(r1, 'xy') # Clearance around part, width = r1
        model_C = model_B.dilate(r2, 'xy')  # Support width = r2
        model_D = model_C.boundingBox() # Make support rectangular
        new_model = model_D.difference(model_B)
        return new_model