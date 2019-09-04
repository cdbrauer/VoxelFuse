"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""
import os
import subprocess
import meshio
import numpy as np
from pyvox.parser import VoxParser
from voxelfuse.materials import materials
from scipy import ndimage
from numba import njit, prange

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
        self.numComponents = 0
        s = np.shape(model)
        self.components = np.zeros(s[:3])

    @classmethod
    def fromVoxFile(cls, filename, x_coord = 0, y_coord = 0, z_coord = 0):
        m1 = VoxParser(filename).parse() # Import data and align axes
        m2 = m1.to_dense()
        m2 = np.flip(m2, 1)
        new_model = formatVoxData(m2, len(materials)) # Reformat data
        return cls(new_model, x_coord, y_coord, z_coord)

    @classmethod
    def fromMeshFile(cls, filename, x_coord=0, y_coord=0, z_coord=0):
        res = 0
        data = makeMesh(filename, True)

        points = data.points
        ii_tri = data.cells['triangle']
        ii_tet = data.cells['tetra']
        tris = points[ii_tri]
        tets = points[ii_tet]
        T = np.concatenate((tets, tets[:, :, 0:1] * 0 + 1), 2)
        T_inv = np.zeros(T.shape)

        for ii, t in enumerate(T):
            T_inv[ii] = np.linalg.inv(t).T

        points_min = points.min(0)
        points_max = points.max(0)
        points_min_r = np.round(points_min, res)
        points_max_r = np.round(points_max, res)

        xx = np.r_[points_min_r[0]:points_max_r[0]+1:10 ** (-res)]
        yy = np.r_[points_min_r[1]:points_max_r[1]+1:10 ** (-res)]
        zz = np.r_[points_min_r[2]:points_max_r[2]+1:10 ** (-res)]

        xx_mid = (xx[1:] + xx[:-1]) / 2
        yy_mid = (yy[1:] + yy[:-1]) / 2
        zz_mid = (zz[1:] + zz[:-1]) / 2

        xyz_mid = np.array(np.meshgrid(xx_mid, yy_mid, zz_mid, indexing='ij'))
        xyz_mid = xyz_mid.transpose(1, 2, 3, 0)
        xyz_mid = xyz_mid.reshape(-1, 3)
        xyz_mid = np.concatenate((xyz_mid, xyz_mid[:, 0:1] * 0 + 1), 1)

        ijk_mid = np.array(
            np.meshgrid(np.r_[:len(xx_mid)], np.r_[:len(yy_mid)], np.r_[:len(zz_mid)], indexing='ij'))
        ijk_mid = ijk_mid.transpose(1, 2, 3, 0)
        ijk_mid2 = ijk_mid.reshape(-1, 3)

        u2 = dot3d(np.asarray(T_inv, order='c'), np.asarray(xyz_mid, order='c'))

        f1 = ((u2[:, :, :] >= 0).sum(1) == 4)
        f2 = ((u2[:, :, :] <= 1).sum(1) == 4)
        f3 = f1 & f2
        ii, jj = f3.nonzero()

        lmn = ijk_mid2[np.unique(jj)]

        voxels = np.zeros(ijk_mid.shape[:3], dtype=np.bool)
        voxels[lmn[:, 0], lmn[:, 1], lmn[:, 2]] = True

        new_model = np.rot90(voxels, axes=(0, 2))
        new_model = np.rot90(new_model, 3, axes=(0, 1))
        new_model = formatVoxData(new_model, len(materials))
        
        return cls(new_model, x_coord, y_coord, z_coord)

    @classmethod
    def emptyLike(cls, voxel_model):
        new_model = np.zeros_like(voxel_model.model)
        return cls(new_model, voxel_model.x, voxel_model.y, voxel_model.z)

    @classmethod
    def copy(cls, voxel_model):
        new_model = np.copy(voxel_model.model)
        # Component labels are not copied
        return cls(new_model, voxel_model.x, voxel_model.y, voxel_model.z)

    """
    Property update operations
    
    - These operations work directly on the model
    - Nothing is returned 
    """
    # Remove excess empty workspace from a model
    def fitWorkspace(self):
        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])

        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1
        z_min = -1
        z_max = -1

        for x in range(x_len):
            if np.sum(self.model[:, :, x, 0]) > 0:
                x_min = x
                break

        for x in range(x_len-1,-1,-1):
            if np.sum(self.model[:, :, x, 0]) > 0:
                x_max = x+1
                break

        for y in range(y_len):
            if np.sum(self.model[y, :, :, 0]) > 0:
                y_min = y
                break

        for y in range(y_len-1,-1,-1):
            if np.sum(self.model[y, :, :, 0]) > 0:
                y_max = y+1
                break

        for z in range(z_len):
            if np.sum(self.model[:, z, :, 0]) > 0:
                z_min = z
                break

        for z in range(z_len-1,-1,-1):
            if np.sum(self.model[:, z, :, 0]) > 0:
                z_max = z+1
                break

        x_min = 0 if x_min == -1 else x_min
        y_min = 0 if y_min == -1 else y_min
        z_min = 0 if z_min == -1 else z_min

        x_max = x_len if x_max == -1 else x_max
        y_max = y_len if y_max == -1 else y_max
        z_max = z_len if z_max == -1 else z_max

        new_model = np.copy(self.model[y_min:y_max, z_min:z_max, x_min:x_max, :])
        new_components = np.copy(self.components[y_min:y_max, z_min:z_max, x_min:x_max])

        self.model = new_model
        self.x = self.x + x_min
        self.y = self.y + y_min
        self.z = self.z + z_min
        self.components = new_components

    # Update component labels for a model.  This uses a disconnected components algorithm and assumes that adjacent voxels with different materials are connected.
    def getComponents(self, connectivity=1):
        mask = self.model[:, :, :, 0]
        struct = ndimage.generate_binary_structure(3, connectivity)
        self.components, self.numComponents = ndimage.label(mask, structure=struct)

    """    
    Selection operations
    
    - Return a model
    """
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

    # Isolate component by component label
    # Component labels must first be updated with getComponents
    # Unrecognized component labels will return an empty object
    def isolateComponent(self, component):
        mask = np.copy(self.components)
        mask[mask != component] = 0
        mask[mask == component] = 1
        mask = np.repeat(mask[..., None], len(materials) + 1, axis=3)
        new_model = np.multiply(self.model, mask)
        return VoxelModel(new_model, self.x, self.y, self.z)

    """
    Mask operations
    
    - Return a mask (except for setMaterial)
    """
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

    # Return a mask of the bounding box of the input model
    def getBoundingBox(self):
        new_model = VoxelModel.copy(self)
        new_model.fitWorkspace()
        new_model.model.fill(1)
        return new_model

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

    # Set the material of a mask or model
    def setMaterialVector(self, material_vector):  # material input is the desired material vector
        x_len = len(self.model[0, 0, :, 0])
        y_len = len(self.model[:, 0, 0, 0])
        z_len = len(self.model[0, :, 0, 0])
        new_model = self.getOccupied().model  # Converts input model to a mask, no effect if input is already a mask
        material_array = np.tile(material_vector[None, None, None, :], (y_len, z_len, x_len, 1))
        new_model = np.multiply(new_model, material_array)
        return VoxelModel(new_model, self.x, self.y, self.z)

    """
    Boolean operations
    
    - Return a model    
    - Material from base model takes priority by default
    """
    def difference(self, model_to_sub):
        a, b, x_new, y_new, z_new = alignDims(self, model_to_sub)
        # mask = np.logical_and(a[:, :, :, 0], np.logical_not(b[:, :, :, 0])) # Full comparison as defined in paper
        mask = np.logical_not(b[:, :, :, 0]) # Shortened version, gives same result due to multiplication step
        mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
        new_model = np.multiply(a, mask)
        return VoxelModel(new_model, x_new, y_new, z_new)

    def intersection(self, model_2, material_priority = 'l'):
        a, b, x_new, y_new, z_new = alignDims(self, model_2)
        mask = np.logical_and(a[:, :, :, 0], b[:, :, :, 0])
        mask = np.repeat(mask[..., None], len(materials)+1, axis=3)

        if material_priority == 'r':
            new_model = np.multiply(b, mask) # material from right model takes priority
        else:
            new_model = np.multiply(a, mask) # material from left model takes priority

        return VoxelModel(new_model, x_new, y_new, z_new)

    def union(self, model_to_add, material_priority = 'l'):
        a, b, x_new, y_new, z_new = alignDims(self, model_to_add)

        # Paper uses a symmetric difference operation combined with the left/right intersection
        # A condensed version of this operation is used here for code simplicity
        if material_priority == 'r':
            mask = np.logical_not(b[:, :, :, 0])
            mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
            new_model = np.multiply(a, mask)
            new_model = new_model + b # material from right model takes priority
        else:
            mask = np.logical_not(a[:, :, :, 0])
            mask = np.repeat(mask[..., None], len(materials)+1, axis=3)
            new_model = np.multiply(b, mask)
            new_model = new_model + a # material from left model takes priority

        return VoxelModel(new_model, x_new, y_new, z_new)

    # Material is computed
    def add(self, model_to_add):
        a, b, x_new, y_new, z_new = alignDims(self, model_to_add)
        mask = np.logical_or(a[:, :, :, 0], b[:, :, :, 0])
        new_model = a + b
        new_model[:, :, :, 0] = mask
        return VoxelModel(new_model, x_new, y_new, z_new)

    def __add__(self, other):
        return self.add(other)

    # Material is computed
    def subtract(self, model_to_sub):
        a, b, x_new, y_new, z_new = alignDims(self, model_to_sub)
        mask = np.logical_or(a[:, :, :, 0], b[:, :, :, 0]) # Note that negative material values are retained
        new_model = a - b
        new_model[:, :, :, 0] = mask
        return VoxelModel(new_model, x_new, y_new, z_new)

    def __sub__(self, other):
        return self.subtract(other)

    """
    Dilate and Erode
    
    - Return a model
    """
    def dilate(self, radius = 1, plane = 'xyz', connectivity = 3):
        if radius == 0:
            return VoxelModel.copy(self)

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
        elif plane == 'x':
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
        elif plane == 'y':
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)
        elif plane == 'z':
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)

        #print('Dilate:')
        for i in range(radius):
            #print(str(i) + '/' + str(radius))
            new_model[:, :, :, 0] = ndimage.binary_dilation(new_model[:, :, :, 0], structure=struct)
            for m in range(len(materials)):
                new_model[:, :, :, m+1] = ndimage.grey_dilation(new_model[:, :, :, m+1], footprint=struct)

        return VoxelModel(new_model, self.x - radius, self.y - radius, self.z - radius)

    def dilateBounded(self, radius = 1, plane = 'xyz', connectivity = 3): # Dilates a model without increasing the size of its bounding box
        if radius == 0:
            return VoxelModel.copy(self)

        self.fitWorkspace()
        new_model = np.copy(self.model)

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
        elif plane == 'x':
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
        elif plane == 'y':
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)
        elif plane == 'z':
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)

        #print('Dilate Bounded:')
        for i in range(radius):
            #print(str(i) + '/' + str(radius))
            new_model[:, :, :, 0] = ndimage.binary_dilation(new_model[:, :, :, 0], structure=struct)
            for m in range(len(materials)):
                new_model[:, :, :, m+1] = ndimage.grey_dilation(new_model[:, :, :, m+1], footprint=struct)

        return VoxelModel(new_model, self.x, self.y, self.z)

    def erode(self, radius = 1, plane = 'xyz', connectivity = 3):
        if radius == 0:
            return VoxelModel.copy(self)

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

        #print('Erode:')
        for i in range(radius):
            #print(str(i) + '/' + str(radius))
            new_model[:, :, :, 0] = ndimage.binary_erosion(new_model[:, :, :, 0], structure=struct)
            for m in range(len(materials)):
                new_model[:, :, :, m+1] = ndimage.grey_erosion(new_model[:, :, :, m+1], footprint=struct)

        return VoxelModel(new_model, self.x - radius, self.y - radius, self.z - radius)

    """
    Material Interface Modification
    
    - Return a model
    """
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
    def dither(self, radius=1):
        new_model = self.blur(radius)
        new_model = new_model.scaleValues()
        new_model.model = dither(new_model.model)
        return new_model

    """
    Cleanup
    
    - Return a model
    """
    def removeNegatives(self): # Remove negative material values (which have no physical meaning)
        new_model = np.copy(self.model)
        new_model[new_model < 1e-10] = 0
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

    def rotate(self, angle, axis = 'z'): # TODO: Check that coords are handled correctly
        if axis == 'x':
            plane = (0, 1)
        elif axis == 'y':
            plane = (1, 2)
        else: # axis == 'z'
            plane = (0, 2)

        new_model = ndimage.rotate(self.model, angle, plane, order=0)

        return VoxelModel(new_model, self.x, self.y, self.z)

    def rotate90(self, times = 1, axis = 'z'): # TODO: Check that coords are handled correctly
        if axis == 'x':
            plane = (0, 1)
        elif axis == 'y':
            plane = (1, 2)
        else: # axis == 'z'
            plane = (0, 2)

        new_model = np.rot90(self.model, times, axes=plane)

        return VoxelModel(new_model, self.x, self.y, self.z)

    """
    Manufacturing Features
    
    - Return a mask
    """
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
        model_D = model_C.getBoundingBox() # Make support rectangular
        new_model = model_D.difference(model_B)
        return new_model

# Helper methods ##############################################################

"""
Function to import mesh data from file
"""
def makeMesh(filename, delete_files=True):
    template = '''
    Merge "{0}";
    Surface Loop(1) = {{1}};
    //+
    Volume(1) = {{1}};
    '''

    geo_string = template.format(filename)
    with open('output.geo', 'w') as f:
        f.writelines(geo_string)

    command_string = 'gmsh output.geo -3 -format msh'
    p = subprocess.Popen(command_string, shell=True)
    p.wait()
    mesh_file = 'output.msh'
    data = meshio.read(mesh_file)
    if delete_files:
        os.remove('output.msh')
        os.remove('output.geo')
    return data

"""
Function to make object dimensions compatible for solid body operations. Takes location coordinates into account.
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

    return modelANew, modelBNew, xNew, yNew, zNew

"""
Function to convert vox file data into VoxelModel format. Separated to allow for acceleration using Numba.
[color index] -> [a, m0, m1, ... mn]
"""
@njit()
def formatVoxData(input_matrix, material_count):
    x_len = len(input_matrix[0, 0, :])
    y_len = len(input_matrix[:, 0, 0])
    z_len = len(input_matrix[0, :, 0])

    new_model = np.zeros((y_len, z_len, x_len, material_count+1))

    # Loop through input_model data
    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                color_index = input_matrix[y, z, x]
                if (color_index > 0) and (color_index < material_count):
                    new_model[y, z, x, 0] = 1
                    new_model[y, z, x, color_index + 1] = 1
                elif color_index >= material_count:
                    # print('Unrecognized material index: ' + str(color_index) + '. Setting to null') # Not compatible with @njit
                    new_model[y, z, x, 0] = 1
                    new_model[y, z, x, 1] = 1

    return new_model

@njit(parallel=True)
def dot3d(a, b):
    x_len = len(a[:, 0, 0])
    y_len = len(a[0, :, 0])
    z_len = len(b[:, 0])

    result = np.zeros((x_len, y_len, z_len), dtype=np.float32)

    for x in prange(x_len):
        for y in prange(y_len):
            for z in prange(z_len):
                result[x, y, z] = a[x, y, :].dot(b[z, :])

    return result

@njit()
def dither(model):
    x_len = len(model[0, 0, :])
    y_len = len(model[:, 0, 0])
    z_len = len(model[0, :, 0])

    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                voxel = model[y, z, x]
                if voxel[0] > 0:
                    max_i = voxel[1:].argmax()+1
                    for i in range(1, len(voxel)):
                        old = model[y, z, x, i]

                        if i == max_i:
                            model[y, z, x, i] = 1
                        else:
                            model[y, z, x, i] = 0

                        error = old - model[y, z, x, i]
                        if y+1 < y_len:
                            model[y+1, z, x, i] += error * (3/10) * model[y+1, z, x, 0]
                        if y+1 < y_len and x+1 < x_len:
                            model[y+1, z, x+1, i] += error * (1/5) * model[y+1, z, x+1, 0]
                        if y+1 < y_len and x+1 < x_len and z+1 < z_len:
                            model[y+1, z+1, x+1, i] += error * (1/5) * model[y+1, z+1, x+1, 0]
                        if x+1 < x_len:
                            model[y, z, x+1, i] += error * (3/10) * model[y, z, x+1, 0]

    return model
