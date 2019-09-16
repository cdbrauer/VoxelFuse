"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""
import os
import subprocess
import meshio
import numpy as np
from enum import Enum
from pyvox.parser import VoxParser
from voxelfuse.materials import material_properties
from scipy import ndimage
from numba import njit, prange

FLOATING_ERROR = 0.0000000001

class Axes(Enum):
    X = (1,0,0)
    Y = (0,1,0)
    Z = (0,0,1)
    XY = (1,1,0)
    XZ = (1,0,1)
    YZ = (0,1,1)
    XYZ = (1,1,1)

class Dir(Enum):
    UP = 1
    DOWN = 2
    BOTH = 3

class Process(Enum):
    LASER = 1
    MILL = 2
    PRINT = 3
    CAST = 4
    INSERT = 5

"""
VoxelModel Class

Initialized from a model array or file and position coordinates

Properties:
  model: array storing the material type present at each voxel
         voxel format: <a, n, m0, m1, ... mn>
         
  x_coord, y_coord, z_coord: position of model origin
"""
class VoxelModel:
    def __init__(self, voxels, materials, x_coord = 0, y_coord = 0, z_coord = 0):
        self.x = x_coord
        self.y = y_coord
        self.z = z_coord
        self.materials = materials
        self.voxels = voxels
        self.numComponents = 0
        self.components = np.zeros_like(voxels)

    @classmethod
    def fromVoxFile(cls, filename, x_coord = 0, y_coord = 0, z_coord = 0):
        # Import data and align axes
        v1 = VoxParser(filename).parse()
        v2 = np.array(v1.to_dense(), dtype=np.int32)
        v2 = np.flip(v2, 1)
        v2 = np.rot90(v2, 1, (2, 0))
        v2 = np.rot90(v2, 1, (1, 2))

        # Generate materials table assuming indices match materials in material_properties
        i = 0
        materials = np.zeros((1, len(material_properties) + 1), dtype=np.float)
        for m in np.unique(v2):
            if m != 0:
                i = i+1
                material_vector = np.zeros(len(material_properties) + 1, dtype=np.float)
                material_vector[0] = 1
                material_vector[m+1] = 1
                materials = np.vstack((materials, material_vector))
                v2[v2 == m] = i

        return cls(v2, materials, x_coord, y_coord, z_coord)

    @classmethod
    def fromMeshFile(cls, filename, x_coord=0, y_coord=0, z_coord=0):
        res = 0
        data = makeMesh(filename, True)

        points = data.points

        # Get lists of indices of point
        ii_tri = data.cells['triangle']
        ii_tet = data.cells['tetra']

        # Convert lists of indices to lists of points
        tris = points[ii_tri]
        tets = points[ii_tet]

        # Create barycentric coordinate system
        T = np.concatenate((tets, tets[:, :, 0:1] * 0 + 1), 2)
        T_inv = np.zeros(T.shape)

        for ii, t in enumerate(T):
            T_inv[ii] = np.linalg.inv(t).T

        # Find bounding box
        points_min = points.min(0)
        points_max = points.max(0)
        points_min_r = np.round(points_min, res)
        points_max_r = np.round(points_max, res)

        # Create 3D grid
        xx = np.r_[points_min_r[0]:points_max_r[0]+1:10 ** (-res)]
        yy = np.r_[points_min_r[1]:points_max_r[1]+1:10 ** (-res)]
        zz = np.r_[points_min_r[2]:points_max_r[2]+1:10 ** (-res)]

        # Find center of every grid point
        xx_mid = (xx[1:] + xx[:-1]) / 2
        yy_mid = (yy[1:] + yy[:-1]) / 2
        zz_mid = (zz[1:] + zz[:-1]) / 2

        # Create grid of voxel centers
        xyz_mid = np.array(np.meshgrid(xx_mid, yy_mid, zz_mid, indexing='ij'))
        xyz_mid = xyz_mid.transpose(1, 2, 3, 0)
        # Convert to list of points
        xyz_mid = xyz_mid.reshape(-1, 3)
        # Add 1 to allow conversion to barycentric coordinates
        xyz_mid = np.concatenate((xyz_mid, xyz_mid[:, 0:1] * 0 + 1), 1)

        # Create list of indices of each voxel
        ijk_mid = np.array(
            np.meshgrid(np.r_[:len(xx_mid)], np.r_[:len(yy_mid)], np.r_[:len(zz_mid)], indexing='ij'))
        ijk_mid = ijk_mid.transpose(1, 2, 3, 0)
        ijk_mid2 = ijk_mid.reshape(-1, 3)

        f3 = findFilledVoxels(np.asarray(T_inv, order='c'), np.asarray(xyz_mid, order='c'))
        ii, jj = f3.nonzero()

        lmn = ijk_mid2[np.unique(jj)]

        voxels = np.zeros(ijk_mid.shape[:3], dtype=np.bool)
        voxels[lmn[:, 0], lmn[:, 1], lmn[:, 2]] = True

        # Assume material 1
        materials = np.zeros((1, len(material_properties) + 1), dtype=np.float)
        material_vector = np.zeros(len(material_properties) + 1, dtype=np.float)
        material_vector[0] = 1
        material_vector[2] = 1
        materials = np.vstack((materials, material_vector))
        
        return cls(voxels, materials, x_coord, y_coord, z_coord)

    @classmethod
    def emptyLike(cls, voxel_model):
        new_model = cls(np.zeros_like(voxel_model.voxels, dtype=np.int32), voxel_model.materials, voxel_model.x, voxel_model.y, voxel_model.z)
        return new_model

    @classmethod
    def copy(cls, voxel_model):
        new_model = cls(np.copy(voxel_model.voxels), voxel_model.materials, voxel_model.x, voxel_model.y, voxel_model.z)
        new_model.numComponents = voxel_model.numComponents
        new_model.components = voxel_model.components
        return new_model

    """
    Property update operations
    
    - These operations work directly on the model
    - Nothing is returned 
    """
    # Remove excess empty workspace from a model
    def fitWorkspace(self):
        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1
        z_min = -1
        z_max = -1

        for x in range(x_len):
            if np.sum(self.voxels[x, :, :]) > 0:
                x_min = x
                break

        for x in range(x_len-1,-1,-1):
            if np.sum(self.voxels[x, :, :]) > 0:
                x_max = x+1
                break

        for y in range(y_len):
            if np.sum(self.voxels[:, y, :]) > 0:
                y_min = y
                break

        for y in range(y_len-1,-1,-1):
            if np.sum(self.voxels[:, y, :]) > 0:
                y_max = y+1
                break

        for z in range(z_len):
            if np.sum(self.voxels[:, :, z]) > 0:
                z_min = z
                break

        for z in range(z_len-1,-1,-1):
            if np.sum(self.voxels[:, :, z]) > 0:
                z_max = z+1
                break

        x_min = 0 if x_min == -1 else x_min
        y_min = 0 if y_min == -1 else y_min
        z_min = 0 if z_min == -1 else z_min

        x_max = x_len if x_max == -1 else x_max
        y_max = y_len if y_max == -1 else y_max
        z_max = z_len if z_max == -1 else z_max

        new_voxels = np.copy(self.voxels[x_min:x_max, y_min:y_max, z_min:z_max])
        new_components = np.copy(self.components[x_min:x_max, y_min:y_max, z_min:z_max])

        self.voxels = new_voxels
        self.x = self.x + x_min
        self.y = self.y + y_min
        self.z = self.z + z_min
        self.components = new_components

    # Remove duplicate rows from a model's material array
    def removeDuplicateMaterials(self):
        new_materials = np.unique(self.materials, axis=0)

        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        new_voxels = np.zeros_like(self.voxels, dtype=int)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    i = self.voxels[x, y, z]
                    m = self.materials[i]
                    ni = np.where(np.equal(new_materials, m).all(1))[0][0]
                    new_voxels[x, y, z] = ni

        self.voxels = new_voxels
        self.materials = new_materials

    # Update component labels for a model.  This uses a disconnected components algorithm and assumes that adjacent voxels with different materials are connected.
    def getComponents(self, connectivity=1):
        mask = np.array(self.voxels[:, :, :] > 0, dtype=np.bool)
        struct = ndimage.generate_binary_structure(3, connectivity)
        self.components, self.numComponents = ndimage.label(mask, structure=struct)

    """    
    Selection operations
    
    - Return a model
    """
    # Get all voxels with a specified material
    # TODO: Should this reference the material properties table?
    # TODO: isolateMaterialVector
    def isolateMaterial(self, material): # material input is an index corresponding to the materials array for the model
        mask = np.array(self.voxels == material, dtype=np.bool)
        materials = np.zeros((2, len(material_properties)+1), dtype=np.float)
        materials[1] = self.materials[material]
        return VoxelModel(mask.astype(int), materials, self.x, self.y, self.z)

    # Get all voxels in a specified layer
    def isolateLayer(self, layer):
        new_voxels = np.zeros_like(self.voxels, dtype=np.int32)
        new_voxels[:, :, layer - self.z] = self.voxels[:, :, layer - self.z]
        return VoxelModel(new_voxels, self.materials, self.x, self.y, self.z)

    # Isolate component by component label
    # Component labels must first be updated with getComponents
    # Unrecognized component labels will return an empty object
    def isolateComponent(self, component):
        mask = np.array(self.components == component, dtype=np.bool)
        new_voxels = np.multiply(self.voxels, mask)
        return VoxelModel(new_voxels, self.materials, self.x, self.y, self.z)

    """
    Mask operations
    
    - Material defaults to the first material in the input model
    """
    # Return all voxels not occupied by the input model
    def getUnoccupied(self):
        mask = np.array(self.voxels == 0, dtype=np.bool)
        return VoxelModel(mask, self.materials[0:2, :], self.x, self.y, self.z)

    def __invert__(self):
        return self.getUnoccupied()

    # Return all voxels occupied by the input model
    def getOccupied(self):
        mask = np.array(self.voxels != 0, dtype=np.bool)
        return VoxelModel(mask, self.materials[0:2, :], self.x, self.y, self.z)

    # Return the bounding box of the input model
    def getBoundingBox(self):
        new_model = VoxelModel.copy(self)
        new_model.fitWorkspace()
        new_model.voxels.fill(1)
        new_model = new_model.getOccupied()
        new_model.materials = self.materials[0:2, :]
        return new_model

    # Set the material of a model
    def setMaterial(self, material): # material input is an index corresponding to the material properties table
        new_voxels = self.getOccupied().voxels # Converts input model to a mask, no effect if input is already a mask
        material_vector = np.zeros(len(material_properties)+1, dtype=np.float)
        material_vector[0] = 1
        material_vector[material+1] = 1
        a = np.zeros(len(material_properties)+1, dtype=np.float)
        b = material_vector
        m = np.vstack((a, b))
        return VoxelModel(new_voxels, m, self.x, self.y, self.z)

    # Set the material of a model
    def setMaterialVector(self, material_vector):  # material input is the desired material vector
        new_voxels = self.getOccupied().voxels  # Converts input model to a mask, no effect if input is already a mask
        a = np.zeros(len(material_properties)+1, dtype=np.float)
        b = material_vector
        materials = np.vstack((a, b))
        return VoxelModel(new_voxels, materials, self.x, self.y, self.z)

    """
    Boolean operations
    
    - Return a model    
    - Material from base model takes priority by default
    """
    def difference(self, model_to_sub):
        a, b, x_new, y_new, z_new = alignDims(self, model_to_sub)
        mask = np.array(b == 0, dtype=np.bool)
        new_voxels = np.multiply(a, mask)
        return VoxelModel(new_voxels, self.materials, x_new, y_new, z_new)

    def intersection(self, model_2):
        a, b, x_new, y_new, z_new = alignDims(self, model_2)
        mask = np.logical_and(np.array(a != 0, dtype=np.bool), np.array(b != 0, dtype=np.bool))

        new_voxels = np.multiply(a, mask) # material from left model takes priority
        materials = self.materials

        return VoxelModel(new_voxels, materials, x_new, y_new, z_new)

    def __and__(self, other):
        return self.intersection(other)

    def union(self, model_to_add):
        materials = np.vstack((self.materials, model_to_add.materials[1:]))
        a, b, x_new, y_new, z_new = alignDims(self, model_to_add)

        i_offset = len(self.materials) - 1
        b = b + i_offset
        b[b == i_offset] = 0

        # Paper uses a symmetric difference operation combined with the left/right intersection
        # A condensed version of this operation is used here for code simplicity
        mask = np.array(a == 0, dtype=np.bool)
        new_voxels = np.multiply(b, mask)
        new_voxels = new_voxels + a # material from left model takes priority

        return VoxelModel(new_voxels, materials, x_new, y_new, z_new)

    def __or__(self, other):
        return self.union(other)

    def xor(self, model_2):
        materials = np.vstack((self.materials, model_2.materials[1:]))
        a, b, x_new, y_new, z_new = alignDims(self, model_2)

        i_offset = len(self.materials) - 1
        b = b + i_offset
        b[b == i_offset] = 0

        mask1 = np.array(b == 0, dtype=np.bool)
        mask2 = np.array(a == 0, dtype=np.bool)

        new_voxels = np.multiply(a, mask1) + np.multiply(b, mask2)

        return VoxelModel(new_voxels, materials, x_new, y_new, z_new)

    def __xor__(self, other):
        return self.xor(other)

    # Material is computed
    def add(self, model_to_add):
        a, b, x_new, y_new, z_new = alignDims(self, model_to_add)

        x_len = a.shape[0]
        y_len = a.shape[1]
        z_len = a.shape[2]

        new_voxels = np.zeros_like(a, dtype=int)
        new_materials = np.zeros((1, len(material_properties)+1), dtype=np.float)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    i1 = a[x, y, z]
                    i2 = b[x, y, z]
                    m1 = self.materials[i1]
                    m2 = model_to_add.materials[i2]

                    m = m1 + m2
                    m[0] = np.logical_or(m1[0], m2[0])
                    i = np.where(np.equal(new_materials, m).all(1))[0]

                    if len(i) > 0:
                        new_voxels[x, y, z] = i[0]
                    else:
                        new_materials = np.vstack((new_materials, m))
                        new_voxels[x, y, z] = len(new_materials) - 1

        return VoxelModel(new_voxels, new_materials, x_new, y_new, z_new)

    def __add__(self, other):
        return self.add(other)

    # Material is computed
    def subtract(self, model_to_sub):
        a, b, x_new, y_new, z_new = alignDims(self, model_to_sub)

        x_len = a.shape[0]
        y_len = a.shape[1]
        z_len = a.shape[2]

        new_voxels = np.zeros_like(a, dtype=int)
        new_materials = np.zeros((1, len(material_properties) + 1), dtype=np.float)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    i1 = a[x, y, z]
                    i2 = b[x, y, z]
                    m1 = self.materials[i1]
                    m2 = model_to_sub.materials[i2]

                    m = m1 - m2
                    m[0] = np.logical_or(m1[0], m2[0])
                    i = np.where(np.equal(new_materials, m).all(1))[0]

                    if len(i) > 0:
                        new_voxels[x, y, z] = i[0]
                    else:
                        new_materials = np.vstack((new_materials, m))
                        new_voxels[x, y, z] = len(new_materials) - 1

        return VoxelModel(new_voxels, new_materials, x_new, y_new, z_new)

    def __sub__(self, other):
        return self.subtract(other)

    """
    Dilate and Erode
    
    - Return a model
    """
    def dilate(self, radius = 1, plane = Axes.XYZ, connectivity = 3): # TODO: Preserve overlapping materials?
        if radius == 0:
            return VoxelModel.copy(self)

        x_len = self.voxels.shape[0] + (radius * 2)
        y_len = self.voxels.shape[1] + (radius * 2)
        z_len = self.voxels.shape[2] + (radius * 2)

        new_voxels = np.zeros((x_len, y_len, z_len), dtype=np.int32)
        new_voxels[radius:-radius, radius:-radius, radius:-radius] = self.voxels
        struct = ndimage.generate_binary_structure(3, connectivity)

        if plane.value[0] != 1:
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
        if plane.value[1] != 1:
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
        if plane.value[2] != 1:
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)

        for i in range(radius):
            new_voxels = ndimage.grey_dilation(new_voxels, footprint=struct)

        return VoxelModel(new_voxels, self.materials, self.x - radius, self.y - radius, self.z - radius)

    def dilateBounded(self, radius = 1, plane = Axes.XYZ, connectivity = 3): # Dilates a model without increasing the size of its bounding box
        if radius == 0:
            return VoxelModel.copy(self)

        self.fitWorkspace()
        new_voxels = np.copy(self.voxels)
        struct = ndimage.generate_binary_structure(3, connectivity)

        if plane.value[0] != 1:
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
        if plane.value[1] != 1:
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
        if plane.value[2] != 1:
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)

        #print('Dilate Bounded:')
        for i in range(radius):
            new_voxels = ndimage.grey_dilation(new_voxels, footprint=struct)

        return VoxelModel(new_voxels, self.materials, self.x - radius, self.y - radius, self.z - radius)

    def erode(self, radius = 1, plane = Axes.XYZ, connectivity = 3):
        if radius == 0:
            return VoxelModel.copy(self)

        x_len = self.voxels.shape[0] + (radius * 2)
        y_len = self.voxels.shape[1] + (radius * 2)
        z_len = self.voxels.shape[2] + (radius * 2)

        new_voxels = np.zeros((x_len, y_len, z_len), dtype=np.int32)
        new_voxels[radius:-radius, radius:-radius, radius:-radius] = self.voxels
        struct = ndimage.generate_binary_structure(3, connectivity)

        if plane.value[0] != 1:
            struct[0, :, :].fill(0)
            struct[2, :, :].fill(0)
        if plane.value[1] != 1:
            struct[:, 0, :].fill(0)
            struct[:, 2, :].fill(0)
        if plane.value[2] != 1:
            struct[:, :, 0].fill(0)
            struct[:, :, 2].fill(0)

        for i in range(radius):
            mask = np.array(new_voxels != 0, dtype=np.bool)
            mask = ndimage.binary_erosion(mask, structure=struct)
            new_voxels = np.multiply(new_voxels, mask)

        return VoxelModel(new_voxels, self.materials, self.x - radius, self.y - radius, self.z - radius)

    """
    Material Interface Modification
    
    - Return a model
    """
    def blur(self, radius=1):
        if radius == 0:
            return VoxelModel.copy(self)

        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        full_model = np.zeros((x_len, y_len, z_len, len(material_properties)+1), dtype=np.float)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    i = self.voxels[x,y,z]
                    full_model[x,y,z,:] = self.materials[i]

        for m in range(len(material_properties)):
            full_model[:, :, :, m+1] = ndimage.gaussian_filter(full_model[:, :, :, m+1], sigma=radius)

        mask = full_model[:, :, :, 0]
        mask = np.repeat(mask[..., None], len(material_properties)+1, axis=3)
        full_model = np.multiply(full_model, mask)

        new_voxels = np.zeros_like(self.voxels, dtype=int)
        new_materials = np.zeros((1, len(material_properties) + 1), dtype=np.float)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    m = full_model[x, y, z, :]
                    i = np.where(np.equal(new_materials, m).all(1))[0]

                    if len(i) > 0:
                        new_voxels[x, y, z] = i[0]
                    else:
                        new_materials = np.vstack((new_materials, m))
                        new_voxels[x, y, z] = len(new_materials) - 1

        return VoxelModel(new_voxels, new_materials, self.x, self.y, self.z)

    def blurRegion(self, radius, region):
        new_model = self.blur(radius)
        new_model = new_model.intersection(region)
        new_model = new_model.union(self)
        return new_model

    # Add dithering options
    def dither(self, radius=1):
        new_model = self.blur(radius)
        new_model = new_model.scaleValues()
        new_model.voxels = dither(new_model.voxels)
        return new_model

    """
    Cleanup
    
    - Return a model
    """
    def removeNegatives(self): # Remove negative material values (which have no physical meaning)
        new_model = VoxelModel.copy(self)
        new_model.materials[new_model.materials < 1e-10] = 0
        material_sums = np.sum(new_model.materials[:,1:], 1) # This and following update the a values
        material_sums[material_sums > 0] = 1
        new_model.materials[:, 0] = material_sums
        new_model.removeDuplicateMaterials()
        return new_model

    def scaleValues(self): # Scale material values while maintaining the ratio between all materials
        new_model = self.removeNegatives()
        material_sums = np.sum(new_model.materials[:, 1:], 1)
        material_sums[material_sums == 0] = 1
        material_sums = np.repeat(material_sums[..., None], len(material_properties), axis=1)
        new_model.materials[:,1:] = np.divide(new_model.materials[:,1:], material_sums)
        return new_model

    def scaleNull(self): # Scale null values to make all voxels contain 100% material
        new_model = self.removeNegatives()
        material_sums = np.sum(new_model.materials[:, 1:], 1)
        material_sums = np.ones(np.shape(material_sums)) - material_sums
        material_sums[material_sums < 0] = 0
        new_model.materials[:,1] = np.multiply(material_sums, new_model.materials[:,0])
        new_model = new_model.scaleValues()
        return new_model

    def rotate(self, angle, axis = Axes.Z): # TODO: Check that coords are handled correctly
        if axis == Axes.X:
            plane = (1, 2)
        elif axis == Axes.Y:
            plane = (0, 2)
        else: # axis == 'z'
            plane = (0, 1)

        new_model = ndimage.rotate(self.voxels, angle, plane, order=0)

        return VoxelModel(new_model, self.materials, self.x, self.y, self.z)

    def rotate90(self, times = 1, axis = Axes.Z): # TODO: Check that coords are handled correctly
        if axis == Axes.X:
            plane = (1, 2)
        elif axis == Axes.Y:
            plane = (0, 2)
        else: # axis == 'z'
            plane = (0, 1)

        new_model = np.rot90(self.voxels, times, axes=plane)

        return VoxelModel(new_model, self.materials, self.x, self.y, self.z)

    """
    Manufacturing Features
    
    - Return a mask
    """
    def projection(self, direction):
        new_voxels = np.zeros_like(self.voxels)

        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        if direction == Dir.BOTH:
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    if np.sum(self.voxels[x, y, :]) > 0:
                        new_voxels[x, y, :].fill(1)

        elif direction == Dir.DOWN:
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if np.sum(self.voxels[x, y, z:]) > 0:
                            new_voxels[x, y, z] = 1
                        elif np.sum(self.voxels[x, y, z:]) == 0:
                            break

        elif direction == Dir.UP:
            # Loop through model data
            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if np.sum(self.voxels[x, y, :z]) > 0:
                            new_voxels[x, y, z] = 1

        # Assume material 1
        materials = np.zeros((1, len(material_properties) + 1), dtype=np.float)
        material_vector = np.zeros(len(material_properties) + 1, dtype=np.float)
        material_vector[0] = 1
        material_vector[2] = 1
        materials = np.vstack((materials, material_vector))

        return VoxelModel(new_voxels, materials, self.x, self.y, self.z)

    def keepout(self, method):
        if method == Process.LASER:
            new_model = self.projection(Dir.BOTH)
        elif method == Process.MILL:
            new_model = self.projection(Dir.DOWN)
        elif method == Process.INSERT:
            new_model = self.projection(Dir.UP)
        else:
            new_model = self
        return new_model

    def clearance(self, method):
        if method == Process.LASER:
            new_model = self.projection(Dir.BOTH).difference(self)
        elif method == Process.MILL:
            new_model = self.projection(Dir.BOTH).difference(self.projection(Dir.DOWN))
        elif (method == Process.INSERT) or (method == Process.PRINT):
            new_model = self.projection(Dir.UP)
        else:
            new_model = self
        return new_model

    def support(self, method, r1=1, r2=1, plane=Axes.XY):
        model_A = self.keepout(method)
        model_A = model_A.dilate(r2, plane).difference(model_A)
        model_A = model_A.difference(self.keepout(method).difference(self).dilate(r1, plane)) # Valid support regions
        return model_A

    def userSupport(self, support_model, method, r1=1, r2=1, plane=Axes.XY):
        model_A = self.support(method, r1, r2, plane)
        model_A = support_model.intersection(model_A)
        return model_A

    def web(self, method, r1=1, r2=1, layer=-1):
        model_A = self.keepout(method)
        if layer != -1:
            model_A = model_A.isolateLayer(layer)
        model_A = model_A.dilate(r1, Axes.XY)
        model_A = model_A.dilate(r2, Axes.XY).getBoundingBox().difference(model_A)
        return model_A

    """
    File IO

    - Read/write .vf files
    """
    def saveVF(self, filename):
        f = open(filename+'.vf', 'w+')

        f.write('<coords>\n' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ',\n</coords>\n')

        f.write('<materials>\n')
        for r in range(len(self.materials[:,0])):
            for c in range(len(self.materials[0,:])):
                f.write(str(self.materials[r,c]) + ',')
            f.write('\n')
        f.write('</materials>\n')

        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        f.write('<size>\n' + str(x_len) + ',' + str(y_len) + ',' + str(z_len) + ',\n</size>\n')

        f.write('<voxels>\n')
        for x in range(x_len):
            for z in range(z_len):
                for y in range(y_len):
                    f.write(str(int(self.voxels[x,y,z])) + ',')
                f.write(';')
            f.write('\n')
        f.write('</voxels>\n')

        f.write('<components>\n' + str(self.numComponents) + '\n</components>\n')

        f.write('<labels>\n')
        for x in range(x_len):
            for z in range(z_len):
                for y in range(y_len):
                    f.write(str(int(self.components[x,y,z])) + ',')
                f.write(';')
            f.write('\n')
        f.write('</labels>\n')

        print("File created: " + f.name)
        f.close()

    @classmethod
    def openVF(cls, filename):
        f = open(filename + '.vf', 'r')
        data = f.readlines()

        loc = np.zeros((6,2), dtype=np.int32)

        for i in range(len(data)):
            if data[i][:-1] == '<coords>':
                loc[0,0] = i+1
            if data[i][:-1] == '</coords>':
                loc[0,1] = i
            if data[i][:-1] == '<materials>':
                loc[1,0] = i+1
            if data[i][:-1] == '</materials>':
                loc[1,1] = i
            if data[i][:-1] == '<size>':
                loc[2,0] = i+1
            if data[i][:-1] == '</size>':
                loc[2,1] = i
            if data[i][:-1] == '<voxels>':
                loc[3,0] = i+1
            if data[i][:-1] == '</voxels>':
                loc[3,1] = i
            if data[i][:-1] == '<components>':
                loc[4,0] = i+1
            if data[i][:-1] == '</components>':
                loc[4,1] = i
            if data[i][:-1] == '<labels>':
                loc[5,0] = i+1
            if data[i][:-1] == '</labels>':
                loc[5,1] = i

        coords = np.array(data[loc[0,0]][:-2].split(","), dtype=np.int32)

        materials = np.array(data[loc[1,0]][:-2].split(","), dtype=np.float)
        for i in range(loc[1,0]+1, loc[1,1]):
            materials = np.vstack((materials, np.array(data[i][:-2].split(","), dtype=np.float)))

        size = tuple(np.array(data[loc[2,0]][:-2].split(","), dtype=np.int32))

        voxels = np.zeros(size, dtype=np.int32)
        for i in range(loc[3,0], loc[3,1]):
            x = i - loc[3,0]
            yz = data[i][:-2].split(";")
            for z in range(len(yz)):
                y = np.array(yz[z][:-1].split(","), dtype=np.int32)
                voxels[x, :, z] = y

        numComponents = int(data[loc[4,0]][:-1])

        components = np.zeros(size, dtype=np.int32)
        for i in range(loc[5,0], loc[5,1]):
            x = i - loc[5, 0]
            yz = data[i][:-2].split(";")
            for z in range(len(yz)):
                y = np.array(yz[z][:-1].split(","), dtype=np.int32)
                components[x, :, z] = y

        new_model = cls(voxels, materials, coords[0], coords[1], coords[2])
        new_model.numComponents = numComponents
        new_model.components = components

        print("File opened: " + f.name)
        f.close()

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
    xMaxA = modelA.x + modelA.voxels.shape[0]
    yMaxA = modelA.y + modelA.voxels.shape[1]
    zMaxA = modelA.z + modelA.voxels.shape[2]

    xMaxB = modelB.x + modelB.voxels.shape[0]
    yMaxB = modelB.y + modelB.voxels.shape[1]
    zMaxB = modelB.z + modelB.voxels.shape[2]

    xNew = min(modelA.x, modelB.x)
    yNew = min(modelA.y, modelB.y)
    zNew = min(modelA.z, modelB.z)

    xMaxNew = max(xMaxA, xMaxB)
    yMaxNew = max(yMaxA, yMaxB)
    zMaxNew = max(zMaxA, zMaxB)

    modelANew = np.zeros((xMaxNew - xNew, yMaxNew - yNew, zMaxNew - zNew), dtype=np.int32)
    modelBNew = np.zeros((xMaxNew - xNew, yMaxNew - yNew, zMaxNew - zNew), dtype=np.int32)

    modelANew[(modelA.x - xNew):(xMaxA - xNew), (modelA.y - yNew):(yMaxA - yNew), (modelA.z - zNew):(zMaxA - zNew)] = modelA.voxels
    modelBNew[(modelB.x - xNew):(xMaxB - xNew), (modelB.y - yNew):(yMaxB - yNew), (modelB.z - zNew):(zMaxB - zNew)] = modelB.voxels

    return modelANew, modelBNew, xNew, yNew, zNew

@njit(parallel=True)
def findFilledVoxels(a, b):
    x_len = len(a[:, 0, 0])
    y_len = len(a[0, :, 0])
    z_len = len(b[:, 0])

    f3 = np.zeros((x_len, z_len), dtype=np.float32)

    for x in prange(x_len):
        temp = np.zeros((y_len, z_len), dtype=np.float32)
        for y in range(y_len):
            for z in range(z_len):
                temp[y, z] = a[x, y, :].dot(b[z, :])
        f1 = ((temp[:, :] >= (0 - FLOATING_ERROR)).sum(0) == 4)
        f2 = ((temp[:, :] <= (1 + FLOATING_ERROR)).sum(0) == 4)
        f3[x] = f1 & f2

    return f3

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
