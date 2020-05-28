"""
Copyright 2020
Cole Brauer, Dan Aukes
"""

import os
import subprocess
import meshio
import numpy as np
import zlib
import base64
from enum import Enum
from pyvox.parser import VoxParser
from voxelfuse.materials import material_properties
from scipy import ndimage
from numba import njit, prange
from tqdm import tqdm

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

class Struct(Enum):
    STANDARD = 1
    SPHERE = 2

"""
VoxelModel Class

Initialized from a model array or file and position coordinates

Properties:
  model: array storing the index of the material present at each voxel
         
  materials: array of all material mixtures present in model, material format: <a, n, m0, m1, ... mn>
         
  coords: (x, y, z), position of model origin
  
  resolution: number of voxels per mm (higher number = finer resolution)
"""
class VoxelModel:
    def __init__(self, voxels, materials, coords = (0, 0, 0), resolution = 1):
        self.voxels = np.copy(voxels)
        self.materials = np.copy(materials)
        self.coords = coords
        self.resolution = resolution
        self.components = np.zeros_like(voxels)
        self.numComponents = 0

    @classmethod
    def fromVoxFile(cls, filename, coords = (0, 0, 0), resolution = 1):
        # Import data and align axes
        v1 = VoxParser(filename).parse()
        v2 = np.array(v1.to_dense(), dtype=np.uint16)
        v2 = np.flip(v2, 1)
        v2 = np.rot90(v2, 1, (2, 0))
        v2 = np.rot90(v2, 1, (1, 2))

        # Generate materials table assuming indices match materials in material_properties
        i = 0
        materials = np.zeros((1, len(material_properties) + 1), dtype=np.float32)
        for m in np.unique(v2):
            if m != 0:
                i = i+1
                material_vector = np.zeros(len(material_properties) + 1, dtype=np.float32)
                material_vector[0] = 1
                material_vector[m+1] = 1
                materials = np.vstack((materials, material_vector))
                v2[v2 == m] = i

        return cls(v2, materials, coords=coords, resolution=resolution)

    @classmethod
    def fromMeshFile(cls, filename, coords = (0, 0, 0), material = 1, resolution = 1):
        data = makeMesh(filename, True)

        points = data.points

        # Get lists of indices of point
        # ii_tri = data.cells_dict['triangle']
        ii_tet = data.cells_dict['tetra']

        # Convert lists of indices to lists of points
        # tris = points[ii_tri]
        tets = points[ii_tet]

        # Create barycentric coordinate system
        T = np.concatenate((tets, tets[:, :, 0:1] * 0 + 1), 2)
        T_inv = np.zeros(T.shape)

        for ii, t in enumerate(T):
            T_inv[ii] = np.linalg.inv(t).T

        # Find bounding box
        base = 1 / resolution
        points_min = points.min(0)
        points_max = points.max(0)
        points_min_r = base * np.round(points_min/base)
        points_max_r = base * np.round(points_max/base)

        # Create 3D grid
        xx = np.r_[points_min_r[0]:points_max_r[0]+1:base]
        yy = np.r_[points_min_r[1]:points_max_r[1]+1:base]
        zz = np.r_[points_min_r[2]:points_max_r[2]+1:base]

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

        new_model =  cls(voxels, generateMaterials(material), coords=coords, resolution=resolution).fitWorkspace()
        return new_model

    @classmethod
    def empty(cls, size, resolution):
        modelData = np.zeros((size[0], size[1], size[2]), dtype=np.uint16)
        materials = np.zeros(len(material_properties) + 1, dtype=np.float)
        new_model = cls(modelData, materials, resolution=resolution)
        return new_model

    @classmethod
    def emptyLike(cls, voxel_model):
        new_model = cls(np.zeros_like(voxel_model.voxels, dtype=np.uint16), voxel_model.materials, coords=voxel_model.coords, resolution=voxel_model.resolution)
        return new_model

    @classmethod
    def copy(cls, voxel_model):
        new_model = cls(voxel_model.voxels, voxel_model.materials, coords=voxel_model.coords, resolution=voxel_model.resolution)
        new_model.numComponents = voxel_model.numComponents
        new_model.components = voxel_model.components
        return new_model

    """
    Property update operations
    
    - Return a model
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
        new_coords = (self.coords[0] + x_min, self.coords[1] + y_min, self.coords[2] + z_min)

        new_model = VoxelModel(new_voxels, self.materials, coords=new_coords, resolution=self.resolution)
        new_model.numComponents = self.numComponents
        new_model.components = new_components
        return new_model

    # Remove duplicate rows from a model's material array
    def removeDuplicateMaterials(self):
        new_materials = np.unique(self.materials, axis=0)

        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        new_voxels = np.zeros_like(self.voxels, dtype=np.uint16)

        for x in tqdm(range(x_len), desc='Removing duplicate materials'):
            for y in range(y_len):
                for z in range(z_len):
                    i = self.voxels[x, y, z]
                    m = self.materials[i]
                    ni = np.where(np.equal(new_materials, m).all(1))[0][0]
                    new_voxels[x, y, z] = ni

        return VoxelModel(new_voxels, new_materials, self.coords, self.resolution)

    # Update component labels for a model.  This uses a disconnected components algorithm and assumes that adjacent voxels with different materials are connected.
    def getComponents(self, connectivity=1):
        mask = np.array(self.voxels[:, :, :] > 0, dtype=np.bool)
        struct = ndimage.generate_binary_structure(3, connectivity)
        new_model = VoxelModel.copy(self)
        new_model.components, new_model.numComponents = ndimage.label(mask, structure=struct)
        new_model.components = new_model.components.astype(dtype=np.uint8)
        return new_model

    """    
    Selection operations
    
    - Return a model
    """
    # Get all voxels with a specified material
    # TODO: Should this reference the material properties table?
    # TODO: isolateMaterialVector
    def isolateMaterial(self, material): # material input is an index corresponding to the materials array for the model
        mask = np.array(self.voxels == material, dtype=np.bool)
        materials = np.zeros((2, len(material_properties)+1), dtype=np.float32)
        materials[1] = self.materials[material]
        return VoxelModel(mask.astype(int), materials, self.coords, self.resolution)

    # Get all voxels in a specified layer
    def isolateLayer(self, layer):
        new_voxels = np.zeros_like(self.voxels, dtype=np.uint16)
        new_voxels[:, :, layer - self.coords[2]] = self.voxels[:, :, layer - self.coords[2]]
        return VoxelModel(new_voxels, self.materials, self.coords, self.resolution)

    # Isolate component by component label
    # Component labels must first be updated with getComponents
    # Unrecognized component labels will return an empty object
    def isolateComponent(self, component):
        mask = np.array(self.components == component, dtype=np.bool)
        new_voxels = np.multiply(self.voxels, mask)
        return VoxelModel(new_voxels, self.materials, self.coords, self.resolution)

    """
    Mask operations
    
    - Material defaults to the first material in the input model
    """
    # Return all voxels not occupied by the input model
    def getUnoccupied(self):
        mask = np.array(self.voxels == 0, dtype=np.bool)
        return VoxelModel(mask, self.materials[0:2, :], self.coords, self.resolution)

    def __invert__(self):
        return self.getUnoccupied()

    # Return all voxels occupied by the input model
    def getOccupied(self):
        mask = np.array(self.voxels != 0, dtype=np.bool)
        return VoxelModel(mask, self.materials[0:2, :], self.coords, self.resolution)

    # Return the bounding box of the input model
    def getBoundingBox(self):
        new_model = VoxelModel.copy(self)
        new_model = new_model.fitWorkspace()
        new_model.voxels.fill(1)
        new_model = new_model.getOccupied()
        new_model.materials = self.materials[0:2, :]
        return new_model

    # Set the material of a model
    def setMaterial(self, material): # material input is an index corresponding to the material properties table
        new_voxels = self.getOccupied().voxels # Converts input model to a mask, no effect if input is already a mask
        material_vector = np.zeros(len(material_properties)+1, dtype=np.float32)
        material_vector[0] = 1
        material_vector[material+1] = 1
        a = np.zeros(len(material_properties)+1, dtype=np.float32)
        b = material_vector
        m = np.vstack((a, b))
        return VoxelModel(new_voxels, m, self.coords, self.resolution)

    # Set the material of a model
    def setMaterialVector(self, material_vector):  # material input is the desired material vector
        new_voxels = self.getOccupied().voxels  # Converts input model to a mask, no effect if input is already a mask
        a = np.zeros(len(material_properties)+1, dtype=np.float32)
        b = material_vector
        materials = np.vstack((a, b))
        return VoxelModel(new_voxels, materials, self.coords, self.resolution)

    """
    Boolean operations
    
    - Return a model    
    - Material from base model takes priority by default
    """
    def difference(self, model_to_sub):
        checkResolution(self, model_to_sub)
        a, b, new_coords = alignDims(self, model_to_sub)
        mask = np.array(b == 0, dtype=np.bool)
        new_voxels = np.multiply(a, mask)
        return VoxelModel(new_voxels, self.materials, new_coords, self.resolution)

    def intersection(self, model_2):
        checkResolution(self, model_2)
        a, b, new_coords = alignDims(self, model_2)
        mask = np.logical_and(np.array(a != 0, dtype=np.bool), np.array(b != 0, dtype=np.bool))

        new_voxels = np.multiply(a, mask) # material from left model takes priority
        materials = self.materials

        return VoxelModel(new_voxels, materials, new_coords, self.resolution)

    def __and__(self, other):
        return self.intersection(other)

    def union(self, model_to_add):
        checkResolution(self, model_to_add)
        materials = np.vstack((self.materials, model_to_add.materials[1:]))
        a, b, new_coords = alignDims(self, model_to_add)

        i_offset = len(self.materials) - 1
        b = b + i_offset
        b[b == i_offset] = 0

        # Paper uses a symmetric difference operation combined with the left/right intersection
        # A condensed version of this operation is used here for code simplicity
        mask = np.array(a == 0, dtype=np.bool)
        new_voxels = np.multiply(b, mask)
        new_voxels = new_voxels + a # material from left model takes priority

        return VoxelModel(new_voxels, materials, new_coords, self.resolution)

    def __or__(self, other):
        return self.union(other)

    def xor(self, model_2):
        checkResolution(self, model_2)
        materials = np.vstack((self.materials, model_2.materials[1:]))
        a, b, new_coords = alignDims(self, model_2)

        i_offset = len(self.materials) - 1
        b = b + i_offset
        b[b == i_offset] = 0

        mask1 = np.array(b == 0, dtype=np.bool)
        mask2 = np.array(a == 0, dtype=np.bool)

        new_voxels = np.multiply(a, mask1) + np.multiply(b, mask2)

        return VoxelModel(new_voxels, materials, new_coords, self.resolution)

    def __xor__(self, other):
        return self.xor(other)

    # Material is computed
    def add(self, model_to_add):
        checkResolution(self, model_to_add)
        a, b, new_coords = alignDims(self, model_to_add)

        x_len = a.shape[0]
        y_len = a.shape[1]
        z_len = a.shape[2]

        new_voxels = np.zeros_like(a, dtype=np.uint16)
        new_materials = np.zeros((1, len(material_properties)+1), dtype=np.float32)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    i1 = int(a[x, y, z])
                    i2 = int(b[x, y, z])
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

        return VoxelModel(new_voxels, new_materials, new_coords, self.resolution)

    def __add__(self, other):
        return self.add(other)

    # Material is computed
    def subtract(self, model_to_sub):
        checkResolution(self, model_to_sub)
        a, b, new_coords = alignDims(self, model_to_sub)

        x_len = a.shape[0]
        y_len = a.shape[1]
        z_len = a.shape[2]

        new_voxels = np.zeros_like(a, dtype=np.uint16)
        new_materials = np.zeros((1, len(material_properties) + 1), dtype=np.float32)

        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    i1 = int(a[x, y, z])
                    i2 = int(b[x, y, z])
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

        return VoxelModel(new_voxels, new_materials, new_coords, self.resolution)

    def __sub__(self, other):
        return self.subtract(other)

    def multiply(self, other):
        if type(other) is VoxelModel:
            checkResolution(self, other)
            a, b, new_coords = alignDims(self, other)

            x_len = a.shape[0]
            y_len = a.shape[1]
            z_len = a.shape[2]

            new_voxels = np.zeros_like(a, dtype=np.uint16)
            new_materials = np.zeros((1, len(material_properties)+1), dtype=np.float32)

            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        i1 = int(a[x, y, z])
                        i2 = int(b[x, y, z])
                        m1 = self.materials[i1]
                        m2 = other.materials[i2]

                        m = np.multiply(m1, m2)
                        m[0] = np.logical_and(m1[0], m2[0])

                        i = np.where(np.equal(new_materials, m).all(1))[0]
                        if len(i) > 0:
                            new_voxels[x, y, z] = i[0]
                        else:
                            new_materials = np.vstack((new_materials, m))
                            new_voxels[x, y, z] = len(new_materials) - 1

            return VoxelModel(new_voxels, new_materials, new_coords, self.resolution)

        else:
            new_model = VoxelModel.copy(self)
            new_model.materials[1:, 1:] = np.multiply(new_model.materials[1:, 1:], other)
            return new_model

    def __mul__(self, other):
        return self.multiply(other)

    def divide(self, other):
        if type(other) is VoxelModel:
            checkResolution(self, other)
            a, b, new_coords = alignDims(self, other)

            x_len = a.shape[0]
            y_len = a.shape[1]
            z_len = a.shape[2]

            new_voxels = np.zeros_like(a, dtype=np.uint16)
            new_materials = np.zeros((1, len(material_properties)+1), dtype=np.float32)

            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        i1 = int(a[x, y, z])
                        i2 = int(b[x, y, z])
                        m1 = self.materials[i1]
                        m2 = other.materials[i2]

                        m2[m2 == 0] = 1
                        m = np.divide(m1, m2)
                        m[0] = m1[0]

                        i = np.where(np.equal(new_materials, m).all(1))[0]
                        if len(i) > 0:
                            new_voxels[x, y, z] = i[0]
                        else:
                            new_materials = np.vstack((new_materials, m))
                            new_voxels[x, y, z] = len(new_materials) - 1

            return VoxelModel(new_voxels, new_materials, new_coords, self.resolution)

        else:
            if other == 0:
                return self

            new_model = VoxelModel.copy(self)
            new_model.materials[1:, 1:] = np.divide(new_model.materials[1:, 1:], other)
            return new_model

    def __truediv__(self, other):
        return self.divide(other)

    """
    Morphology Operations
    
    - Return a model
    """
    def dilate(self, radius = 1, plane = Axes.XYZ, structType = Struct.STANDARD, connectivity = 3): # TODO: Preserve overlapping materials?
        if radius == 0:
            return VoxelModel.copy(self)

        x_len = self.voxels.shape[0] + (radius * 2)
        y_len = self.voxels.shape[1] + (radius * 2)
        z_len = self.voxels.shape[2] + (radius * 2)

        new_voxels = np.zeros((x_len, y_len, z_len), dtype=np.uint16)
        new_voxels[radius:-radius, radius:-radius, radius:-radius] = self.voxels

        if structType == Struct.SPHERE:
            struct = structSphere(radius, plane)
            new_voxels = ndimage.grey_dilation(new_voxels, footprint=struct)
        else: # Struct.STANDARD
            struct = structStandard(connectivity, plane)
            for i in range(radius):
                new_voxels = ndimage.grey_dilation(new_voxels, footprint=struct)

        return VoxelModel(new_voxels, self.materials, (self.coords[0] - radius, self.coords[1] - radius, self.coords[2] - radius), self.resolution)

    def dilateBounded(self, radius = 1, plane = Axes.XYZ, structType = Struct.STANDARD, connectivity = 3): # Dilates a model without increasing the size of its bounding box
        if radius == 0:
            return VoxelModel.copy(self)

        new_voxels = np.copy(self.fitWorkspace().voxels)

        if structType == Struct.SPHERE:
            struct = structSphere(radius, plane)
            new_voxels = ndimage.grey_dilation(new_voxels, footprint=struct)
        else: # Struct.STANDARD
            struct = structStandard(connectivity, plane)
            for i in range(radius):
                new_voxels = ndimage.grey_dilation(new_voxels, footprint=struct)

        return VoxelModel(new_voxels, self.materials, self.coords, self.resolution)

    def erode(self, radius = 1, plane = Axes.XYZ, structType = Struct.STANDARD, connectivity = 3):
        if radius == 0:
            return VoxelModel.copy(self)

        new_voxels = np.copy(self.voxels)
        mask = np.array(new_voxels != 0, dtype=np.bool)

        if structType == Struct.SPHERE:
            struct = structSphere(radius, plane)
            mask = ndimage.binary_erosion(mask, structure=struct)
        else: # Struct.STANDARD
            struct = structStandard(connectivity, plane)
            mask = ndimage.binary_erosion(mask, structure=struct, iterations=radius)

        new_voxels = np.multiply(new_voxels, mask)

        return VoxelModel(new_voxels, self.materials, self.coords, self.resolution)

    def closing(self, radius = 1, plane = Axes.XYZ, structType = Struct.SPHERE, connectivity = 1):
        if radius == 0:
            return VoxelModel.copy(self)
        else:
            return self.dilate(radius, plane, structType, connectivity).erode(radius, plane, structType, connectivity).fitWorkspace()

    def opening(self, radius = 1, plane = Axes.XYZ, structType = Struct.SPHERE, connectivity = 1):
        if radius == 0:
            return VoxelModel.copy(self)

        new_voxels = np.copy(self.voxels)
        mask = np.array(new_voxels != 0, dtype=np.bool)

        if structType == Struct.SPHERE:
            struct = structSphere(radius, plane)
            mask = ndimage.binary_opening(mask, structure=struct)
        else:  # Struct.STANDARD
            struct = structStandard(connectivity, plane)
            mask = ndimage.binary_opening(mask, structure=struct, iterations=radius)

        new_voxels = np.multiply(new_voxels, mask)

        return VoxelModel(new_voxels, self.materials, self.coords, self.resolution)

    """
    Material Interface Modification
    
    - Return a model
    """
    def blur(self, radius=1):
        if radius == 0:
            return VoxelModel.copy(self)

        full_model = toFullMaterials(self.voxels, self.materials, len(material_properties)+1)

        for m in tqdm(range(len(material_properties)), desc='Blur - applying gaussian filter'):
            full_model[:, :, :, m+1] = ndimage.gaussian_filter(full_model[:, :, :, m+1], sigma=radius/2)

        mask = full_model[:, :, :, 0]
        mask = np.repeat(mask[..., None], len(material_properties)+1, axis=3)
        full_model = np.multiply(full_model, mask)

        return toIndexedMaterials(full_model, self, self.resolution)

    def blurRegion(self, radius, region):
        new_model = self.blur(radius)
        new_model = new_model.intersection(region)
        new_model = new_model.union(self)
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

    def round(self, toNearest = 0.1):
        new_materials = np.copy(self.materials)
        new_model = VoxelModel.copy(self)

        mult = new_materials / toNearest
        floorDiff = np.round(abs(mult - np.floor(mult)), 10)
        ceilDiff = np.round(abs(mult - np.ceil(mult)), 10)

        new_materials[floorDiff < ceilDiff] = toNearest * np.floor(mult[floorDiff < ceilDiff])
        new_materials[floorDiff >= ceilDiff] = toNearest * np.ceil(mult[floorDiff >= ceilDiff])

        new_model.materials = new_materials
        return new_model

    def clearNull(self):
        new_model = VoxelModel.copy(self)
        new_model.materials[1:, 1] = np.zeros(np.shape(new_model.materials[1:,1]))
        return new_model

    def setDensity(self, density = 1.0):
        new_model = self.clearNull()
        new_model = new_model.scaleValues()
        null_material_values = np.multiply(np.ones(np.shape(new_model.materials[1:,1])), 1-density)
        new_model.materials[1:, 1] = null_material_values
        new_model.materials[1:, 2:] = np.multiply(new_model.materials[1:, 2:], density)
        return new_model

    """
    Transformations

    - Return a model
    """
    def translate(self, vector = (0, 0, 0)):
        new_model = VoxelModel.copy(self)
        new_model.coords = (self.coords[0]+vector[0], self.coords[1]+vector[1], self.coords[2]+vector[2])
        return new_model

    def translateMM(self, vector = (0.0, 0.0, 0.0)):
        xV = int(round(vector[0] * self.resolution))
        yV = int(round(vector[1] * self.resolution))
        zV = int(round(vector[2] * self.resolution))
        vector_voxels = (xV, yV, zV)
        new_model = self.translate(vector_voxels)
        return new_model

    def rotate(self, angle, axis = Axes.Z):
        if axis == Axes.X:
            plane = (1, 2)
        elif axis == Axes.Y:
            plane = (0, 2)
        else: # axis == 'z'
            plane = (0, 1)

        centerCoords = self.getCenter()
        new_voxels = ndimage.rotate(self.voxels, angle, plane, order=0)
        new_model = VoxelModel(new_voxels, self.materials, self.coords, self.resolution)
        new_model = new_model.setCenter(centerCoords)

        return new_model

    def rotate90(self, times = 1, axis = Axes.Z):
        if axis == Axes.X or axis == 0:
            plane = (1, 2)
        elif axis == Axes.Y or axis == 1:
            plane = (0, 2)
        else: # axis == Axes.Z or axis = 2
            plane = (0, 1)

        centerCoords = self.getCenter()
        new_voxels = np.rot90(self.voxels, times, axes=plane)
        new_model = VoxelModel(new_voxels, self.materials, self.coords, self.resolution)
        new_model = new_model.setCenter(centerCoords)

        return new_model

    def scale(self, factor, adjustResolution = True):
        model = self.fitWorkspace()

        x_len = int(model.voxels.shape[0] * factor)
        y_len = int(model.voxels.shape[1] * factor)
        z_len = int(model.voxels.shape[2] * factor)

        new_voxels = np.zeros((x_len, y_len, z_len))
        for x in tqdm(range(x_len), desc='Scaling'):
            for y in range(y_len):
                for z in range(z_len):
                    x_source = int(((x+1) / x_len) * (model.voxels.shape[0]-1))
                    y_source = int(((y+1) / y_len) * (model.voxels.shape[1]-1))
                    z_source = int(((z+1) / z_len) * (model.voxels.shape[2]-1))
                    new_voxels[x,y,z] = model.voxels[x_source, y_source, z_source]

        model.voxels = new_voxels.astype(dtype=np.uint16)
        model = model.setCoords(model.coords)

        if adjustResolution:
            model.resolution = model.resolution * factor

        return model

    def scaleToSize(self, x_len, y_len, z_len):
        model = self.fitWorkspace()

        new_voxels = np.zeros((x_len, y_len, z_len))
        for x in tqdm(range(x_len), desc='Scaling'):
            for y in range(y_len):
                for z in range(z_len):
                    x_source = int(((x+1) / x_len) * (model.voxels.shape[0]-1))
                    y_source = int(((y+1) / y_len) * (model.voxels.shape[1]-1))
                    z_source = int(((z+1) / z_len) * (model.voxels.shape[2]-1))
                    new_voxels[x,y,z] = model.voxels[x_source, y_source, z_source]

        model.voxels = new_voxels.astype(dtype=np.uint16)
        new_model = model.setCoords(model.coords)

        return new_model

    def getCenter(self):
        model = self.fitWorkspace()

        x_center = (model.voxels.shape[0] / 2) + model.coords[0]
        y_center = (model.voxels.shape[1] / 2) + model.coords[1]
        z_center = (model.voxels.shape[2] / 2) + model.coords[2]

        centerCoords = (x_center, y_center, z_center)
        return centerCoords

    def setCenter(self, coords):
        new_model = self.fitWorkspace()

        x_new = int(round(coords[0] - (new_model.voxels.shape[0] / 2)))
        y_new = int(round(coords[1] - (new_model.voxels.shape[1] / 2)))
        z_new = int(round(coords[2] - (new_model.voxels.shape[2] / 2)))

        new_model.coords = (x_new, y_new, z_new)
        return new_model

    def setCoords(self, coords):
        new_model = self.fitWorkspace()
        new_model.coords = coords
        return new_model

    """
    Model Info
    """
    def getVolume(self, component = 0, material = 0):
        new_model = VoxelModel.copy(self)
        if component > 0:
            new_model = new_model.isolateComponent(component)
        if material > 0:
            new_model = new_model.isolateMaterial(material)
        volumeVoxels = np.count_nonzero(new_model.voxels)
        volumeMM3 = volumeVoxels * ((1/self.resolution)**3)
        return volumeVoxels, volumeMM3

    """
    Manufacturing Features
    
    - Return a mask
    """
    def projection(self, direction, material = 1):
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

        return VoxelModel(new_voxels, generateMaterials(material), self.coords, self.resolution)

    def keepout(self, method, material = 1):
        if method == Process.LASER:
            new_model = self.projection(Dir.BOTH, material)
        elif method == Process.MILL:
            new_model = self.projection(Dir.DOWN, material)
        elif method == Process.INSERT:
            new_model = self.projection(Dir.UP, material)
        else:
            new_model = self
        return new_model

    def clearance(self, method, material = 1):
        if method == Process.LASER:
            new_model = self.projection(Dir.BOTH, material).difference(self)
        elif method == Process.MILL:
            new_model = self.projection(Dir.BOTH, material).difference(self.projection(Dir.DOWN, material))
        elif (method == Process.INSERT) or (method == Process.PRINT):
            new_model = self.projection(Dir.UP, material)
        else:
            new_model = self
        return new_model

    def support(self, method, r1=1, r2=1, plane=Axes.XY, material = 1):
        model_A = self.keepout(method, material)
        model_A = model_A.dilate(r2, plane).difference(model_A)
        model_A = model_A.difference(self.keepout(method, material).difference(self).dilate(r1, plane)) # Valid support regions
        return model_A

    def userSupport(self, support_model, method, r1=1, r2=1, plane=Axes.XY, material = -1): # Material = -1 -> use support model material
        if material > -1:
            model_A = self.support(method, r1, r2, plane)
            model_A = support_model.intersection(model_A)
        else:
            model_A = self.support(method, r1, r2, plane, material)
            model_A = model_A.intersection(support_model)
        return model_A

    def web(self, method, r1=1, r2=1, layer=-1, material = 1): # Layer = -1 -> use all layers
        model_A = self.keepout(method, material)
        if layer != -1:
            model_A = model_A.isolateLayer(layer)
        model_A = model_A.dilate(r1, Axes.XY)
        model_A = model_A.dilate(r2, Axes.XY).getBoundingBox().difference(model_A)
        return model_A

    """
    File IO

    - Read/write .vf files
    - Read/write .vxc files
    - Read/write .vxa files
    """
    def saveVF(self, filename):
        f = open(filename+'.vf', 'w+')
        print('Saving file: ' + f.name)

        x_coord = self.coords[0]
        y_coord = self.coords[1]
        z_coord = self.coords[2]

        f.write('<coords>\n' + str(x_coord) + ',' + str(y_coord) + ',' + str(z_coord) + ',\n</coords>\n')

        f.write('<resolution>\n' + str(self.resolution) + '\n</resolution>\n')

        f.write('<materials>\n')
        for r in tqdm(range(len(self.materials[:,0])), desc='Writing materials'):
            for c in range(len(self.materials[0,:])):
                f.write(str(self.materials[r,c]) + ',')
            f.write('\n')
        f.write('</materials>\n')

        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        f.write('<size>\n' + str(x_len) + ',' + str(y_len) + ',' + str(z_len) + ',\n</size>\n')

        f.write('<voxels>\n')
        for x in tqdm(range(x_len), desc='Writing voxels'):
            for z in range(z_len):
                for y in range(y_len):
                    f.write(str(int(self.voxels[x,y,z])) + ',')
                f.write(';')
            f.write('\n')
        f.write('</voxels>\n')

        f.write('<components>\n' + str(self.numComponents) + '\n</components>\n')

        if self.numComponents > 0:
            f.write('<labels>\n')
            for x in tqdm(range(x_len), desc='Writing components'):
                for z in range(z_len):
                    for y in range(y_len):
                        f.write(str(int(self.components[x,y,z])) + ',')
                    f.write(';')
                f.write('\n')
            f.write('</labels>\n')

        f.close()

    @classmethod
    def openVF(cls, filename):
        if filename[-3:] == '.vf':
            f = open(filename, 'r')
        else:
            f = open(filename + '.vf', 'r')
        print('Opening file: ' + f.name)

        data = f.readlines()
        loc = np.ones((7,2), dtype=np.uint16)
        loc = np.multiply(loc, -1)

        for i in tqdm(range(len(data)), desc='Finding tags'):
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
            if data[i][:-1] == '<resolution>':
                loc[6,0] = i+1
            if data[i][:-1] == '</resolution>':
                loc[6,1] = i

        coords = np.array(data[loc[0,0]][:-2].split(","), dtype=np.int16)

        if loc[6,0] > -1:
            resolution = int(data[loc[6,0]][:-1])
        else:
            resolution = 1

        materials = np.array(data[loc[1,0]][:-2].split(","), dtype=np.float32)
        for i in tqdm(range(loc[1,0]+1, loc[1,1]), desc='Reading materials'):
            materials = np.vstack((materials, np.array(data[i][:-2].split(","), dtype=np.float32)))

        size = tuple(np.array(data[loc[2,0]][:-2].split(","), dtype=np.uint16))

        voxels = np.zeros(size, dtype=np.uint16)
        for i in tqdm(range(loc[3,0], loc[3,1]), desc='Reading voxels'):
            x = i - loc[3,0]
            yz = data[i][:-2].split(";")
            for z in range(len(yz)):
                y = np.array(yz[z][:-1].split(","), dtype=np.uint16)
                voxels[x, :, z] = y

        numComponents = int(data[loc[4,0]][:-1])

        components = np.zeros(size, dtype=np.uint8)
        if numComponents > 0:
            for i in tqdm(range(loc[5,0], loc[5,1]), desc='Reading components'):
                x = i - loc[5, 0]
                yz = data[i][:-2].split(";")
                for z in range(len(yz)):
                    y = np.array(yz[z][:-1].split(","), dtype=np.uint8)
                    components[x, :, z] = y

        new_model = cls(voxels, materials, coords=tuple(coords), resolution=resolution)
        new_model.numComponents = numComponents
        new_model.components = components

        f.close()

        return new_model

    def saveVXC(self, filename, compression=False):
        f = open(filename + '.vxc', 'w+')
        print('Saving file: ' + f.name)

        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        self.writeVXCData(f, compression)

        f.close()

    def writeVXCData(self, f, compression=False):
        f.write('<VXC Version="' + str(0.94) + '">\n')

        # Lattice settings
        f.write('  <Lattice>\n')
        f.write('    <Lattice_Dim>' + str((1/self.resolution) * 0.001) + '</Lattice_Dim>\n')
        f.write('    <X_Dim_Adj>' + str(1) + '</X_Dim_Adj>\n')
        f.write('    <Y_Dim_Adj>' + str(1) + '</Y_Dim_Adj>\n')
        f.write('    <Z_Dim_Adj>' + str(1) + '</Z_Dim_Adj>\n')
        f.write('    <X_Line_Offset>' + str(0) + '</X_Line_Offset>\n')
        f.write('    <Y_Line_Offset>' + str(0) + '</Y_Line_Offset>\n')
        f.write('    <X_Layer_Offset>' + str(0) + '</X_Layer_Offset>\n')
        f.write('    <Y_Layer_Offset>' + str(0) + '</Y_Layer_Offset>\n')
        f.write('  </Lattice>\n')

        # Voxel settings
        f.write('  <Voxel>\n')
        f.write('    <Vox_Name>BOX</Vox_Name>\n')
        f.write('    <X_Squeeze>' + str(1) + '</X_Squeeze>\n')
        f.write('    <Y_Squeeze>' + str(1) + '</Y_Squeeze>\n')
        f.write('    <Z_Squeeze>' + str(1) + '</Z_Squeeze>\n')
        f.write('  </Voxel>\n')

        # Materials
        f.write('  <Palette>\n')
        for row in tqdm(range(1, len(self.materials[:, 0])), desc='Writing materials'):
            avgProps = {}
            for key in material_properties[0]:
                if key == 'name' or key == 'process':
                    string = ''
                    for i in range(len(material_properties)):
                        if self.materials[row][i+1] > 0:
                            string = string + material_properties[i][key] + ' '
                    avgProps.update({key: string})
                elif key == 'MM' or key == 'FM':
                    var = 0
                    for i in range(len(material_properties)):
                        if self.materials[row][i + 1] > 0:
                            var = max(var, material_properties[i][key])
                    avgProps.update({key: var})
                else:
                    var = 0
                    for i in range(len(material_properties)):
                        var = var + self.materials[row][i + 1] * material_properties[i][key]
                    avgProps.update({key: var})

            f.write('    <Material ID="' + str(row) + '">\n')
            f.write('      <MatType>' + str(0) + '</MatType>\n')
            f.write('      <Name>' + avgProps['name'][0:-1] + '</Name>\n')
            f.write('      <Display>\n')
            f.write('        <Red>' + str(avgProps['r']) + '</Red>\n')
            f.write('        <Green>' + str(avgProps['g']) + '</Green>\n')
            f.write('        <Blue>' + str(avgProps['b']) + '</Blue>\n')
            f.write('        <Alpha>' + str(1) + '</Alpha>\n')
            f.write('      </Display>\n')
            f.write('      <Mechanical>\n')
            f.write('        <MatModel>' + str(int(avgProps['MM'])) + '</MatModel>\n')
            f.write('        <Elastic_Mod>' + str(avgProps['E']) + '</Elastic_Mod>\n')
            f.write('        <Plastic_Mod>' + str(avgProps['Z']) + '</Plastic_Mod>\n')
            f.write('        <Yield_Stress>' + str(avgProps['eY']) + '</Yield_Stress>\n')
            f.write('        <FailModel>' + str(int(avgProps['FM'])) + '</FailModel>\n')
            f.write('        <Fail_Stress>' + str(avgProps['eF']) + '</Fail_Stress>\n')
            f.write('        <Fail_Strain>' + str(avgProps['SF']) + '</Fail_Strain>\n')
            f.write('        <Density>' + str(avgProps['p'] * 1e3) + '</Density>\n') # Convert g/cm^3 to kg/m^3
            f.write('        <Poissons_Ratio>' + str(avgProps['v']) + '</Poissons_Ratio>\n')
            f.write('        <CTE>' + str(avgProps['CTE']) + '</CTE>\n')
            f.write('        <MaterialTempPhase>' + str(avgProps['TP']) + '</MaterialTempPhase>\n')
            f.write('        <uStatic>' + str(avgProps['uS']) + '</uStatic>\n')
            f.write('        <uDynamic>' + str(avgProps['uD']) + '</uDynamic>\n')
            f.write('      </Mechanical>\n')
            f.write('    </Material>\n')
        f.write('  </Palette>\n')

        # Structure
        if compression:
            f.write('  <Structure Compression="ZLIB">\n')
        else:
            f.write('  <Structure Compression="ASCII_READABLE">\n')

        x_len = self.voxels.shape[0]
        y_len = self.voxels.shape[1]
        z_len = self.voxels.shape[2]

        f.write('    <X_Voxels>' + str(x_len) + '</X_Voxels>\n')
        f.write('    <Y_Voxels>' + str(y_len) + '</Y_Voxels>\n')
        f.write('    <Z_Voxels>' + str(z_len) + '</Z_Voxels>\n')
        f.write('    <Data>\n')

        for z in tqdm(range(z_len), desc='Writing voxels'):
            layer = np.copy(self.voxels[:, :, z])
            layer = layer.transpose()
            layerData = layer.flatten()
            layerData = layerData.astype('uint8')

            if compression:
                layerData = zlib.compress(layerData.tobytes())
                layerData = base64.encodebytes(layerData)
                layerDataStr = str(layerData)[2:-3]
            else:
                layerDataStr = ''
                for vox in layerData:
                    layerDataStr = layerDataStr + str(vox)

            f.write('      <Layer><![CDATA[' + layerDataStr + ']]></Layer>\n')

        f.write('    </Data>\n')
        f.write('  </Structure>\n')
        f.write('</VXC>\n')

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
    ax = modelA.coords[0]
    ay = modelA.coords[1]
    az = modelA.coords[2]

    bx = modelB.coords[0]
    by = modelB.coords[1]
    bz = modelB.coords[2]

    xMaxA = ax + modelA.voxels.shape[0]
    yMaxA = ay + modelA.voxels.shape[1]
    zMaxA = az + modelA.voxels.shape[2]

    xMaxB = bx + modelB.voxels.shape[0]
    yMaxB = by + modelB.voxels.shape[1]
    zMaxB = bz + modelB.voxels.shape[2]

    xNew = min(ax, bx)
    yNew = min(ay, by)
    zNew = min(az, bz)

    xMaxNew = max(xMaxA, xMaxB)
    yMaxNew = max(yMaxA, yMaxB)
    zMaxNew = max(zMaxA, zMaxB)

    voxelsANew = np.zeros((xMaxNew - xNew, yMaxNew - yNew, zMaxNew - zNew), dtype=np.uint16)
    voxelsBNew = np.zeros((xMaxNew - xNew, yMaxNew - yNew, zMaxNew - zNew), dtype=np.uint16)

    voxelsANew[(ax - xNew):(xMaxA - xNew), (ay - yNew):(yMaxA - yNew), (az - zNew):(zMaxA - zNew)] = modelA.voxels
    voxelsBNew[(bx - xNew):(xMaxB - xNew), (by - yNew):(yMaxB - yNew), (bz - zNew):(zMaxB - zNew)] = modelB.voxels

    return voxelsANew, voxelsBNew, (xNew, yNew, zNew)

def checkResolution(modelA, modelB):
    a = modelA.resolution
    b = modelB.resolution
    if a != b:
        print('WARNING: inconsistent resolutions: ' + str(a) + ', ' + str(b))
        return 0
    else:
        return 1

"""
Functions to generate structuring elements
"""
def structSphere(radius, plane):
    diameter = (radius * 2) + 1
    struct = np.zeros((diameter, diameter, diameter), dtype=np.bool)
    for x in range(diameter):
        for y in range(diameter):
            for z in range(diameter):
                xd = (x - radius)
                yd = (y - radius)
                zd = (z - radius)
                r = np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)

                if r < (radius + .5):
                    struct[x, y, z] = 1

    if plane.value[0] != 1:
        struct[:radius, :, :].fill(0)
        struct[-radius:, :, :].fill(0)
    if plane.value[1] != 1:
        struct[:, :radius, :].fill(0)
        struct[:, -radius:, :].fill(0)
    if plane.value[2] != 1:
        struct[:, :, :radius].fill(0)
        struct[:, :, -radius:].fill(0)

    return struct

def structStandard(connectivity, plane):
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

    return struct

def generateMaterials(m):
    materials = np.zeros(len(material_properties) + 1, dtype=np.float)
    material_vector = np.zeros(len(material_properties) + 1, dtype=np.float)
    material_vector[0] = 1
    material_vector[m+1] = 1
    materials = np.vstack((materials, material_vector))
    return materials

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
def toFullMaterials(voxels, materials, n_materials):
    x_len = voxels.shape[0]
    y_len = voxels.shape[1]
    z_len = voxels.shape[2]

    full_model = np.zeros((x_len, y_len, z_len, n_materials), dtype=np.float32)

    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                i = voxels[x,y,z]
                full_model[x,y,z,:] = materials[i]

    return full_model

def toIndexedMaterials(voxels, model, resolution):
    x_len = model.voxels.shape[0]
    y_len = model.voxels.shape[1]
    z_len = model.voxels.shape[2]

    new_voxels = np.zeros((x_len, y_len, z_len), dtype=np.int32)
    new_materials = np.zeros((1, len(material_properties) + 1), dtype=np.float32)

    for x in range(x_len): # tqdm(range(x_len), desc='Converting to indexed materials'):
        for y in range(y_len):
            for z in range(z_len):
                m = voxels[x, y, z, :]
                i = np.where(np.equal(new_materials, m).all(1))[0]

                if len(i) > 0:
                    new_voxels[x, y, z] = i[0]
                else:
                    new_materials = np.vstack((new_materials, m))
                    new_voxels[x, y, z] = len(new_materials) - 1

    return VoxelModel(new_voxels, new_materials, coords=model.coords, resolution=resolution)