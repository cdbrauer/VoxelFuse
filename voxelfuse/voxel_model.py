"""
VoxelModel Class

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import os
import subprocess
import meshio
import numpy as np
import zlib
import base64
from enum import Enum
from typing import Tuple, TextIO
from pyvox.parser import VoxParser
from voxelfuse.materials import material_properties
from scipy import ndimage
from numba import njit, prange
from tqdm import tqdm

# Floating point error threshold for rounding to zero
FLOATING_ERROR = 0.0000000001

class Axes(Enum):
    """
    Options for axes and planes.
    """
    X = (1,0,0)
    Y = (0,1,0)
    Z = (0,0,1)
    XY = (1,1,0)
    XZ = (1,0,1)
    YZ = (0,1,1)
    XYZ = (1,1,1)

class Dir(Enum):
    """
    Options for projection directions.
    """
    UP = 1
    DOWN = 2
    BOTH = 3

class Process(Enum):
    """
    Options for manufacturing process types.
    """
    LASER = 1
    MILL = 2
    PRINT = 3
    CAST = 4
    INSERT = 5

class Struct(Enum):
    """
    Options for structuring element shapes.
    """
    STANDARD = 1
    SPHERE = 2

class VoxelModel:
    """
    VoxelModel object that stores geometry, position, and material information.
    """

    def __init__(self, voxels, materials, coords: Tuple[int, int, int] = (0, 0, 0), resolution: float = 1):
        """
        Initialize a VoxelModel object.

        :param voxels: Array storing the index of the material present at each voxel
        :param materials: Array of all material mixtures present in model, material format: (a, m0, m1, ... mn)
        :param coords: Model origin coordinates
        :param resolution: Number of voxels per mm (higher number = finer resolution)
        """
        self.voxels = np.copy(voxels) # Use np.copy to break references
        self.materials = np.copy(materials)
        self.coords = coords
        self.resolution = resolution
        self.components = np.zeros_like(voxels)
        self.numComponents = 0

    @classmethod
    def fromVoxFile(cls, filename: str, coords: Tuple[int, int, int] = (0, 0, 0), resolution: float = 1):
        """
        Create a VoxelModel from an imported .vox file.

        ----

        Example:

        ``model1 = VoxelModel.fromVoxFile('cylinder-red.vox', (0, 5, 0), 1)``

        ----

        :param filename: File name with extension
        :param coords: Model origin coordinates
        :param resolution: Number of voxels per mm
        :return: VoxelModel
        """
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
    def fromMeshFile(cls, filename: str, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1, gmsh_on_path: bool = False):
        """
        Create a VoxelModel from an imported mesh file.

        ----

        Example:

        ``model1 = VoxelModel.fromMeshFile('center.stl', (67, 3, 0), 2, 1)``

        ____

        :param filename: File name with extension
        :param coords: Model origin coordinates
        :param material: Material index corresponding to materials.py
        :param resolution: Number of voxels per mm
        :param gmsh_on_path: Enable/disable using system gmsh rather than bundled gmsh
        :return: VoxelModel
        """
        data = makeMesh(filename, True, gmsh_on_path)

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
    def empty(cls, size: Tuple[int, int, int], resolution: float = 1):
        """
        Initialize an empty VoxelModel.

        :param size: Size of the empty model in voxels
        :param resolution: Number of voxels per mm
        :return: VoxelModel
        """
        modelData = np.zeros(size, dtype=np.uint16)
        materials = np.zeros((1, len(material_properties) + 1), dtype=np.float)
        new_model = cls(modelData, materials, resolution=resolution)
        return new_model

    @classmethod
    def emptyLike(cls, voxel_model):
        """
        Initialize an empty VoxelModel with the same size, coords, and resolution as another model.

        :param voxel_model: Reference VoxelModel object
        :return: VoxelModel
        """
        new_model = cls(np.zeros_like(voxel_model.voxels, dtype=np.uint16), voxel_model.materials, coords=voxel_model.coords, resolution=voxel_model.resolution)
        return new_model

    @classmethod
    def copy(cls, voxel_model):
        """
        Initialize an VoxelModel that is a copy of another model.

        :param voxel_model: Reference VoxelModel object
        :return: VoxelModel
        """
        new_model = cls(voxel_model.voxels, voxel_model.materials, coords=voxel_model.coords, resolution=voxel_model.resolution)
        new_model.numComponents = voxel_model.numComponents
        new_model.components = voxel_model.components
        return new_model

    # Property update operations ##############################

    def fitWorkspace(self):
        """
        Remove excess empty space from a model.

        Resize the workspace around a model to remove excess empty space.
        Model coordinates are updated to reflect the change.

        :return: VoxelModel
        """
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

    def removeDuplicateMaterials(self):
        """
        Remove duplicate rows from a model's material array.

        :return: VoxelModel
        """
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

    def getComponents(self, connectivity: int = 1):
        """
        Update component labels for a model.

        This function uses a disconnected components algorithm and assumes that adjacent
        voxels with different materials are connected. Connectivity can be set to 1-3
        and defines the shape of the structuring element.

        :param connectivity: Connectivity of structuring element (1-3)
        :return: VoxelModel
        """
        mask = np.array(self.voxels[:, :, :] > 0, dtype=np.bool)
        struct = ndimage.generate_binary_structure(3, connectivity)
        new_model = VoxelModel.copy(self)
        new_model.components, new_model.numComponents = ndimage.label(mask, structure=struct)
        new_model.components = new_model.components.astype(dtype=np.uint8)
        return new_model

    # Selection operations ##############################

    # TODO: Should this reference the material properties table?
    # TODO: isolateMaterialVector
    def isolateMaterial(self, material: int):
        """
        Get all voxels with a specified material.

        ----

        Example:

        ``model2 = model1.isolateMaterial(4)``

        ----

        :param material: Material index corresponding to the materials array for the model
        :return: VoxelModel
        """
        mask = np.array(self.voxels == material, dtype=np.bool)
        materials = np.zeros((2, len(material_properties)+1), dtype=np.float32)
        materials[1] = self.materials[material]
        return VoxelModel(mask.astype(int), materials, self.coords, self.resolution)

    def isolateLayer(self, layer: int):
        """
        Get all voxels in a specified layer.

        ----

        Example:

        ``model2 = model1.isolateLayer(8)``

        ----

        :param layer: Voxel layer to isolate
        :return: VoxelModel
        """
        new_voxels = np.zeros_like(self.voxels, dtype=np.uint16)
        new_voxels[:, :, layer - self.coords[2]] = self.voxels[:, :, layer - self.coords[2]]
        return VoxelModel(new_voxels, self.materials, self.coords, self.resolution)

    def isolateComponent(self, component: int):
        """
        Isolate a component by its component label.

        Component labels must first be updated with getComponents.
        Unrecognized component labels will return an empty object.

        :param component: Component label to isolate
        :return: VoxelModel
        """
        mask = np.array(self.components == component, dtype=np.bool)
        new_voxels = np.multiply(self.voxels, mask)
        return VoxelModel(new_voxels, self.materials, self.coords, self.resolution)

    # Mask operations ##############################
    # Material defaults to the first material in the input model

    def getUnoccupied(self):
        """
        Get all voxels not occupied by the input model.

        This operation can also be applied using the invert operator (~).

        ----

        Examples:

        ``model2 = model1.getUnoccupied()``

        ``model2 = ~model1``

        ----

        :return: VoxelModel
        """
        mask = np.array(self.voxels == 0, dtype=np.bool)
        return VoxelModel(mask, self.materials[0:2, :], self.coords, self.resolution)

    def __invert__(self):
        """
        Get all voxels not occupied by the input model.

        Overload invert operator (~) for VoxelModel objects with getUnoccupied().

        :return: VoxelModel
        """
        return self.getUnoccupied()

    def getOccupied(self):
        """
        Get all voxels occupied by the input model.

        :return: VoxelModel
        """
        mask = np.array(self.voxels != 0, dtype=np.bool)
        return VoxelModel(mask, self.materials[0:2, :], self.coords, self.resolution)

    def getBoundingBox(self):
        """
        Get all voxels contained in the bounding box of the input model.

        :return: VoxelModel
        """
        new_model = VoxelModel.copy(self)
        new_model = new_model.fitWorkspace()
        new_model.voxels.fill(1)
        new_model = new_model.getOccupied()
        new_model.materials = self.materials[0:2, :]
        return new_model

    def setMaterial(self, material: int):
        """
        Set the material of all voxels in a model.

        ----

        Example:

        ``model2 = model1.getBoundingBox()``

        ``model3 = model2.setMaterial(2)``

        ----

        :param material: Material index corresponding to materials.py
        :return: VoxelModel
        """
        new_voxels = self.getOccupied().voxels # Converts input model to a mask, no effect if input is already a mask
        material_vector = np.zeros(len(material_properties)+1, dtype=np.float32)
        material_vector[0] = 1
        material_vector[material+1] = 1
        a = np.zeros(len(material_properties)+1, dtype=np.float32)
        b = material_vector
        m = np.vstack((a, b))
        return VoxelModel(new_voxels, m, self.coords, self.resolution)

    def setMaterialVector(self, material_vector):  # material input is the desired material vector
        """
        Set the material of all voxels in a model.

        ----

        Example:

        ``material_vector = np.zeros(len(materials) + 1)``

        ``material_vector[0] = 1 # Set a to 1``

        ``material_vector[3] = 0.3 # Set material 3 to 30%``

        ``material_vector[4] = 0.7 # Set material 4 to 70%``

        ``model2 = model1.setMaterialVector(material_vector)``

        ----

        :param material_vector: Material mixture vector, format: (a, m0, m1, ... mn)
        :return: VoxelMode
        """
        new_voxels = self.getOccupied().voxels  # Converts input model to a mask, no effect if input is already a mask
        a = np.zeros(len(material_properties)+1, dtype=np.float32)
        b = material_vector
        materials = np.vstack((a, b))
        return VoxelModel(new_voxels, materials, self.coords, self.resolution)

    # Boolean operations ##############################
    # Material from base model takes priority

    def union(self, model_to_add):
        """
        Find the geometric union of two models.

        The materials from self will take priority in overlapping areas
        of the resulting model. This operation can also be applied using
        the OR operator (|)

        ----

        Examples:

        ``model3 = model1.union(model2)``

        ``model3 = model1 | model2``

        ----

        :param model_to_add: VoxelModel to union with self
        :return: VoxelModel
        """
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
        """
        Find the geometric union of two models.

        Overload OR operator (|) for VoxelModel objects with union().

        :param other: VoxelModel to union with self
        :return: VoxelModel
        """
        return self.union(other)

    def difference(self, model_to_sub):
        """
        Find the geometric difference of two models.

        ----

        Example:

        ``model3 = model1.difference(model2)``

        ----

        :param model_to_sub: VoxelModel to subtract from self
        :return: VoxelModel
        """
        checkResolution(self, model_to_sub)
        a, b, new_coords = alignDims(self, model_to_sub)
        mask = np.array(b == 0, dtype=np.bool)
        new_voxels = np.multiply(a, mask)
        return VoxelModel(new_voxels, self.materials, new_coords, self.resolution)

    def intersection(self, model_2):
        """
        Find the geometric intersection of two models.

        The materials from self will be used in the resulting model.
        This operation can also be applied using the AND operator (&)

        ----

        Examples:

        ``model3 = model1.intersection(model2)``

        ``model3 = model1 & model2``

        ----

        :param model_2: VoxelModel to intersect with self
        :return: VoxelModel
        """
        checkResolution(self, model_2)
        a, b, new_coords = alignDims(self, model_2)
        mask = np.logical_and(np.array(a != 0, dtype=np.bool), np.array(b != 0, dtype=np.bool))

        # Paper provides for left/right intersection
        # For code simplicity, only a left intersection is provided here
        new_voxels = np.multiply(a, mask) # material from left model takes priority
        materials = self.materials

        return VoxelModel(new_voxels, materials, new_coords, self.resolution)

    def __and__(self, other):
        """
        Find the geometric intersection of two models.

        Overload AND operator (&) for VoxelModel objects with intersection().

        :param other: VoxelModel to intersect with self
        :return: VoxelModel
        """
        return self.intersection(other)

    def xor(self, model_2):
        """
        Find the geometric exclusive or of two models.

        This operation can also be applied using the XOR operator (^)

        ----

        Examples:

        ``model3 = model1.xor(model2)``

        ``model3 = model1 ^ model2``

        ----

        :param model_2: VoxelModel to xor with self
        :return: VoxelModel
        """
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
        """
        Find the geometric exclusive or of two models.

        Overload XOR operator (^) for VoxelModel objects with xor().

        :param other: VoxelModel to xor with self
        :return: VoxelModel
        """
        return self.xor(other)

    # Material is computed
    def add(self, model_to_add):
        """
        Find the material-wise addition of two models.

        The materials of the result are calculated by adding the material vectors for each voxel together.

        Example -- Adding a voxel containing material 1 and a voxel containing material 3:

        >> Voxel A = [1, 0, 1, 0, 0]\n
        >> Voxel B = [1, 0, 0, 0, 1]\n
        >> A + B = [1, 0, 1, 0, 1]\n
        >> Scale Result (see Cleanup Operations) → [1, 0, 0.5, 0, 0.5]\n

        This operation can also be applied using the addition operator (+).

        ----

        Examples:

        ``model3 = model1.add(model2)``

        ``model3 = model1 + model2``

        ----

        :param model_to_add: VoxelModel to add to self
        :return: VoxelModel
        """
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
        """
        Find the material-wise addition of two models.

        Overload addition operator (+) for VoxelModel objects with add().

        :param other: VoxelModel to add to self
        :return: VoxelModel
        """
        return self.add(other)

    # Material is computed
    def subtract(self, model_to_sub):
        """
        Find the material-wise difference of two models.

        The materials of the result are calculated for each voxel by subtracting the
        second material vector from the first.

        Example -- Subtracting a voxel containing material 3 from the result of the
        addition example:

        >> Voxel A = [1, 0, 0.5, 0, 0.5]\n
        >> Voxel B = [1, 0, 0, 0, 1]\n
        >> A - B = [1, 0, 0.5, 0, -0.5]\n
        >> Remove negatives (see Cleanup Operations) → [1, 0, 0.5, 0, 0]\n

        This operation can also be applied using the subtraction operator (-).

        ----

        Examples:

        ``model3 = model1.subtract(model2)``

        ``model3 = model1 - model2``

        ----

        :param model_to_sub: VoxelModel to subtract from self
        :return: VoxelModel
        """
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
        """
        Find the material-wise difference of two models.

        Overload subtraction operator (-) for VoxelModel objects with subtract().

        :param other: VoxelModel to subtract from self
        :return: VoxelModel
        """
        return self.subtract(other)

    def multiply(self, other):
        """
        Find the material-wise multiplication of two models.

        The materials of the result are calculated by multiplying the material vectors
        for each voxel. This function also supports multiplication by a scalar.

        This operation can also be applied using the multiplication operator (*).

        ----

        Examples:

        ``model3 = model1.multiply(model2)``

        ``model3 = model1 * model2``

        ``model5 = model4 * 3``

        ----

        :param other: VoxelModel to multiply with self
        :return: VoxelModel
        """
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
        """
        Find the material-wise multiplication of two models.

        Overload multiplication operator (*) for VoxelModel objects with multiply().

        :param other: VoxelModel to multiply with self
        :return: VoxelModel
        """
        return self.multiply(other)

    def divide(self, other):
        """
        Find the material-wise division of two models.

        The materials of the result are calculated for each voxel by dividing
        the first material vector by the second. This function also supports
        division by a scalar.

        This operation can also be applied using the division operator (/).

        ----

        Examples:

        ``model3 = model1.divide(model2)``

        ``model3 = model1 / model2``

        ``model5 = model4 / 3``

        ----

        :param other: VoxelModel to divide self by
        :return: VoxelModel
        """
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
        """
        Find the material-wise division of two models.

        Overload division operator (/) for VoxelModel objects with divide().

        :param other: VoxelModel to divide self by
        :return: VoxelModel
        """
        return self.divide(other)

    # Morphology Operations ##############################

    def dilate(self, radius: int = 1, plane: Axes = Axes.XYZ, structType: Struct = Struct.STANDARD, connectivity: int = 3): # TODO: Preserve overlapping materials?
        """
        Dilate a model along the specified axes.

        ----

        Examples:

        ``model2 = model1.dilate(3)``

        ``model4 = model3.dilate(1, Axes.XY, Struct.SPHERE, 2)``

        ----

        :param radius: Dilation radius in voxels
        :param plane: Dilation directions, set using Axes class
        :param structType: Shape of structuring element, set using Struct class
        :param connectivity: Connectivity of structuring element (1-3)
        :return: VoxelModel
        """
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

    def dilateBounded(self, radius: int = 1, plane: Axes = Axes.XYZ, structType: Struct = Struct.STANDARD, connectivity: int = 3):
        """
        Dilate a model along the specified axes without increasing the size of its bounding box.

        :param radius: Dilation radius in voxels
        :param plane: Dilation directions, set using Axes class
        :param structType: Shape of structuring element, set using Struct class
        :param connectivity: Connectivity of structuring element (1-3)
        :return: VoxelModel
        """
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

    def erode(self, radius: int = 1, plane: Axes = Axes.XYZ, structType: Struct = Struct.STANDARD, connectivity: int = 3):
        """
        Erode a model along the specified axes.

        ----

        Examples:

        ``model2 = model1.erode(5, connectivity=2)``

        ``model4 = model3.erode(2, Axes.X, Struct.SPHERE, 1)``

        ----

        :param radius: Erosion radius in voxels
        :param plane: Erosion directions, set using Axes class
        :param structType: Shape of structuring element, set using Struct class
        :param connectivity: Connectivity of structuring element (1-3)
        :return: VoxelModel
        """
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

    def closing(self, radius: int = 1, plane: Axes = Axes.XYZ, structType: Struct = Struct.STANDARD, connectivity: int = 3):
        """
        Apply a closing algorithm along the specified axes.

        This algorithm consists of dilation followed by erosion and will remove small holes.
        Depending on the structuring element used, this will apply a chamfer or fillet effect
        to inside corners.

        :param radius: Radius for dilation/erosion in voxels
        :param plane:  Dilation/erosion directions, set using Axes class
        :param structType: Shape of structuring element, set using Struct class
        :param connectivity: onnectivity of structuring element (1-3)
        :return: VoxelModel
        """
        if radius == 0:
            return VoxelModel.copy(self)
        else:
            return self.dilate(radius, plane, structType, connectivity).erode(radius, plane, structType, connectivity).fitWorkspace()

    def opening(self, radius: int = 1, plane: Axes = Axes.XYZ, structType: Struct = Struct.STANDARD, connectivity: int = 3):
        """
        Apply an opening algorithm along the specified axes.

        This algorithm consists of erosion followed by dilation and will remove small features.
        Depending on the structuring element used, this will apply a chamfer or fillet effect
        to outside corners.

        :param radius: Radius for dilation/erosion in voxels
        :param plane:  Dilation/erosion directions, set using Axes class
        :param structType: Shape of structuring element, set using Struct class
        :param connectivity: Connectivity of structuring element (1-3)
        :return: VoxelModel
        """
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

    # Material Interface Modification ##############################

    def blur(self, radius: float = 1):
        """
        Apply a Gaussian blur with the defined radius to the entire model.

        The blur radius corresponds to two times the standard deviation
        (2 * sigma) of the Gaussian distribution. The blurred effect is limited
        to voxels that were occupied by material in the input model.

        ----

        Example:

        ``model2 = model1.blur(2)``

        ___

        :param radius: Blur radius in voxels
        :return: VoxelModel
        """
        if radius == 0:
            return VoxelModel.copy(self)

        full_model = toFullMaterials(self.voxels, self.materials, len(material_properties)+1)

        for m in tqdm(range(len(material_properties)), desc='Blur - applying gaussian filter'):
            full_model[:, :, :, m+1] = ndimage.gaussian_filter(full_model[:, :, :, m+1], sigma=radius/2)

        mask = full_model[:, :, :, 0]
        mask = np.repeat(mask[..., None], len(material_properties)+1, axis=3)
        full_model = np.multiply(full_model, mask)

        return toIndexedMaterials(full_model, self, self.resolution)

    def blurRegion(self, radius: float, region):
        """
        Apply a Gaussian blur with the defined radius to voxels that intersect with the region model.

        The blur radius corresponds to two times the standard deviation
        (2 * sigma) of the Gaussian distribution. The blurred effect is limited
        to voxels that were occupied by material in the intersection result
        and the material of the region model is ignored.

        ----

        Example:

        ``model2 = model1.blurRegion(3, regionModel)``

        ___

        :param radius: Blur radius in voxels
        :param region: VoxelModel defining the target blur region
        :return: VoxelModel
        """
        new_model = self.blur(radius)
        new_model = new_model.intersection(region)
        new_model = new_model.union(self)
        return new_model

    def dither(self, radius=1, use_full=True, x_error=0.0, y_error=0.0, z_error=0.0, error_spread_threshold=0.8, blur=True):
        if radius == 0:
            return VoxelModel.copy(self)

        if blur:
            new_model = self.blur(radius)
            new_model = new_model.scaleValues()
        else:
            new_model = self.scaleValues()

        full_model = toFullMaterials(new_model.voxels, new_model.materials, len(material_properties) + 1)
        full_model = ditherOptimized(full_model, use_full, x_error, y_error, z_error, error_spread_threshold)

        return toIndexedMaterials(full_model, self, self.resolution)

    # Cleanup ##############################

    def removeNegatives(self):
        """
        Remove negative material values from a model (these have no physical meaning).

        ----

        Example:

        ``model2 = model1.removeNegatives()``

        ___

        :return: VoxelModel
        """
        new_model = VoxelModel.copy(self)
        new_model.materials[new_model.materials < 1e-10] = 0
        material_sums = np.sum(new_model.materials[:,1:], 1) # This and following update the a values
        material_sums[material_sums > 0] = 1
        new_model.materials[:, 0] = material_sums
        return new_model

    def scaleValues(self):
        """
        Scale nonzero material values to make all voxels contain 100% material while
        maintaining the ratio between materials.

        ----

        Example:

        ``model2 = model1.scaleValues()``

        ___

        :return: VoxelModel
        """
        new_model = self.removeNegatives()
        material_sums = np.sum(new_model.materials[:, 1:], 1)
        material_sums[material_sums == 0] = 1
        material_sums = np.repeat(material_sums[..., None], len(material_properties), axis=1)
        new_model.materials[:,1:] = np.divide(new_model.materials[:,1:], material_sums)
        return new_model

    def scaleNull(self):
        """
        Scale null material values to make all voxels contain 100% material.

        Voxels that contained less than 100% material will contain the same material percentages as
        before, but will have varying density. Voxels that contained greater than 100% material
        will be scaled using scaleValues().

        ----

        Example:

        ``model2 = model1.scaleNull()``

        ___

        :return: VoxelModel
        """
        new_model = self.removeNegatives()
        material_sums = np.sum(new_model.materials[:, 1:], 1)
        material_sums = np.ones(np.shape(material_sums)) - material_sums
        material_sums[material_sums < 0] = 0
        new_model.materials[:,1] = np.multiply(material_sums, new_model.materials[:,0])
        new_model = new_model.scaleValues()
        return new_model

    def round(self, toNearest: float = 0.1):
        """
        Round material percentages to nearest multiple of an input value.

        :param toNearest: Value to round to
        :return: VoxelModel
        """
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
        """
        Set all null material percentages to zero.

        :return: VoxelModel
        """
        new_model = VoxelModel.copy(self)
        new_model.materials[1:, 1] = np.zeros(np.shape(new_model.materials[1:,1]))
        return new_model

    def setDensity(self, density: float = 1.0):
        """
        Set the density of all voxels.

        :param density: Target density value
        :return: VoxelModel
        """
        new_model = self.clearNull()
        new_model = new_model.scaleValues()
        null_material_values = np.multiply(np.ones(np.shape(new_model.materials[1:,1])), 1-density)
        new_model.materials[1:, 1] = null_material_values
        new_model.materials[1:, 2:] = np.multiply(new_model.materials[1:, 2:], density)
        return new_model

    # Transformations ##############################

    def translate(self, vector: Tuple[int, int, int]):
        """
        Translate a model by the specified vector.

        :param vector: Translation vector in voxels
        :return: VoxelModel
        """

        new_model = VoxelModel.copy(self)
        new_model.coords = (self.coords[0]+vector[0], self.coords[1]+vector[1], self.coords[2]+vector[2])
        return new_model

    def translateMM(self, vector: Tuple[float, float, float]):
        """
        Translate a model by the specified vector.

        :param vector: Translation vector in mm
        :return: VoxelModel
        """

        xV = int(round(vector[0] * self.resolution))
        yV = int(round(vector[1] * self.resolution))
        zV = int(round(vector[2] * self.resolution))
        vector_voxels = (xV, yV, zV)
        new_model = self.translate(vector_voxels)
        return new_model

    def rotate(self, angle: float, axis: Axes = Axes.Z):
        """
        Rotate a model about its center.

        Floating point errors may slightly affect the angle of the resulting model.
        To rotate a model in precise 90 degree increments, use rotate90().

        :param angle: Angle of rotation in degrees
        :param axis: Axis of rotation, set using Axes class
        :return: VoxelModel
        """
        if axis == Axes.X:
            plane = (1, 2)
        elif axis == Axes.Y:
            plane = (0, 2)
        else: # axis == Axes.Z
            plane = (0, 1)

        centerCoords = self.getCenter()
        new_voxels = ndimage.rotate(self.voxels, angle, plane, order=0)
        new_model = VoxelModel(new_voxels, self.materials, self.coords, self.resolution)
        new_model = new_model.setCenter(centerCoords)

        return new_model

    def rotate90(self, times: int = 1, axis: Axes = Axes.Z):
        """
        Rotate a model about its center in increments of 90 degrees.

        :param times: Number of 90 degree increments to rotate model
        :param axis: Axis of rotation, set using Axes class
        :return: VoxelModel
        """
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

    def scale(self, factor: float, adjustResolution: bool = True):
        """
        Scale a model.

        If adjustResolution is enabled, the resolution attribute of the model will
        also be multiplied by the scaling factor.
        Enable adjustResolution if using this operation to change the resolution of a model.
        Disable adjustResolution if using this operation to change the size of a model.

        :param factor: Scale factor
        :param adjustResolution: Enable/disable automatic resolution adjustment
        :return: VoxelModel
        """
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

    def scaleToSize(self, size: Tuple[int, int, int]):
        """
        Scale a model to fit the given dimensions.

        :param size: Target dimensions in voxels
        :return: VoxelModel
        """
        model = self.fitWorkspace()

        new_voxels = np.zeros(size)
        for x in tqdm(range(size[0]), desc='Scaling'):
            for y in range(size[1]):
                for z in range(size[2]):
                    x_source = int(((x+1) / size[0]) * (model.voxels.shape[0]-1))
                    y_source = int(((y+1) / size[1]) * (model.voxels.shape[1]-1))
                    z_source = int(((z+1) / size[2]) * (model.voxels.shape[2]-1))
                    new_voxels[x,y,z] = model.voxels[x_source, y_source, z_source]

        model.voxels = new_voxels.astype(dtype=np.uint16)
        new_model = model.setCoords(model.coords)

        return new_model

    def getCenter(self):
        """
        Find the coordinates of the center of a model.

        :return: Center coordinates in voxels
        """
        model = self.fitWorkspace()

        x_center = (model.voxels.shape[0] / 2) + model.coords[0]
        y_center = (model.voxels.shape[1] / 2) + model.coords[1]
        z_center = (model.voxels.shape[2] / 2) + model.coords[2]

        centerCoords = (x_center, y_center, z_center)
        return centerCoords

    def setCenter(self, coords: Tuple[float, float, float]):
        """
        Set the center of a model to the specified coordinates.

        :param coords: Target coordinates in voxels
        :return: VoxelModel
        """
        new_model = self.fitWorkspace()

        x_new = int(round(coords[0] - (new_model.voxels.shape[0] / 2)))
        y_new = int(round(coords[1] - (new_model.voxels.shape[1] / 2)))
        z_new = int(round(coords[2] - (new_model.voxels.shape[2] / 2)))

        new_model.coords = (x_new, y_new, z_new)
        return new_model

    def setCoords(self, coords: Tuple[int, int, int]):
        """
        Set the origin of a model to the specified coordinates.

        :param coords: Target coordinates in voxels
        :return: VoxelModel
        """
        new_model = self.fitWorkspace()
        new_model.coords = coords
        return new_model

    # Model Info ##############################

    def getVolume(self, component: int = 0, material: int = 0):
        """
        Get the volume of a model or model component.

        :param component: Component label to measure, set to 0 for all components
        :param material: Material index to measure, set to 0 for all materials
        :return: Volume in voxels, volume in mm^3
        """
        new_model = VoxelModel.copy(self)
        if component > 0:
            new_model = new_model.isolateComponent(component)
        if material > 0:
            new_model = new_model.isolateMaterial(material)
        volumeVoxels = np.count_nonzero(new_model.voxels)
        volumeMM3 = volumeVoxels * ((1/self.resolution)**3)
        return volumeVoxels, volumeMM3

    def getMaterialProperties(self, material):
        """
        Get the average material properties of a row in a model's material array.

        :param material: Material index
        :return: Dictionary of material properties
        """
        avgProps = {}
        for key in material_properties[0]:
            if key == 'name' or key == 'process':
                string = ''
                for i in range(len(material_properties)):
                    if self.materials[material][i + 1] > 0:
                        string = string + material_properties[i][key] + ' '
                avgProps.update({key: string})
            elif key == 'MM' or key == 'FM':
                var = 0
                for i in range(len(material_properties)):
                    if self.materials[material][i + 1] > 0:
                        var = max(var, material_properties[i][key])
                avgProps.update({key: var})
            else:
                var = 0
                for i in range(len(material_properties)):
                    var = var + self.materials[material][i + 1] * material_properties[i][key]
                avgProps.update({key: var})
        return avgProps

    def getVoxelProperties(self, coords: Tuple[int, int, int]):
        """
        Get the average material properties of a specific voxel.

        :param coords: Voxel coordinates
        :return: Dictionary of material properties
        """
        return self.getMaterialProperties(self.voxels[coords[0], coords[1], coords[2]])

    # Manufacturing Features ##############################

    def projection(self, direction: Dir, material: int = 1):
        """
        Generate a model representing all voxels within the workspace that contain
        material or that lie in the specified direction with respect to a voxel
        that contains material.

        ---

        Example:

        ``modelResult = model1.projection(Dir.DOWN)``

        ---

        :param direction: Projection direction, set using Dir class
        :param material: Material index corresponding to materials.py
        :return: VoxelModel
        """
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

    def keepout(self, method: Process, material: int = 1):
        """
        Generate a model representing the keep-out region of a model.

        The keep-out region for a given process and part represents material which
        the process may not modify while creating the part. This feature primarily
        applies to subtractive processes. It includes material that will be present
        in the final part and regions of the workspace that cannot be accessed
        without affecting this material. In general, additive processes will have
        no keep-out region because they deposit material from the bottom up.

        ----

        Example:

        ``modelResult = model1.keepout(Process.MILL)``

        ----

        :param method: Target manufacturing method, set using Process class
        :param material: Material index corresponding to materials.py
        :return: VoxelModel
        """
        if method == Process.LASER:
            new_model = self.projection(Dir.BOTH, material)
        elif method == Process.MILL:
            new_model = self.projection(Dir.DOWN, material)
        elif method == Process.INSERT:
            new_model = self.projection(Dir.UP, material)
        else:
            new_model = self
        return new_model

    def clearance(self, method: Process, material: int = 1):
        """
        Generate a model representing the clearance region of a model.

        The clearance region for a given process and part represents regions that
        will be affected by the process acting on the part. Clearance can be
        used to identify regions of a model that conflict with the manufacturing
        of another model.

        ----

        Example:

        ``modelResult = model1.clearance(Process.PRINT)``

        ----

        :param method: Target manufacturing method, set using Process class
        :param material: Material index corresponding to materials.py
        :return: VoxelModel
        """
        if method == Process.LASER:
            new_model = self.projection(Dir.BOTH, material).difference(self)
        elif method == Process.MILL:
            new_model = self.projection(Dir.BOTH, material).difference(self.projection(Dir.DOWN, material))
        elif (method == Process.INSERT) or (method == Process.PRINT):
            new_model = self.projection(Dir.UP, material)
        else:
            new_model = self
        return new_model

    def support(self, method: Process, r1: int = 1, r2: int = 1, plane: Axes = Axes.XY, material: int = 1):
        """
        Generate a model representing where support material may be added to an
        object as characterized by the process that is used to remove the supports.

        ----

        Example:

        ``modelResult = model1.support(Process.LASER)``

        ----

        :param method: Target support removal method, set using Process class
        :param r1: Parameter used to determine areas where support is ineffective
                   based on proximity to empty regions that are inaccessible to the removal process
        :param r2: Desired thickness of the support material
        :param plane: Directions in which to add support material, set using Axes class
        :param material: Material index corresponding to materials.py
        :return: VoxelModel
        """
        model_A = self.keepout(method, material)
        model_A = model_A.dilate(r2, plane).difference(model_A)
        model_A = model_A.difference(self.keepout(method, material).difference(self).dilate(r1, plane)) # Valid support regions
        return model_A

    def userSupport(self, support_model, method: Process, r1: int = 1, r2: int = 1, plane: Axes = Axes.XY, material: int = -1):
        """
        Generate a model representing the intersection of the supportable region and a user support model.

        ----

        Example:

        ``modelResult = model1.userSupport(model2, Process.LASER)``

        ----

        :param support_model: User provided support model
        :param method: Target support removal method, set using Process class
        :param r1: Parameter used to determine areas where support is ineffective
                   based on proximity to empty regions that are inaccessible to the removal process
        :param r2: Desired thickness of the support material
        :param plane: Directions in which to add support material, set using Axes class
        :param material: Material index corresponding to materials.py, set to -1 to use support model material
        :return: VoxelModel
        """
        if material > -1:
            model_A = self.support(method, r1, r2, plane)
            model_A = support_model.intersection(model_A)
        else:
            model_A = self.support(method, r1, r2, plane, material)
            model_A = model_A.intersection(support_model)
        return model_A

    def web(self, method: Process, r1: int = 1, r2: int = 1, layer: int = -1, material = 1):
        """
        Generate a model representing the scrap material surrounding a model.

        Web can be used in the creation of supports or layer alignment fixtures.

        ----

        Example:

        ``modelResult = model1.web(Process.LASER, 1, 5)``

        ----

        :param method: Target web removal method, set using Process class
        :param r1: Distance from surface of part to inside of web in voxels
        :param r2: Width of web in voxels
        :param layer: Voxel layer at which to generate web, set to -1 to generate for all layers
        :param material: Material index corresponding to materials.py
        :return: VoxelModel
        """
        model_A = self.keepout(method, material)
        if layer != -1:
            model_A = model_A.isolateLayer(layer)
        model_A = model_A.dilate(r1, Axes.XY)
        model_A = model_A.dilate(r2, Axes.XY).getBoundingBox().difference(model_A)
        return model_A

    # File IO ##############################

    def saveVF(self, filename: str):
        """
        Save model data to a .vf file

        The native VoxelFuse file format stores the same information as the attributes of
        a VoxelModel object. This includes geometry and material mixture data. Material
        attributes remain stored in the materials.py file, so the same version of
        this file must be used when saving and opening models. The .vf file type can be reopened
        by a VoxelFuse script.

        ----

        Example:

        ``modelResult.saveVF("test-file")``

        ----

        :param filename: File name
        :return: None
        """
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
    def openVF(cls, filename: str):
        """
        Load model data from a .vf file

        This method will create a new VoxelModel object using the data from the .vf file.
        Material attributes are stored in the materials.py file, so the same version of
        this file must be used when saving and opening models.

        ----

        Example:

        ``model1.openVF("test-file")``

        ----

        :param filename: File name
        :return: VoxelModel
        """
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

    def saveVXC(self, filename: str, compression: bool = False):
        """
        Save model data to a .vxc file

        The VoxCad file format stores geometry and full material palette data. The material
        palette includes the properties for each material and material mixtures are
        converted into distinct palette entries.

        This format supports compression for the voxel data. Enabling compression allows
        for larger models, but it may introduce geometry errors that particularly affect
        small models.

        The .vxc file type can be opened using VoxCad simulation software. However, it
        cannot currently be reopened by a VoxelFuse script.

        ----

        Example:

        ``modelResult.saveVXC("test-file", compression=False)``

        ----

        :param filename: File name
        :param compression: Enable/disable voxel data compression
        :return: None
        """
        f = open(filename + '.vxc', 'w+')
        print('Saving file: ' + f.name)

        empty_model = VoxelModel.empty((1,1,1), self.resolution)
        export_model = (VoxelModel.copy(self).fitWorkspace()) | empty_model  # Fit workspace and union with an empty object at the origin to clear offsets if object is raised
        export_model.coords = (0, 0, 0)  # Set coords to zero to move object to origin if it is at negative coordinates

        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        export_model.writeVXCData(f, compression)

        f.close()

    def writeVXCData(self, f: TextIO, compression: bool = False):
        """
        Write geometry and material data to a text file using the .vxc format.

        :param f: File to write to
        :param compression:  Enable/disable voxel data compression
        :return: None
        """
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
            avgProps = self.getMaterialProperties(row)

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

# Helper functions ##############################################################
def makeMesh(filename: str, delete_files: bool = True, gmsh_on_path: bool = False):
    """
    Import mesh data from file

    :param filename: File name with extension
    :param delete_files: Enable/disable deleting temporary files when finished
    :param gmsh_on_path: Enable/disable using system gmsh rather than bundled gmsh
    :return: Mesh data (points, tris, and tets)
    """
    template = '''
    Merge "{0}";
    Surface Loop(1) = {{1}};
    //+
    Volume(1) = {{1}};
    '''

    geo_string = template.format(filename)
    with open('output.geo', 'w') as f:
        f.writelines(geo_string)

    if gmsh_on_path:
        command_string = 'gmsh '
    else:
        # Check OS type
        if os.name.startswith('nt'):
            # Windows
            command_string = f'"{os.path.dirname(os.path.realpath(__file__))}\\utils\\gmsh.exe"'
        else:
            # Linux
            command_string = f'"{os.path.dirname(os.path.realpath(__file__))}/utils/gmsh"'

    command_string = command_string + ' output.geo -3 -format msh'

    print('Launching gmsh using: ' + command_string)
    p = subprocess.Popen(command_string, shell=True)
    p.wait()

    mesh_file = 'output.msh'
    data = meshio.read(mesh_file)

    if delete_files:
        os.remove('output.msh')
        os.remove('output.geo')
    return data

def alignDims(modelA, modelB):
    """
    Make object dimensions compatible for solid body operations.

    This function accounts for location coordinates.

    :param modelA: Input model A
    :param modelB: Input model B
    :return: Resized model A, resized model B, New model coordinates
    """
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
    """
    Check if model resolutions are compatible for solid geometry operations.

    Incompatible resolutions will print an error. This will not prevent the
    operation from running, but it may indicate that the models are at
    different scales or that a resolution value has been incorrectly set.

    :param modelA: Input model A
    :param modelB: Input model B
    :return: Check successful T/F
    """
    a = modelA.resolution
    b = modelB.resolution
    if a != b:
        print('WARNING: inconsistent resolutions: ' + str(a) + ', ' + str(b))
        return False
    else:
        return True

"""
Functions to generate structuring elements
"""
def structSphere(radius: int, plane: Axes):
    """
    Generate a spherical structuring element.

    :param radius: Radius of structuring element in voxels
    :param plane: Structuring element directions, set using Axes class
    :return: Structuring element array
    """
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

def structStandard(connectivity: int, plane: Axes):
    """
    Generate a 3x3x3 structuring element with the specified connectivity.

    Outer face of structuring element illustrated for connectivity values 1-3:

    0,0,0 | 0,1,0 | 1,1,1\n
    0,1,0 | 1,1,1 | 1,1,1\n
    0,0,0 | 0,1,0 | 1,1,1

    :param connectivity: Connectivity of structuring element (1-3)
    :param plane: Structuring element directions, set using Axes class
    :return: Structuring element array
    """
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
    """
    Generate the materials table for a single-material VoxelModel.

    :param m: Material index corresponding to materials.py
    :return: Array containing the specified material and the empty material
    """
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
    """
    Convert from index-based material mixture storage to storing
    full material mixtures at every voxel.

    This representation requires much more memory, but is
    needed for some operations. Also see toIndexedMaterials().

    :param voxels: VoxelModel.voxels
    :param materials: VoxelModel.materials
    :param n_materials: Number of materials in the material properties table
    :return: Model data array
    """
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
    """
    Convert from storing full material mixtures at every voxel
    to index-based material mixture storage.

    Also see toFullMaterials().

    :param voxels: Model data array
    :param model: Reference VoxelModel for size and coords
    :param resolution:
    :return:
    """
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

@njit()
def addError(model, error, constant, i, x, y, z, x_len, y_len, z_len, error_spread_threshold):
    if y < y_len and x < x_len and z < z_len:
        high = np.where(model[x, y, z, 1:] > error_spread_threshold)[0]
        if len(high) == 0:
            model[x, y, z, i] += error * constant * model[x, y, z, 0]

@njit()
def ditherOptimized(full_model, use_full, x_error, y_error, z_error, error_spread_threshold):
    x_len = full_model.shape[0]
    y_len = full_model.shape[1]
    z_len = full_model.shape[2]

    for z in range(z_len):
        for y in range(y_len):
            for x in range(x_len):
                voxel = full_model[x, y, z]
                if voxel[0] == 1.0:
                    max_i = voxel[1:].argmax()+1
                    for i in range(1, len(voxel)):
                        if full_model[x, y, z, i] != 0:
                            old = full_model[x, y, z, i]

                            if i == max_i:
                                full_model[x, y, z, i] = 1
                            else:
                                full_model[x, y, z, i] = 0

                            error = old - full_model[x, y, z, i]

                            if use_full:
                                # Based on Fundamentals of 3D Halftoning by Lou and Stucki
                                addError(full_model, error, 4/21, i, x+1, y, z, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, 1/21, i, x+2, y, z, x_len, y_len, z_len, error_spread_threshold)

                                addError(full_model, error, 4/21, i, x, y+1, z, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, 1/21, i, x, y+2, z, x_len, y_len, z_len, error_spread_threshold)

                                addError(full_model, error, 1/21, i, x+1, y+1, z, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, 1/21, i, x-1, y+1, z, x_len, y_len, z_len, error_spread_threshold)

                                addError(full_model, error, 1/21, i, x, y-1, z+1, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, 1/21, i, x-1, y, z+1, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, 1/21, i, x, y+1, z+1, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, 1/21, i, x+1, y, z+1, x_len, y_len, z_len, error_spread_threshold)

                                addError(full_model, error, 4/21, i, x, y, z+1, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, 1/21, i, x, y, z+2, x_len, y_len, z_len, error_spread_threshold)
                            else:
                                addError(full_model, error, x_error, i, x+1, y, z, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, y_error, i, x, y+1, z, x_len, y_len, z_len, error_spread_threshold)
                                addError(full_model, error, z_error, i, x, y, z+1, x_len, y_len, z_len, error_spread_threshold)

    return full_model