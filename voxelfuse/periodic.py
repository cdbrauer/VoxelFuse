"""
Functions for generating triply periodic structures.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import math
import numpy as np
from typing import Tuple
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.materials import material_properties
from tqdm import tqdm

def gyroid(size: Tuple[int, int, int] = (15, 15, 15), scale: int = 15, coords: Tuple[int, int, int] = (0, 0, 0), material1: int = 1, material2: int = 2, resolution: float = 1):
    """
    Generate a Gyroid pattern over a rectangular region.

    This function will generate two models representing the "positive" and
    "negative" halves of the pattern. If a thin surface is desired,
    these two models can be dilated and the intersection of the results found.
    However, due to the voxel-based representation, this "surface" will have
    a non-zero thickness.

    :param size: Size of rectangular region
    :param scale: Period of the surface function in voxels
    :param coords: Model origin coordinates
    :param material1: Material index for negative model, corresponds to materials.py
    :param material2: Material index for positive model, corresponds to materials.py
    :param resolution: Number of voxels per mm
    :return: Negative model, Positive model
    """
    s = (2 * math.pi) / scale  # scaling multipler

    modelData = np.zeros((size[0], size[1], size[2]), dtype=np.uint16)
    surface_model_inner = VoxelModel(modelData, generateMaterials(material1), coords=coords, resolution=resolution)
    surface_model_outer = VoxelModel(modelData, generateMaterials(material2), coords=coords, resolution=resolution)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = math.sin(x*s) * math.cos(y*s)
                t_calc = math.sin(y*s) * math.cos(z*s) + t_calc
                t_calc = math.sin(z*s) * math.cos(x*s) + t_calc

                if t_calc < 0:
                    surface_model_inner.voxels[x, y, z] = 1
                else:
                    surface_model_outer.voxels[x, y, z] = 1

    return surface_model_inner, surface_model_outer

def schwarzP(size: Tuple[int, int, int] = (15, 15, 15), scale: int = 15, coords: Tuple[int, int, int] = (0, 0, 0), material1: int = 1, material2: int = 2, resolution: float = 1):
    """
    Generate a Schwarz P-surface over a rectangular region.

    This function will generate two models representing the "positive" and
    "negative" halves of the pattern. If a thin surface is desired,
    these two models can be dilated and the intersection of the results found.
    However, due to the voxel-based representation, this "surface" will have
    a non-zero thickness.

    :param size: Size of rectangular region
    :param scale: Period of the surface function in voxels
    :param coords: Model origin coordinates
    :param material1: Material index for negative model, corresponds to materials.py
    :param material2: Material index for positive model, corresponds to materials.py
    :param resolution: Number of voxels per mm
    :return: Negative model, Positive model
    """
    s = (2 * math.pi) / scale  # scaling multipler

    modelData = np.zeros((size[0], size[1], size[2]), dtype=np.uint16)
    surface_model_inner = VoxelModel(modelData, generateMaterials(material1), coords=coords, resolution=resolution)
    surface_model_outer = VoxelModel(modelData, generateMaterials(material2), coords=coords, resolution=resolution)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = math.cos(x*s)
                t_calc = math.cos(y*s) + t_calc
                t_calc = math.cos(z*s) + t_calc

                if t_calc < 0:
                    surface_model_inner.voxels[x, y, z] = 1
                else:
                    surface_model_outer.voxels[x, y, z] = 1

    return surface_model_inner, surface_model_outer

def schwarzD(size: Tuple[int, int, int] = (15, 15, 15), scale: int = 15, coords: Tuple[int, int, int] = (0, 0, 0), material1: int = 1, material2: int = 2, resolution: float = 1):
    """
    Generate a Schwarz D-surface over a rectangular region.

    This function will generate two models representing the "positive" and
    "negative" halves of the pattern. If a thin surface is desired,
    these two models can be dilated and the intersection of the results found.
    However, due to the voxel-based representation, this "surface" will have
    a non-zero thickness.

    :param size: Size of rectangular region
    :param scale: Period of the surface function in voxels
    :param coords: Model origin coordinates
    :param material1: Material index for negative model, corresponds to materials.py
    :param material2: Material index for positive model, corresponds to materials.py
    :param resolution: Number of voxels per mm
    :return: Negative model, Positive model
    """
    s = (2 * math.pi) / scale # scaling multipler

    modelData = np.zeros((size[0], size[1], size[2]), dtype=np.uint16)
    surface_model_inner = VoxelModel(modelData, generateMaterials(material1), coords=coords, resolution=resolution)
    surface_model_outer = VoxelModel(modelData, generateMaterials(material2), coords=coords, resolution=resolution)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = math.sin(x*s) * math.sin(y*s) * math.sin(z*s)
                t_calc = math.sin(x*s) * math.cos(y*s) * math.cos(z*s) + t_calc
                t_calc = math.cos(x*s) * math.sin(y*s) * math.cos(z*s) + t_calc
                t_calc = math.cos(x*s) * math.cos(y*s) * math.sin(z*s) + t_calc

                if t_calc < 0:
                    surface_model_inner.voxels[x, y, z] = 1
                else:
                    surface_model_outer.voxels[x, y, z] = 1

    return surface_model_inner, surface_model_outer

def FRD(size: Tuple[int, int, int] = (15, 15, 15), scale: int = 15, coords: Tuple[int, int, int] = (0, 0, 0), material1: int = 1, material2: int = 2, resolution: float = 1):
    """
    Generate a FRD surface over a rectangular region.

    This function will generate two models representing the "positive" and
    "negative" halves of the pattern. If a thin surface is desired,
    these two models can be dilated and the intersection of the results found.
    However, due to the voxel-based representation, this "surface" will have
    a non-zero thickness.

    :param size: Size of rectangular region
    :param scale: Period of the surface function in voxels
    :param coords: Model origin coordinates
    :param material1: Material index for negative model, corresponds to materials.py
    :param material2: Material index for positive model, corresponds to materials.py
    :param resolution: Number of voxels per mm
    :return: Negative model, Positive model
    """
    s = (2 * math.pi) / scale # scaling multipler

    modelData = np.zeros((size[0], size[1], size[2]), dtype=np.uint16)
    surface_model_inner = VoxelModel(modelData, generateMaterials(material1), coords=coords, resolution=resolution)
    surface_model_outer = VoxelModel(modelData, generateMaterials(material2), coords=coords, resolution=resolution)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = 4*(math.cos(x*s) * math.cos(y*s) * math.cos(z*s))
                t_calc = -(math.cos(2*x*s) * math.cos(2*y*s)) + t_calc
                t_calc = -(math.cos(2*y*s) * math.cos(2*z*s)) + t_calc
                t_calc = -(math.cos(2*x*s) * math.cos(2*z*s)) + t_calc

                if t_calc < 0:
                    surface_model_inner.voxels[x, y, z] = 1
                else:
                    surface_model_outer.voxels[x, y, z] = 1

    return surface_model_inner, surface_model_outer

# Helper functions ##############################################################
def generateMaterials(m: int):
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
