"""
Copyright 2018-2019
Dan Aukes, Cole Brauer

Functions for generating triply periodic structures
"""

import math
import numpy as np
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.materials import material_properties
from tqdm import tqdm

def gyroid(size = (15, 15, 15), scale = 15, coords = (0, 0, 0), material = 1):
    #a - Scale (voxels/unit)
    #size - Volume

    s = (2 * math.pi) / scale  # scaling multipler

    model_data = np.zeros((size[0], size[1], size[2]), dtype=np.int32)
    surface_model = VoxelModel(model_data, generateMaterials(material), coords)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = math.sin(x*s) * math.cos(y*s)
                t_calc = math.sin(y*s) * math.cos(z*s) + t_calc
                t_calc = math.sin(z*s) * math.cos(x*s) + t_calc

                if t_calc < 0:
                    surface_model.voxels[x, y, z] = 1

    return surface_model

def schwarzP(size = (15, 15, 15), scale = 15, coords = (0, 0, 0), material = 1):
    #a - Scale (voxels/unit)
    #size - Volume

    s = (2 * math.pi) / scale  # scaling multipler

    modelData = np.zeros((size[0], size[1], size[2]), dtype=np.int32)
    surface_model = VoxelModel(modelData, generateMaterials(material), coords)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = math.cos(x*s)
                t_calc = math.cos(y*s) + t_calc
                t_calc = math.cos(z*s) + t_calc

                if t_calc < 0:
                    surface_model.voxels[x, y, z] = 1

    return surface_model

def schwarzD(size = (15, 15, 15), scale = 15, coords = (0, 0, 0), material = 1):
    #a - Scale (voxels/unit)
    #size - Volume

    s = (2 * math.pi) / scale # scaling multipler

    modelData = np.zeros((size[0], size[1], size[2]), dtype=np.int32)
    surface_model = VoxelModel(modelData, generateMaterials(material), coords)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = math.sin(x*s) * math.sin(y*s) * math.sin(z*s)
                t_calc = math.sin(x*s) * math.cos(y*s) * math.cos(z*s) + t_calc
                t_calc = math.cos(x*s) * math.sin(y*s) * math.cos(z*s) + t_calc
                t_calc = math.cos(x*s) * math.cos(y*s) * math.sin(z*s) + t_calc

                if t_calc < 0:
                    surface_model.voxels[x, y, z] = 1

    return surface_model

def FRD(size = (15, 15, 15), scale = 15, coords = (0, 0, 0), material = 1):
    #a - Scale (voxels/unit)
    #size - Volume

    s = (2 * math.pi) / scale # scaling multipler

    modelData = np.zeros((size[0], size[1], size[2]), dtype=np.int32)
    surface_model = VoxelModel(modelData, generateMaterials(material), coords)

    for x in tqdm(range(size[0]), desc='Generating Gyroid'):
        for y in range(size[1]):
            for z in range(size[2]):
                t_calc = 4*(math.cos(x*s) * math.cos(y*s) * math.cos(z*s))
                t_calc = -(math.cos(2*x*s) * math.cos(2*y*s)) + t_calc
                t_calc = -(math.cos(2*y*s) * math.cos(2*z*s)) + t_calc
                t_calc = -(math.cos(2*x*s) * math.cos(2*z*s)) + t_calc

                if t_calc < 0:
                    surface_model.voxels[x, y, z] = 1

    return surface_model


# Helper functions

def generateMaterials(m):
    materials = np.zeros(len(material_properties) + 1, dtype=np.float)
    material_vector = np.zeros(len(material_properties) + 1, dtype=np.float)
    material_vector[0] = 1
    material_vector[m+1] = 1
    materials = np.vstack((materials, material_vector))
    return materials
