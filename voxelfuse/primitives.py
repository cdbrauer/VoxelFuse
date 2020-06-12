"""
Functions for generating primitive solids.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import numpy as np
from typing import Tuple
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.materials import material_properties

# Basic primitives
def empty(coords: Tuple[int, int, int] = (0, 0, 0), resolution: float = 1):
    """
    Create a VoxelModel containing a single empty voxel at the specified coordinates.

    :param coords: Model origin coordinates
    :param resolution: Number of voxels per mm
    :return: VoxelModel
    """
    model_data = np.zeros((1, 1, 1), dtype=np.uint16)
    materials = np.zeros((1, len(material_properties) + 1), dtype=np.float)
    model = VoxelModel(model_data, materials, coords=coords, resolution=resolution)
    return model

def cube(size: int = 1, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cube.

    :param size: Length of cube side in voxels
    :param coords: Model origin coordinates
    :param material: Material index corresponding to materials.py
    :param resolution: Number of voxels per mm
    :return: VoxelModel
    """
    model_data = np.ones((size, size, size), dtype=np.uint16)
    model = VoxelModel(model_data, generateMaterials(material), coords=coords, resolution=resolution)
    return model

def cuboid(size: Tuple[int, int, int] = (1, 1, 1), coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cuboid.

    :param size: Lengths of cuboid sides in voxels
    :param coords: Model origin coordinates
    :param material: Material index corresponding to materials.py
    :param resolution: Number of voxels per mm
    :return: VoxelModel
    """
    model_data = np.ones((size[0], size[1], size[2]), dtype=np.uint16)
    model = VoxelModel(model_data, generateMaterials(material), coords=coords, resolution=resolution)
    return model

def sphere(radius: int = 1, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a sphere.

    :param radius: Radius of sphere in voxels
    :param coords: Model origin coordinates
    :param material: Material index corresponding to materials.py
    :param resolution: Number of voxels per mm
    :return: VoxelModel
    """
    diameter = (radius*2) + 1
    model_data = np.zeros((diameter, diameter, diameter), dtype=np.uint16)

    for x in range(diameter):
        for y in range(diameter):
            for z in range(diameter):
                xd = (x - radius)
                yd = (y - radius)
                zd = (z - radius)
                r = np.sqrt(xd**2 + yd**2 + zd**2)

                if r < (radius + .5):
                    model_data[x, y, z] = 1

    model = VoxelModel(model_data, generateMaterials(material), coords=(coords[0]-radius, coords[1]-radius, coords[2]-radius), resolution=resolution)
    return model

def cylinder(radius: int = 1, height: int = 1, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cylinder.

    :param radius: Radius of cylinder in voxels
    :param height: Height of cylinder in voxels
    :param coords: Model origin coordinates
    :param material: Material index corresponding to materials.py
    :param resolution: Number of voxels per mm
    :return: VoxelModel
    """
    diameter = (radius * 2) + 1
    model_data = np.zeros((diameter, diameter, 1), dtype=np.uint16)

    for x in range(diameter):
        for y in range(diameter):
                xd = (x - radius)
                yd = (y - radius)
                r = np.sqrt(xd ** 2 + yd ** 2)

                if r < (radius + .5):
                    model_data[x, y, 0] = 1

    model_data = np.repeat(model_data, height, 2)

    model = VoxelModel(model_data, generateMaterials(material), coords=(coords[0]-radius, coords[1]-radius, coords[2]), resolution=resolution)
    return model

def cone(min_radius: int = 0, max_radius: int = 4, height: int = 5, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cylinder.

    :param min_radius: Point radius of cone in voxels
    :param max_radius: Base radius of cone in voxels
    :param height: Height of cone in voxels
    :param coords: Model origin coordinates
    :param material: Material index corresponding to materials.py
    :param resolution: Number of voxels per mm
    :return: VoxelModel
    """
    max_diameter = (max_radius*2)+1
    model_data = np.zeros((max_diameter, max_diameter, height), dtype=np.uint16)

    for z in range(height):
        radius = (abs(max_radius - min_radius) * (((height-1) - z)/(height-1))) + min_radius

        for x in range(max_diameter):
            for y in range(max_diameter):
                xd = (x - max_radius)
                yd = (y - max_radius)
                r = np.sqrt(xd ** 2 + yd ** 2)

                if r < (radius + .5):
                    model_data[x, y, z] = 1

    model = VoxelModel(model_data, generateMaterials(material), coords=(coords[0]-max_radius, coords[1]-max_radius, coords[2]), resolution=resolution)
    return model

def pyramid(min_radius: int = 0, max_radius: int = 4, height: int = 5, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cylinder.

    :param min_radius: Point radius of pyramid in voxels
    :param max_radius: Base radius of pyramid in voxels
    :param height: Height of pyramid in voxels
    :param coords: Model origin coordinates
    :param material: Material index corresponding to materials.py
    :param resolution: Number of voxels per mm
    :return: VoxelModel
    """
    max_diameter = (max_radius * 2) + 1
    model_data = np.zeros((max_diameter, max_diameter, height), dtype=np.uint16)

    for z in range(height):
        radius = round((abs(max_radius - min_radius) * (z / (height - 1))))

        if radius == 0:
            model_data[:, :, z].fill(1)
        else:
            model_data[radius:-radius, radius:-radius, z].fill(1)

    model = VoxelModel(model_data, generateMaterials(material), coords=(coords[0]-max_radius, coords[1]-max_radius, coords[2]), resolution=resolution)
    return model

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
