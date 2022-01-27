"""
Functions for generating primitive solids.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import numpy as np
from typing import Tuple

from voxelfuse.voxel_model import VoxelModel, generateMaterials
from voxelfuse.materials import material_properties

# Basic primitives
def empty(coords: Tuple[int, int, int] = (0, 0, 0), resolution: float = 1, num_materials: int = len(material_properties)):
    """
    Create a VoxelModel containing a single empty voxel at the specified coordinates.

    Args:
        coords: Model origin coordinates
        resolution: Number of voxels per mm
        num_materials: Number of material types in materials vector
    
    Returns:
        VoxelModel
    """
    return VoxelModel.empty((1, 1, 1), coords, resolution, num_materials)

def cube(size: int, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cube.

    Args:
        size: Length of cube side in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm
    
    Returns:
        VoxelModel
    """
    model_data = np.ones((size, size, size), dtype=np.uint16)
    model = VoxelModel(model_data, generateMaterials(material), coords=coords, resolution=resolution)
    return model

def cuboid(size: Tuple[int, int, int], coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cuboid.

    Args:
        size: Lengths of cuboid sides in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm
    
    Returns:
        VoxelModel
    """
    model_data = np.ones((size[0], size[1], size[2]), dtype=np.uint16)
    model = VoxelModel(model_data, generateMaterials(material), coords=coords, resolution=resolution)
    return model

def sphere(radius: int, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a sphere.

    Args:
        radius: Radius of sphere in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm
    
    Returns:
        VoxelModel
    """
    diameter = (radius*2) + 1 # +1 ensures there is a voxel centered on the origin
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

def ellipsoid(size: Tuple[int, int, int], coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing an ellipsoid.

    Args:
        size: Lengths of ellipsoid principal semi-axes in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm

    Returns:
        VoxelModel
    """
    # lengths of principal semi-axes
    a, b, c = size

    # bounding box size (+1 ensures there is a voxel centered on the origin)
    dx = (a*2) + 1
    dy = (b*2) + 1
    dz = (c*2) + 1
    model_data = np.zeros((dx, dy, dz), dtype=np.uint16)

    # check each voxel in the bounding box and determine if it falls inside the ellipsoid
    for x in range(dx):
        for y in range(dy):
            for z in range(dz):
                # get coords relative to origin
                xx = (x - a)
                yy = (y - b)
                zz = (z - c)

                # apply ellipsoid formula
                f = np.sqrt(((xx ** 2) / (a **2)) +
                            ((yy ** 2) / (b **2)) +
                            ((zz ** 2) / (c **2)))
                if f < 1:
                    model_data[x, y, z] = 1

    model = VoxelModel(model_data, generateMaterials(material), coords=(coords[0] - a, coords[1] - b, coords[2] - c), resolution=resolution)
    return model

def cylinder(radius: int, height: int, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cylinder.

    Args:
        radius: Radius of cylinder in voxels
        height: Height of cylinder in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm
    
    Returns:
        VoxelModel
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

def cone(min_radius: int, max_radius: int, height: int, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cylinder.

    Args:
        min_radius: Point radius of cone in voxels
        max_radius: Base radius of cone in voxels
        height: Height of cone in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm
    
    Returns:
        VoxelModel
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

def pyramid(min_radius: int, max_radius: int, height: int, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a cylinder.

    Args:
        min_radius: Point radius of pyramid in voxels
        max_radius: Base radius of pyramid in voxels
        height: Height of pyramid in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm
    
    Returns:
        VoxelModel
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

def prism(size: Tuple[int, int, int], point_offset: int = 0, coords: Tuple[int, int, int] = (0, 0, 0), material: int = 1, resolution: float = 1):
    """
    Create a VoxelModel containing a prism.

    Args:
        size: Size of prism in voxels (base,
        point_offset: Distance that prism point is offset from model center in voxels
        coords: Model origin coordinates
        material: Material index corresponding to materials.py
        resolution: Number of voxels per mm

    Returns:
        VoxelModel
    """
    point_pos = (size[0]/2) + point_offset
    min_x = min(0, round(point_pos))
    max_x = max(size[0], round(point_pos))
    dx = max_x - min_x
    model_data = np.zeros((dx, size[1], size[2]), dtype=np.uint16)

    for z in range(size[2]):
        width = (size[0]/size[2])*(size[2] - z)
        side_1_index = round((point_pos/size[2])*z) - min_x
        side_2_index = round((point_pos/size[2])*z + width) - min_x

        model_data[side_1_index:side_2_index, :, z].fill(1)

    model = VoxelModel(model_data, generateMaterials(material), coords=(coords[0] + min_x, coords[1], coords[2]), resolution=resolution)
    return model