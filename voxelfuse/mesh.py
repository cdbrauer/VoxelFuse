"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
import meshio
from numba import njit
from tqdm import tqdm

from voxelfuse.materials import material_properties

"""
Mesh Class

Initialized from a voxel model

Properties:
  verts - coordinates of surface vertices
  colors - color associated with each vertex
  tris - sets of vertices associated with triangular faces
"""
class Mesh:
    def __init__(self, input_model, verts, verts_colors, tris):
        self.verts = verts
        self.colors = verts_colors
        self.tris = tris
        self.model = input_model

    # Create mesh from voxel data
    @classmethod
    def fromVoxelModel(cls, voxel_model):
        voxel_model_array = voxel_model.voxels.astype(np.uint16)

        # Find exterior voxels
        exterior_voxels_array = voxel_model.difference(voxel_model.erode(radius=1, connectivity=1)).voxels
        
        x_len = len(voxel_model_array[:, 0, 0])
        y_len = len(voxel_model_array[0, :, 0])
        z_len = len(voxel_model_array[0, 0, :])
        
        # Create list of exterior voxel coordinates
        exterior_voxels_coords = []
        for x in tqdm(range(x_len), desc='Finding exterior voxels'):
            for y in range(y_len):
                for z in range(z_len):
                    if exterior_voxels_array[x, y, z] != 0:
                        exterior_voxels_coords.append([x, y, z])

        # Get voxel array
        voxel_model_array[voxel_model_array < 0] = 0

        model_materials = voxel_model.materials

        # Initialize arrays
        verts = []
        verts_colors = []
        tris = []
        vi = 1  # Tracks current vertex index

        # Loop through voxel_model_array data
        for voxel_coords in tqdm(exterior_voxels_coords, desc='Meshing'):
            x = voxel_coords[0]
            y = voxel_coords[1]
            z = voxel_coords[2]

            r = 0
            g = 0
            b = 0

            for i in range(len(material_properties)):
                r = r + model_materials[voxel_model_array[x, y, z]][i+1] * material_properties[i]['r']
                g = g + model_materials[voxel_model_array[x, y, z]][i+1] * material_properties[i]['g']
                b = b + model_materials[voxel_model_array[x, y, z]][i+1] * material_properties[i]['b']

            r = 1 if r > 1 else r
            g = 1 if g > 1 else g
            b = 1 if b > 1 else b

            a = 1 - model_materials[voxel_model_array[x, y, z]][1]

            voxel_color = [r, g, b, a]

            # Add cube vertices
            new_verts, verts_indices, new_tris, vi = addVerticesAndTriangles(voxel_model_array, x, y, z, vi)
            verts += new_verts
            tris += new_tris

            # Apply color to all vertices
            for i in range(0, np.count_nonzero(verts_indices)):
                verts_colors.append(voxel_color)

        verts = np.array(verts)
        verts_colors = np.array(verts_colors)
        tris = np.array(tris)

        return cls(voxel_model_array, verts, verts_colors, tris)

    # Export model from mesh data
    def export(self, filename):
        cells = {
            "triangle": self.tris
        }

        output_mesh = meshio.Mesh(self.verts, cells)
        meshio.write(filename, output_mesh)

@njit()
def check_adjacent_x(input_model, x_coord, y_coord, z_coord, x_dir):
    x_len = len(input_model[:, 0, 0])
    x_coord_new = x_coord+x_dir

    if (x_coord_new < x_len) and (x_coord_new >= 0) and not np.equal(input_model[x_coord_new, y_coord, z_coord], input_model[x_coord, y_coord, z_coord]):
        return True
    elif (x_coord_new >= x_len) or (x_coord_new < 0):
        return True
    else:
        return False

@njit()
def check_adjacent_y(input_model, x_coord, y_coord, z_coord, y_dir):
    y_len = len(input_model[0, :, 0])
    y_coord_new = y_coord+y_dir

    if (y_coord_new < y_len) and (y_coord_new >= 0) and not np.equal(input_model[x_coord, y_coord_new, z_coord], input_model[x_coord, y_coord, z_coord]):
        return True
    elif (y_coord_new >= y_len) or (y_coord_new < 0):
        return True
    else:
        return False

@njit()
def check_adjacent_z(input_model, x_coord, y_coord, z_coord, z_dir):
    z_len = len(input_model[0, 0, :])
    z_coord_new = z_coord+z_dir

    if (z_coord_new < z_len) and (z_coord_new >= 0) and not np.equal(input_model[x_coord, y_coord, z_coord_new], input_model[x_coord, y_coord, z_coord]):
        return True
    elif (z_coord_new >= z_len) or (z_coord_new < 0):
        return True
    else:
        return False

@njit()
def addVerticesAndTriangles(voxel_model_array, x, y, z, vi):
    adjacent = [
        [check_adjacent_x(voxel_model_array, x, y, z, 1), check_adjacent_x(voxel_model_array, x, y, z, -1)],
        [check_adjacent_y(voxel_model_array, x, y, z, 1), check_adjacent_y(voxel_model_array, x, y, z, -1)],
        [check_adjacent_z(voxel_model_array, x, y, z, 1), check_adjacent_z(voxel_model_array, x, y, z, -1)],
    ]

    verts_indices = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    verts = []
    tris = []

    if adjacent[0][0] or adjacent[1][0] or adjacent[2][0]:
        verts.append([x + 0.5, y + 0.5, z + 0.5])
        verts_indices[0] = vi
        vi = vi + 1

    if adjacent[0][0] or adjacent[1][1] or adjacent[2][0]:
        verts.append([x + 0.5, y - 0.5, z + 0.5])
        verts_indices[1] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][0] or adjacent[2][0]:
        verts.append([x - 0.5, y + 0.5, z + 0.5])
        verts_indices[2] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][1] or adjacent[2][0]:
        verts.append([x - 0.5, y - 0.5, z + 0.5])
        verts_indices[3] = vi
        vi = vi + 1

    if adjacent[0][0] or adjacent[1][0] or adjacent[2][1]:
        verts.append([x + 0.5, y + 0.5, z - 0.5])
        verts_indices[4] = vi
        vi = vi + 1

    if adjacent[0][0] or adjacent[1][1] or adjacent[2][1]:
        verts.append([x + 0.5, y - 0.5, z - 0.5])
        verts_indices[5] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][0] or adjacent[2][1]:
        verts.append([x - 0.5, y + 0.5, z - 0.5])
        verts_indices[6] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][1] or adjacent[2][1]:
        verts.append([x - 0.5, y - 0.5, z - 0.5])
        verts_indices[7] = vi
        vi = vi + 1

    if (verts_indices[0] != 0) and (verts_indices[1] != 0) and (verts_indices[2] != 0) and (
            verts_indices[3] != 0):
        tris.append([verts_indices[0] - 1, verts_indices[1] - 1, verts_indices[2] - 1])
        tris.append([verts_indices[1] - 1, verts_indices[3] - 1, verts_indices[2] - 1])

    if (verts_indices[0] != 0) and (verts_indices[1] != 0) and (verts_indices[4] != 0) and (
            verts_indices[5] != 0):
        tris.append([verts_indices[1] - 1, verts_indices[0] - 1, verts_indices[5] - 1])
        tris.append([verts_indices[0] - 1, verts_indices[4] - 1, verts_indices[5] - 1])

    if (verts_indices[0] != 0) and (verts_indices[2] != 0) and (verts_indices[4] != 0) and (
            verts_indices[6] != 0):
        tris.append([verts_indices[0] - 1, verts_indices[2] - 1, verts_indices[4] - 1])
        tris.append([verts_indices[2] - 1, verts_indices[6] - 1, verts_indices[4] - 1])

    if (verts_indices[2] != 0) and (verts_indices[3] != 0) and (verts_indices[6] != 0) and (
            verts_indices[7] != 0):
        tris.append([verts_indices[2] - 1, verts_indices[3] - 1, verts_indices[6] - 1])
        tris.append([verts_indices[3] - 1, verts_indices[7] - 1, verts_indices[6] - 1])

    if (verts_indices[1] != 0) and (verts_indices[3] != 0) and (verts_indices[5] != 0) and (
            verts_indices[7] != 0):
        tris.append([verts_indices[3] - 1, verts_indices[1] - 1, verts_indices[7] - 1])
        tris.append([verts_indices[1] - 1, verts_indices[5] - 1, verts_indices[7] - 1])

    if (verts_indices[4] != 0) and (verts_indices[5] != 0) and (verts_indices[6] != 0) and (
            verts_indices[7] != 0):
        tris.append([verts_indices[5] - 1, verts_indices[4] - 1, verts_indices[7] - 1])
        tris.append([verts_indices[4] - 1, verts_indices[6] - 1, verts_indices[7] - 1])

    return verts, verts_indices, tris, vi
