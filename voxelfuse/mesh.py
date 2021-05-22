"""
Mesh Class

Initialized from a voxel model

----

Copyright 2021 - Cole Brauer, Dan Aukes
"""

import numpy as np
import meshio
import mcubes
import k3d
from typing import List
from numba import njit
from tqdm import tqdm

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.materials import material_properties

class Mesh:
    """
    Mesh object that can be exported or passed to a Plot object.
    """

    def __init__(self, voxels: np.ndarray, verts: np.ndarray, verts_colors: np.ndarray, tris: np.ndarray, resolution: float):
        """
        Initialize a Mesh object.

        :param voxels: Voxel data array
        :param verts: List of coordinates of surface vertices
        :param verts_colors: List of colors associated with each vertex
        :param tris: List of the sets of vertices associated with triangular faces
        :param resolution: Number of voxels per mm
        """

        self.model = voxels
        self.verts = verts
        self.colors = verts_colors
        self.tris = tris
        self.res = resolution

    # Create mesh from voxel data
    @classmethod
    def fromVoxelModel(cls, voxel_model: VoxelModel, resolution: float = -1):
        """
        Generate a mesh object from a VoxelModel object.

        ----

        Example:

        ``mesh1 = Mesh.fromVoxelModel(model1)``

        ----

        :param voxel_model: VoxelModel object to be converted to a mesh
        :param resolution: Number of voxels per mm, -1 to use model resolution
        :return: Mesh
        """
        if resolution < 0:
            resolution = voxel_model.resolution

        voxel_model_fit = voxel_model.fitWorkspace()
        voxel_model_array = voxel_model_fit.voxels.astype(np.uint16)
        model_materials = voxel_model_fit.materials
        model_offsets = voxel_model_fit.coords

        # Find exterior voxels
        exterior_voxels_array = voxel_model_fit.difference(voxel_model_fit.erode(radius=1, connectivity=1)).voxels
        
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

            for i in range(voxel_model.materials.shape[1]-1):
                r = r + model_materials[voxel_model_array[x, y, z]][i+1] * material_properties[i]['r']
                g = g + model_materials[voxel_model_array[x, y, z]][i+1] * material_properties[i]['g']
                b = b + model_materials[voxel_model_array[x, y, z]][i+1] * material_properties[i]['b']

            r = 1 if r > 1 else r
            g = 1 if g > 1 else g
            b = 1 if b > 1 else b

            a = 1 - model_materials[voxel_model_array[x, y, z]][1]

            voxel_color = [r, g, b, a]

            # Add cube vertices
            new_verts, verts_indices, new_tris, vi = addVerticesAndTriangles(voxel_model_array, model_offsets, resolution, x, y, z, vi)
            verts += new_verts
            tris += new_tris

            # Apply color to all vertices
            for i in range(0, np.count_nonzero(verts_indices)):
                verts_colors.append(voxel_color)

        verts = np.array(verts)
        verts_colors = np.array(verts_colors)
        tris = np.array(tris)

        return cls(voxel_model_array, verts, verts_colors, tris, resolution)

    # Create and save a mesh file using a marching cubes algorithm
    @classmethod
    def marchingCubes(cls, voxel_model: VoxelModel, resolution: float = -1, smooth: bool = False):
        """
        Generate a mesh object from a VoxelModel object using a marching cubes algorithm.

        This meshing approach is best suited to high resolution models where some smoothing is acceptable.

        :param voxel_model: VoxelModel object to be converted to a mesh
        :param resolution: Number of voxels per mm, -1 to use model resolution
        :param smooth: Enable smoothing
        :return: None
        """
        if resolution < 0:
            resolution = voxel_model.resolution

        voxel_model_fit = voxel_model.fitWorkspace().getOccupied()
        voxels = voxel_model_fit.voxels.astype(np.uint16)
        x, y, z = voxels.shape

        voxels_padded = np.zeros((x + 2, y + 2, z + 2))
        voxels_padded[1:-1, 1:-1, 1:-1] = voxels

        if smooth:
            voxels_padded = mcubes.smooth(voxels_padded)
            levelset = 0
        else:
            levelset = 0.5

        verts, tris = mcubes.marching_cubes(voxels_padded, levelset)

        # Adjust coordinate scale
        verts = np.divide(verts, resolution)

        # Generate colors
        verts_colors = []
        voxel_color = [0.8, 0.8, 0.8, 1]
        for i in range(len(verts)):
            verts_colors.append(voxel_color)
        verts_colors = np.array(verts_colors)

        return cls(voxels_padded, verts, verts_colors, tris, resolution)

    # Add mesh to a K3D plot in Jupyter Notebook
    def plot(self, plot = None, name: str = 'mesh', voxel_scale: bool = True, wireframe: bool = True, **kwargs):
        """
        Add mesh to a K3D plot.

        Additional display options:
            flat_shading: `bool`.
                Whether mesh should display with flat shading.
            opacity: `float`.
                Opacity of mesh.
            volume: `array_like`.
                3D array of `float`
            volume_bounds: `array_like`.
                6-element tuple specifying the bounds of the volume data (x0, x1, y0, y1, z0, z1)
            opacity_function: `array`.
                A list of float tuples (attribute value, opacity), sorted by attribute value. The first
                typles should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
            side: `string`.
                Control over which side to render for a mesh. Legal values are `front`, `back`, `double`.
            texture: `bytes`.
                Image data in a specific format.
            texture_file_format: `str`.
                Format of the data, it should be the second part of MIME format of type 'image/',
                for example 'jpeg', 'png', 'gif', 'tiff'.
            uvs: `array_like`.
                Array of float uvs for the texturing, coresponding to each vertex.
            kwargs: `dict`.
                Dictionary arguments to configure transform and model_matrix.

        :param plot: Plot object to add mesh to
        :param name: Mesh name
        :param voxel_scale: Enable to use a voxel plot scale, disable to use a mm plot scale
        :param wireframe: Enable displaying mesh as a wireframe
        :param kwargs: Additional display options (see above)
        :return: K3D plot object
        """
        # Get verts
        verts = self.verts

        # Adjust coordinate scale
        if voxel_scale:
            verts = np.multiply(verts, self.res)
            verts = np.add(verts, 0.5)

        # Get tris
        tris = np.empty_like(self.tris)
        tris[:, 0] = self.tris[:, 1]
        tris[:, 1] = self.tris[:, 0]
        tris[:, 2] = self.tris[:, 2]

        # Get colors
        colors = []
        for c in self.colors:
            colors.append(rgb_to_hex(c[0], c[1], c[2]))
        colors = np.array(colors, dtype=np.uint32)

        # Plot
        if plot is None:
            plot = k3d.plot()

        plot += k3d.mesh(verts.astype(np.float32), tris.astype(np.uint32), colors=colors, name=name, wireframe=wireframe, **kwargs)
        return plot

    # Export model from mesh data
    def export(self, filename: str):
        """
        Save a copy of the mesh with the specified name and file format.

        ----

        Example:

        ``mesh1.export('result.stl')``

        ----

        :param filename: File name with extension
        :return: None
        """
        cells = {
            "triangle": self.tris
        }

        output_mesh = meshio.Mesh(self.verts, cells)
        meshio.write(filename, output_mesh)

# Helper functions ##############################################################
def rgb_to_hex(r: float, g: float, b: float):
    """
    Convert RGB values to a single hexadecimal value.

    :param r: Red percentage (0-1)
    :param g: Green percentage (0-1)
    :param b: Blue percentage (0-1)
    :return: Hexadecimal color as an integer
    """
    r = round(r * 255)
    g = round(g * 255)
    b = round(b * 255)
    hex_str = '0x{:02x}{:02x}{:02x}'.format(r, g, b)
    return int(hex_str, base=16)

@njit()
def check_adjacent_x(input_model, x_coord, y_coord, z_coord, x_dir):
    """
    Check if a target voxel has another voxel of the same material
    adjacent to it in the X direction.

    :param input_model: VoxelModel
    :param x_coord: Target voxel X location
    :param y_coord: Target voxel Y location
    :param z_coord: Target voxel Z location
    :param x_dir: Specify X direction and distance (usually 1 or -1)
    :return: Adjacent voxel present/not present
    """
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
    """
    Check if a target voxel has another voxel of the same material
    adjacent to it in the Y direction.

    :param input_model: VoxelModel
    :param x_coord: Target voxel X location
    :param y_coord: Target voxel Y location
    :param z_coord: Target voxel Z location
    :param y_dir: Specify Y direction and distance (usually 1 or -1)
    :return: Adjacent voxel present/not present
    """
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
    """
    Check if a target voxel has another voxel of the same material
    adjacent to it in the Z direction.

    :param input_model: VoxelModel
    :param x_coord: Target voxel X location
    :param y_coord: Target voxel Y location
    :param z_coord: Target voxel Z location
    :param z_dir: Specify Z direction and distance (usually 1 or -1)
    :return: Adjacent voxel present/not present
    """
    z_len = len(input_model[0, 0, :])
    z_coord_new = z_coord+z_dir

    if (z_coord_new < z_len) and (z_coord_new >= 0) and not np.equal(input_model[x_coord, y_coord, z_coord_new], input_model[x_coord, y_coord, z_coord]):
        return True
    elif (z_coord_new >= z_len) or (z_coord_new < 0):
        return True
    else:
        return False

@njit()
def addVerticesAndTriangles(voxel_model_array, model_offsets, resolution, x, y, z, vi):
    """
    Find the applicable mesh vertices and triangles for a target voxel.

    :param voxel_model_array: VoxelModel.voxels
    :param model_offsets: VoxelModel.coords
    :param resolution: Number of voxels per mm
    :param x: Target voxel X location
    :param y: Target voxel Y location
    :param z: Target voxel Z location
    :param vi: Current vertex index
    :return: New verts, Indices for new verts, New tris, New current vert index
    """
    adjacent = [
        [check_adjacent_x(voxel_model_array, x, y, z, 1), check_adjacent_x(voxel_model_array, x, y, z, -1)],
        [check_adjacent_y(voxel_model_array, x, y, z, 1), check_adjacent_y(voxel_model_array, x, y, z, -1)],
        [check_adjacent_z(voxel_model_array, x, y, z, 1), check_adjacent_z(voxel_model_array, x, y, z, -1)],
    ]

    verts_indices = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    verts = []
    tris = []

    if adjacent[0][0] or adjacent[1][0] or adjacent[2][0]:
        verts.append([(x + 0.5 + model_offsets[0])/resolution, (y + 0.5 + model_offsets[1])/resolution, (z + 0.5 + model_offsets[2])/resolution])
        verts_indices[0] = vi
        vi = vi + 1

    if adjacent[0][0] or adjacent[1][1] or adjacent[2][0]:
        verts.append([(x + 0.5 + model_offsets[0])/resolution, (y - 0.5 + model_offsets[1])/resolution, (z + 0.5 + model_offsets[2])/resolution])
        verts_indices[1] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][0] or adjacent[2][0]:
        verts.append([(x - 0.5 + model_offsets[0])/resolution, (y + 0.5 + model_offsets[1])/resolution, (z + 0.5 + model_offsets[2])/resolution])
        verts_indices[2] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][1] or adjacent[2][0]:
        verts.append([(x - 0.5 + model_offsets[0])/resolution, (y - 0.5 + model_offsets[1])/resolution, (z + 0.5 + model_offsets[2])/resolution])
        verts_indices[3] = vi
        vi = vi + 1

    if adjacent[0][0] or adjacent[1][0] or adjacent[2][1]:
        verts.append([(x + 0.5 + model_offsets[0])/resolution, (y + 0.5 + model_offsets[1])/resolution, (z - 0.5 + model_offsets[2])/resolution])
        verts_indices[4] = vi
        vi = vi + 1

    if adjacent[0][0] or adjacent[1][1] or adjacent[2][1]:
        verts.append([(x + 0.5 + model_offsets[0])/resolution, (y - 0.5 + model_offsets[1])/resolution, (z - 0.5 + model_offsets[2])/resolution])
        verts_indices[5] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][0] or adjacent[2][1]:
        verts.append([(x - 0.5 + model_offsets[0])/resolution, (y + 0.5 + model_offsets[1])/resolution, (z - 0.5 + model_offsets[2])/resolution])
        verts_indices[6] = vi
        vi = vi + 1

    if adjacent[0][1] or adjacent[1][1] or adjacent[2][1]:
        verts.append([(x - 0.5 + model_offsets[0])/resolution, (y - 0.5 + model_offsets[1])/resolution, (z - 0.5 + model_offsets[2])/resolution])
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
