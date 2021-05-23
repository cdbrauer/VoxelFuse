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
from typing import List, Tuple
from numba import njit
from tqdm import tqdm

from voxelfuse.voxel_model import VoxelModel, rgb_to_hex
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
    def fromVoxelModel(cls, voxel_model: VoxelModel):
        """
        Generate a mesh object from a VoxelModel object.

        ----

        Example:

        ``mesh1 = Mesh.fromVoxelModel(model1)``

        ----

        :param voxel_model: VoxelModel object to be converted to a mesh
        :return: Mesh
        """
        voxel_model_fit = voxel_model.fitWorkspace()
        voxel_model_array = voxel_model_fit.voxels.astype(np.uint16)
        model_materials = voxel_model_fit.materials
        model_offsets = voxel_model_fit.coords

        # Find exterior voxels
        exterior_voxels_array = voxel_model_fit.difference(voxel_model_fit.erode(radius=1, connectivity=1)).voxels
        
        x_len, y_len, z_len = voxel_model_array.shape
        
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
        verts_indices = np.zeros((x_len+1, y_len+1, z_len+1))
        tris = []
        vi = 1  # Tracks current vertex index

        # Loop through voxel_model_array data
        for voxel_coords in tqdm(exterior_voxels_coords, desc='Meshing'):
            x, y, z = voxel_coords

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
            new_verts, verts_indices, new_tris, vi = addVerticesAndTriangles(voxel_model_array, verts_indices, model_offsets, x, y, z, vi)
            verts += new_verts
            tris += new_tris

            # Apply color to all vertices
            for i in range(len(new_verts)):
                verts_colors.append(voxel_color)

        verts = np.array(verts)
        verts_colors = np.array(verts_colors)
        tris = np.array(tris)

        # Reverse face normals
        tris_rev = np.empty_like(tris)
        tris_rev[:, 0] = tris[:, 1]
        tris_rev[:, 1] = tris[:, 0]
        tris_rev[:, 2] = tris[:, 2]

        return cls(voxel_model_array, verts, verts_colors, tris_rev, voxel_model.resolution)

    # Create mesh using a marching cubes algorithm
    @classmethod
    def marchingCubes(cls, voxel_model: VoxelModel, smooth: bool = False):
        """
        Generate a mesh object from a VoxelModel object using a marching cubes algorithm.

        This meshing approach is best suited to high resolution models where some smoothing is acceptable.

        :param voxel_model: VoxelModel object to be converted to a mesh
        :param smooth: Enable smoothing
        :return: None
        """
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

        # Shift model to align with origin
        verts = np.subtract(verts, 0.5)

        # Generate colors
        verts_colors = []
        voxel_color = [0.8, 0.8, 0.8, 1]
        for i in range(len(verts)):
            verts_colors.append(voxel_color)
        verts_colors = np.array(verts_colors)

        return cls(voxels_padded, verts, verts_colors, tris, voxel_model.resolution)

    # Set the resolution of the mesh
    def setResolution(self, resolution: float):
        """
        Change the defined resolution of a mesh.

        The mesh resolution will determine the scale of plots and exported mesh files.

        :param resolution: Number of voxels per mm (higher number = finer resolution)
        :return: None
        """
        self.res = resolution

    # Add mesh to a K3D plot in Jupyter Notebook
    def plot(self, plot = None, name: str = 'mesh', wireframe: bool = True, mm_scale: bool = False, **kwargs):
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
        :param wireframe: Enable displaying mesh as a wireframe
        :param mm_scale: Enable to use a mm plot scale, disable to use a voxel plot scale
        :param kwargs: Additional display options (see above)
        :return: K3D plot object
        """
        # Get verts
        verts = self.verts

        # Adjust coordinate scale
        if mm_scale:
            verts = np.divide(verts, self.res)

        # Get tris
        tris = self.tris

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
        # Adjust coordinate scale
        verts = np.divide(self.verts, self.res)

        cells = {
            "triangle": self.tris
        }

        output_mesh = meshio.Mesh(verts, cells)
        meshio.write(filename, output_mesh)

# Helper functions ##############################################################
@njit()
def check_adjacent_x(input_model: np.ndarray, x_coord: int, y_coord: int, z_coord: int, x_dir: int):
    """
    Check if a target voxel has another voxel of the same material
    adjacent to it in the X direction.

    :param input_model: VoxelModel.voxels
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
def check_adjacent_y(input_model: np.ndarray, x_coord: int, y_coord: int, z_coord: int, y_dir: int):
    """
    Check if a target voxel has another voxel of the same material
    adjacent to it in the Y direction.

    :param input_model: VoxelModel.voxels
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
def check_adjacent_z(input_model: np.ndarray, x_coord: int, y_coord: int, z_coord: int, z_dir):
    """
    Check if a target voxel has another voxel of the same material
    adjacent to it in the Z direction.

    :param input_model: VoxelModel.voxels
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
def addVerticesAndTriangles(voxel_model_array: np.ndarray, verts_indices: np.ndarray, model_offsets: Tuple, x: int, y: int, z: int, vi: int):
    """
    Find the applicable mesh vertices and triangles for a target voxel.

    :param voxel_model_array: VoxelModel.voxels
    :param verts_indices: verts indices array
    :param model_offsets: VoxelModel.coords
    :param x: Target voxel X location
    :param y: Target voxel Y location
    :param z: Target voxel Z location
    :param vi: Current vertex index
    :return: New verts, Updated verts indices array, New tris, Updated current vert index
    """
    adjacent = [
        [check_adjacent_x(voxel_model_array, x, y, z, 1), check_adjacent_x(voxel_model_array, x, y, z, -1)],
        [check_adjacent_y(voxel_model_array, x, y, z, 1), check_adjacent_y(voxel_model_array, x, y, z, -1)],
        [check_adjacent_z(voxel_model_array, x, y, z, 1), check_adjacent_z(voxel_model_array, x, y, z, -1)],
    ]

    cube_verts_indices = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    verts = []
    tris = []

    if adjacent[0][0] or adjacent[1][0] or adjacent[2][0]:
        vert_pos = (x+1, y+1, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[0] = verts_indices[vert_pos]

    if adjacent[0][0] or adjacent[1][1] or adjacent[2][0]:
        vert_pos = (x+1, y, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[1] = verts_indices[vert_pos]

    if adjacent[0][1] or adjacent[1][0] or adjacent[2][0]:
        vert_pos = (x, y+1, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[2] = verts_indices[vert_pos]

    if adjacent[0][1] or adjacent[1][1] or adjacent[2][0]:
        vert_pos = (x, y, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[3] = verts_indices[vert_pos]

    if adjacent[0][0] or adjacent[1][0] or adjacent[2][1]:
        vert_pos = (x+1, y+1, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[4] = verts_indices[vert_pos]

    if adjacent[0][0] or adjacent[1][1] or adjacent[2][1]:
        vert_pos = (x+1, y, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[5] = verts_indices[vert_pos]

    if adjacent[0][1] or adjacent[1][0] or adjacent[2][1]:
        vert_pos = (x, y+1, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[6] = verts_indices[vert_pos]

    if adjacent[0][1] or adjacent[1][1] or adjacent[2][1]:
        vert_pos = (x, y, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[7] = verts_indices[vert_pos]

    if (cube_verts_indices[0] != 0) and (cube_verts_indices[1] != 0) and (cube_verts_indices[2] != 0) and (
            cube_verts_indices[3] != 0):
        tris.append([cube_verts_indices[0] - 1, cube_verts_indices[1] - 1, cube_verts_indices[2] - 1])
        tris.append([cube_verts_indices[1] - 1, cube_verts_indices[3] - 1, cube_verts_indices[2] - 1])

    if (cube_verts_indices[0] != 0) and (cube_verts_indices[1] != 0) and (cube_verts_indices[4] != 0) and (
            cube_verts_indices[5] != 0):
        tris.append([cube_verts_indices[1] - 1, cube_verts_indices[0] - 1, cube_verts_indices[5] - 1])
        tris.append([cube_verts_indices[0] - 1, cube_verts_indices[4] - 1, cube_verts_indices[5] - 1])

    if (cube_verts_indices[0] != 0) and (cube_verts_indices[2] != 0) and (cube_verts_indices[4] != 0) and (
            cube_verts_indices[6] != 0):
        tris.append([cube_verts_indices[0] - 1, cube_verts_indices[2] - 1, cube_verts_indices[4] - 1])
        tris.append([cube_verts_indices[2] - 1, cube_verts_indices[6] - 1, cube_verts_indices[4] - 1])

    if (cube_verts_indices[2] != 0) and (cube_verts_indices[3] != 0) and (cube_verts_indices[6] != 0) and (
            cube_verts_indices[7] != 0):
        tris.append([cube_verts_indices[2] - 1, cube_verts_indices[3] - 1, cube_verts_indices[6] - 1])
        tris.append([cube_verts_indices[3] - 1, cube_verts_indices[7] - 1, cube_verts_indices[6] - 1])

    if (cube_verts_indices[1] != 0) and (cube_verts_indices[3] != 0) and (cube_verts_indices[5] != 0) and (
            cube_verts_indices[7] != 0):
        tris.append([cube_verts_indices[3] - 1, cube_verts_indices[1] - 1, cube_verts_indices[7] - 1])
        tris.append([cube_verts_indices[1] - 1, cube_verts_indices[5] - 1, cube_verts_indices[7] - 1])

    if (cube_verts_indices[4] != 0) and (cube_verts_indices[5] != 0) and (cube_verts_indices[6] != 0) and (
            cube_verts_indices[7] != 0):
        tris.append([cube_verts_indices[5] - 1, cube_verts_indices[4] - 1, cube_verts_indices[7] - 1])
        tris.append([cube_verts_indices[4] - 1, cube_verts_indices[6] - 1, cube_verts_indices[7] - 1])

    return verts, verts_indices, tris, vi
