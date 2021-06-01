"""
Mesh Class

Initialized from a voxel model

----

Copyright 2021 - Cole Brauer, Dan Aukes
"""

import sys
import numpy as np
import meshio
import k3d
import mcubes
from quad_mesh_simplify import simplify_mesh
from typing import Union as TypeUnion, Tuple
from numba import njit
from tqdm import tqdm

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qg
import pyqtgraph.opengl as pgo

from voxelfuse.voxel_model import VoxelModel, rgb_to_hex
from voxelfuse.materials import material_properties

class Mesh:
    """
    Mesh object that can be exported or passed to a Plot object.
    """

    def __init__(self, voxels: TypeUnion[np.ndarray, None], verts: np.ndarray, verts_colors: np.ndarray, tris: np.ndarray, resolution: float):
        """
        Initialize a Mesh object.

        Args:
            voxels: Voxel data array
            verts: List of coordinates of surface vertices
            verts_colors: List of colors associated with each vertex
            tris: List of the sets of vertices associated with triangular faces
            resolution: Number of voxels per mm
        """
        if voxels is not None:
            self.model = voxels
        else:
            self.model = np.array([[[0]]])

        self.verts = verts
        self.colors = verts_colors
        self.tris = tris
        self.res = resolution

    @classmethod
    def fromMeshFile(cls, filename: str, color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1)):
        """
        Import a mesh file to a mesh object.

        ----

        Example:

        ``mesh1 = vf.Mesh.fromMeshFile(example.stl)``

        ----

        Args:
            filename: File name with extension
            color: Mesh color in the format (r, g, b, a)
        
        Returns:
            Mesh
        """
        # Open file
        data = meshio.read(filename)

        # Read verts
        verts = np.array(data.points)

        # Align to origin
        x_min = np.min(verts[:, 0])
        y_min = np.min(verts[:, 1])
        z_min = np.min(verts[:, 2])
        verts[:, 0] = np.subtract(verts[:, 0], x_min)
        verts[:, 1] = np.subtract(verts[:, 1], y_min)
        verts[:, 2] = np.subtract(verts[:, 2], z_min)

        # Generate colors
        verts_colors = generateColors(len(verts), color)

        # Read tris
        tris = []
        for cell in data.cells:
            if cell[0] == 'triangle':
                for tri in cell[1]:
                    tris.append(tri)
        tris = np.array(tris)

        return cls(None, verts, verts_colors, tris, 1)

    @classmethod
    def fromVoxelModel(cls, voxel_model: VoxelModel, color: Tuple[float, float, float, float] = None):
        """
        Generate a mesh object from a VoxelModel object.

        ----

        Example:

        ``mesh1 = vf.Mesh.fromVoxelModel(model1)``

        ----

        Args:
            voxel_model: VoxelModel object to be converted to a mesh
            color: Mesh color in the format (r, g, b, a), None to use voxel colors
        
        Returns:
            Mesh
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

            if color is None:
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
            else:
                voxel_color = list(color)

            # Add cube vertices
            new_verts, verts_indices, new_tris, vi = addVerticesAndTriangles(voxel_model_array, verts_indices, model_offsets, x, y, z, vi)
            verts += new_verts
            tris += new_tris

            # Apply color to all vertices
            for i in range(len(new_verts)):
                verts_colors.append(voxel_color)

        verts = np.array(verts, dtype=np.float32)
        verts_colors = np.array(verts_colors, dtype=np.float32)
        tris = np.array(tris, dtype=np.uint32)

        return cls(voxel_model_array, verts, verts_colors, tris, voxel_model.resolution)

    @classmethod
    def simpleSquares(cls, voxel_model: VoxelModel, color: Tuple[float, float, float, float] = None):
        """
        Generate a mesh object from a VoxelModel object using large square faces.

        This function can greatly reduce the file size of generated meshes. However, it may not correctly recognize
        small (1 voxel) model features and currently produces files with a nonstandard vertex arrangement. Use at your own
        risk.

        ----

        Example:

        ``mesh1 = vf.Mesh.simpleSquares(model1)``

        ----

        Args:
            voxel_model: VoxelModel object to be converted to a mesh
            color: Mesh color in the format (r, g, b, a), None to use voxel colors
        
        Returns:
            Mesh
        """
        voxel_model_fit = voxel_model.fitWorkspace()
        voxel_model_array = voxel_model_fit.voxels.astype(np.uint16)
        model_materials = voxel_model_fit.materials
        model_offsets = voxel_model_fit.coords

        x_len, y_len, z_len = voxel_model_array.shape

        # Determine vertex types
        vert_type = np.zeros((x_len + 1, y_len + 1, z_len + 1), dtype=np.uint8)
        vert_color = np.zeros((x_len + 1, y_len + 1, z_len + 1, 4), dtype=np.float32)
        for x in tqdm(range(x_len), desc='Finding voxel vertices'):
            for y in range(y_len):
                for z in range(z_len):
                    if voxel_model_array[x, y, z] > 0:
                        vert_type[x:x+2, y:y+2, z:z+2] = 1 # Type 1 = occupied/exterior

                        if color is None:
                            r = 0
                            g = 0
                            b = 0

                            for i in range(voxel_model.materials.shape[1] - 1):
                                r = r + model_materials[voxel_model_array[x, y, z]][i + 1] * material_properties[i]['r']
                                g = g + model_materials[voxel_model_array[x, y, z]][i + 1] * material_properties[i]['g']
                                b = b + model_materials[voxel_model_array[x, y, z]][i + 1] * material_properties[i]['b']

                            r = 1 if r > 1 else r
                            g = 1 if g > 1 else g
                            b = 1 if b > 1 else b

                            a = 1 - model_materials[voxel_model_array[x, y, z]][1]

                            voxel_color = np.array([r, g, b, a])
                        else:
                            voxel_color = np.array(color)

                        for cx in range(x, x+2):
                            for cy in range(y, y+2):
                                for cz in range(z, z+2):
                                    vert_color[cx, cy, cz, :] = voxel_color

        for x in tqdm(range(1, x_len), desc='Finding interior vertices'):
            for y in range(1, y_len):
                for z in range(1, z_len):
                    vert_type = markInterior(vert_type, x, y, z)

        for x in tqdm(range(0, x_len+1), desc='Finding feature vertices'):
            for y in range(0, y_len+1):
                for z in range(0, z_len+1):
                    vert_type = markInsideCorner(vert_type, x, y, z)

        # Initialize arrays
        vi = 0 # Tracks current vertex index
        verts = []
        colors = []
        tris = []
        quads = []
        vert_index = np.multiply(np.ones_like(vert_type, dtype=np.int32), -1)

        for x in tqdm(range(x_len + 1), desc='Meshing'):
            for y in range(y_len + 1):
                for z in range(z_len + 1):
                    dirs = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
                    for d in dirs:
                        vi, vert_type, vert_index, new_verts, new_colors, new_tris, new_quads = findSquare(vi, vert_type, vert_index, vert_color, voxel_model_array, x, y, z, d[0], d[1], d[2])
                        verts += new_verts
                        colors += new_colors
                        tris += new_tris
                        quads += new_quads

        verts = np.array(verts, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        tris = np.array(tris, dtype=np.uint32)
        quads = np.array(quads, dtype=np.uint32)

        # Shift model to align with origin
        verts[:, 0] = np.add(verts[:, 0], model_offsets[0])
        verts[:, 1] = np.add(verts[:, 1], model_offsets[1])
        verts[:, 2] = np.add(verts[:, 2], model_offsets[2])

        return cls(voxel_model_array, verts, colors, tris, voxel_model.resolution)

    @classmethod
    def marchingCubes(cls, voxel_model: VoxelModel, smooth: bool = False, color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1)):
        """
        Generate a mesh object from a VoxelModel object using a marching cubes algorithm.

        This meshing approach is best suited to high resolution models where some smoothing is acceptable.

        Args:
            voxel_model: VoxelModel object to be converted to a mesh
            smooth: Enable smoothing
            color: Mesh color in the format (r, g, b, a)
        
        Returns:
            None
        """
        voxel_model_fit = voxel_model.fitWorkspace().getOccupied()
        voxels = voxel_model_fit.voxels.astype(np.uint16)
        x, y, z = voxels.shape
        model_offsets = voxel_model_fit.coords

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
        verts[:, 0] = np.add(verts[:, 0], model_offsets[0])
        verts[:, 1] = np.add(verts[:, 1], model_offsets[1])
        verts[:, 2] = np.add(verts[:, 2], model_offsets[2])

        verts_colors = generateColors(len(verts), color)

        return cls(voxels_padded, verts, verts_colors, tris, voxel_model.resolution)

    @classmethod
    def copy(cls, mesh):
        """
        Initialize a Mesh that is a copy of another mesh.

        Args:
            mesh: Reference Mesh object
        
        Returns:
            Mesh
        """
        new_mesh = cls(np.copy(mesh.model), np.copy(mesh.verts), np.copy(mesh.colors), np.copy(mesh.tris), mesh.res)
        return new_mesh

    def setResolution(self, resolution: float):
        """
        Change the defined resolution of a mesh.

        The mesh resolution will determine the scale of plots and exported mesh files.

        Args:
            resolution: Number of voxels per mm (higher number = finer resolution)
        
        Returns:
            Mesh
        """
        new_mesh = Mesh.copy(self)
        new_mesh.res = resolution
        return new_mesh

    def scale(self, factor: float):
        """
        Apply a scaling factor to a mesh.

        Args:
            factor: Scaling factor
        
        Returns:
            Mesh
        """
        new_mesh = Mesh.copy(self)
        new_mesh.verts = np.multiply(self.verts, factor)
        return new_mesh

    def simplify(self, percent_verts: float, color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1)):
        """
        Simplify a mesh to contain a given percentage of the original number of vertices.

        More information on the simplification algorithm is available at: https://github.com/jannessm/quadric-mesh-simplification

        Args:
            percent_verts: Percentage of vertex count allowed in the result mesh, 0-1
            color: Mesh color in the format (r, g, b, a)
        
        Returns:
            Mesh
        """
        num_verts = self.verts.shape[0]
        target_verts = num_verts * percent_verts

        new_verts, new_tris = simplify_mesh(positions=self.verts.astype(np.double), face=self.tris.astype(np.uint32), num_nodes=target_verts)
        verts_colors = generateColors(len(new_verts), color)

        return Mesh(np.copy(self.model), new_verts, verts_colors, new_tris, self.res)

    def translate(self, vector: Tuple[float, float, float]):
        """
        Translate a model by the specified vector.

        Args:
            vector: Translation vector in voxels
        
        Returns:
            Mesh
        """
        new_mesh = Mesh.copy(self)
        new_mesh.verts[:, 0] = np.add(self.verts[:, 0], vector[0])
        new_mesh.verts[:, 1] = np.add(self.verts[:, 1], vector[1])
        new_mesh.verts[:, 2] = np.add(self.verts[:, 2], vector[2])
        return new_mesh

    def translateMM(self, vector: Tuple[float, float, float]):
        """
        Translate a model by the specified vector.

        Args:
            vector: Translation vector in mm
        
        Returns:
            Mesh
        """
        xV = vector[0] * self.res
        yV = vector[1] * self.res
        zV = vector[2] * self.res
        new_mesh = self.translate((xV, yV, zV))
        return new_mesh

    def setColor(self, color: Tuple[float, float, float, float]):
        """
        Change the color of a mesh.

        Args:
            color: Mesh color in the format (r, g, b, a)
        
        Returns:
            Mesh
        """
        new_mesh = Mesh.copy(self)
        new_mesh.colors = generateColors(len(self.verts), color)
        return new_mesh

    def plot(self, plot = None, name: str = 'mesh', wireframe: bool = True, mm_scale: bool = False, **kwargs):
        """
        Add mesh to a K3D plot in Jupyter Notebook.

        Additional display options:

        - flat_shading: `bool`. Whether mesh should display with flat shading.
        - opacity: `float`. Opacity of mesh.
        - volume: `array_like`. 3D array of `float`
        - volume_bounds: `array_like`. 6-element tuple specifying the bounds of the volume data (x0, x1, y0, y1, z0, z1)
        - opacity_function: `array`. A list of float tuples (attribute value, opacity), sorted by attribute value. The first tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        - side: `string`. Control over which side to render for a mesh. Legal values are `front`, `back`, `double`.
        - texture: `bytes`. Image data in a specific format.
        - texture_file_format: `str`. Format of the data, it should be the second part of MIME format of type 'image/', for example 'jpeg', 'png', 'gif', 'tiff'.
        - uvs: `array_like`. Array of float uvs for the texturing, coresponding to each vertex.
        - kwargs: `dict`. Dictionary arguments to configure transform and model_matrix.

        More information available at: https://github.com/K3D-tools/K3D-jupyter

        Args:
            plot: Plot object to add mesh to
            name: Mesh name
            wireframe: Enable displaying mesh as a wireframe
            mm_scale: Enable to use a mm plot scale, disable to use a voxel plot scale
            kwargs: Additional display options (see above)
        
        Returns:
            K3D plot object
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

    def viewer(self, grids: bool = False, drawEdges: bool = True,
               edgeColor: Tuple[float, float, float, float] = (0, 0, 0, 0.5),
               positionOffset: Tuple[int, int, int] = (0, 0, 0), viewAngle: Tuple[int, int, int] = (40, 30, 300),
               resolution: Tuple[int, int] = (1280, 720), name: str = 'Plot 1', export: bool = False):
        """
        Display the mesh in a 3D viewer window.

        This function will block program execution until viewer window is closed

        Args:
            grids: Enable/disable display of XYZ axes and grids
            drawEdges: Enable/disable display of voxel edges
            edgeColor: Set display color of voxel edges
            positionOffset: Offset of the camera target from the center of the model in voxels
            viewAngle: Elevation, Azimuth, and Distance of the camera
            resolution: Window resolution in px
            name: Plot window name
            export: Enable/disable exporting a screenshot of the plot
        
        Returns:
            None
        """
        app = qtw.QApplication(sys.argv)

        mesh_data = pgo.MeshData(vertexes=self.verts, faces=self.tris, vertexColors=self.colors, faceColors=None)
        mesh_item = pgo.GLMeshItem(meshdata=mesh_data, shader='balloon', drawEdges=drawEdges, edgeColor=edgeColor,
                                   smooth=False, computeNormals=False, glOptions='translucent')

        widget = pgo.GLViewWidget()
        widget.setBackgroundColor('w')
        widget.addItem(mesh_item)

        if grids:
            # Add grids
            gx = pgo.GLGridItem()
            gx.setSize(x=50, y=50, z=50)
            gx.rotate(90, 0, 1, 0)
            gx.translate(-0.5, 24.5, 24.5)
            widget.addItem(gx)
            gy = pgo.GLGridItem()
            gy.setSize(x=50, y=50, z=50)
            gy.rotate(90, 1, 0, 0)
            gy.translate(24.5, -0.5, 24.5)
            widget.addItem(gy)
            gz = pgo.GLGridItem()
            gz.setSize(x=50, y=50, z=50)
            gz.translate(24.5, 24.5, -0.5)
            widget.addItem(gz)

            # Add axes
            ptsx = np.array([[-0.5, -0.5, -0.5], [50, -0.5, -0.5]])
            pltx = pgo.GLLinePlotItem(pos=ptsx, color=(1, 0, 0, 1), width=1, antialias=True)
            widget.addItem(pltx)
            ptsy = np.array([[-0.5, -0.5, -0.5], [-0.5, 50, -0.5]])
            plty = pgo.GLLinePlotItem(pos=ptsy, color=(0, 1, 0, 1), width=1, antialias=True)
            widget.addItem(plty)
            ptsz = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, 50]])
            pltz = pgo.GLLinePlotItem(pos=ptsz, color=(0, 0, 1, 1), width=1, antialias=True)
            widget.addItem(pltz)

        # Set plot options
        widget.opts['center'] = qg.QVector3D(((self.model.shape[0] / self.res) / 2) + positionOffset[0],
                                             ((self.model.shape[1] / self.res) / 2) + positionOffset[1],
                                             ((self.model.shape[2] / self.res) / 2) + positionOffset[2])
        widget.opts['elevation'] = viewAngle[0]
        widget.opts['azimuth'] = viewAngle[1]
        widget.opts['distance'] = viewAngle[2]
        widget.resize(resolution[0], resolution[1])

        # Show plot
        widget.setWindowTitle(str(name))
        widget.show()

        app.processEvents()

        # if export: # TODO: Fix export code
        #     widget.paintGL()
        #     widget.grabFrameBuffer().save(str(name) + '.png')

        print('Close viewer to resume program')
        app.exec_()
        app.quit()

    # Export model from mesh data
    def export(self, filename: str):
        """
        Save a copy of the mesh with the specified name and file format.

        ----

        Example:

        ``mesh1.export('result.stl')``

        ----

        Args:
            filename: File name with extension
        
        Returns:
            None
        """
        # Adjust coordinate scale
        verts = np.divide(self.verts, self.res)

        cells = {
            "triangle": self.tris
        }

        output_mesh = meshio.Mesh(verts, cells)
        meshio.write(filename, output_mesh)

# Helper functions ##############################################################
def generateColors(n: int, color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1)):
    """
    Generate a colors list with the given number of elements

    Args:
        n: Number of vertices in target model
        color: Mesh color in the format (r, g, b, a)
    
    Returns:
        List of vertex colors
    """
    verts_colors = []
    voxel_color = list(color)
    for i in range(n):
        verts_colors.append(voxel_color)
    verts_colors = np.array(verts_colors)
    return verts_colors

@njit()
def check_adjacent(input_model: np.ndarray, x_coord: int, y_coord: int, z_coord: int, x_dir: int, y_dir: int, z_dir: int):
    """
    Check if a target voxel has another voxel adjacent to it in the specified direction.

    Args:
        input_model: VoxelModel.voxels
        x_coord: Target voxel X location
        y_coord: Target voxel Y location
        z_coord: Target voxel Z location
        x_dir: Specify X direction and distance (usually 1 or -1)
        y_dir: Specify Y direction and distance (usually 1 or -1)
        z_dir: Specify Z direction and distance (usually 1 or -1)
    
    Returns:
        Adjacent voxel present/not present
    """
    x_coord_new = x_coord+x_dir
    y_coord_new = y_coord+y_dir
    z_coord_new = z_coord+z_dir

    x_in_bounds = (x_coord_new >= 0) and (x_coord_new < input_model.shape[0])
    y_in_bounds = (y_coord_new >= 0) and (y_coord_new < input_model.shape[1])
    z_in_bounds = (z_coord_new >= 0) and (z_coord_new < input_model.shape[2])

    if x_in_bounds and y_in_bounds and z_in_bounds and input_model[x_coord_new, y_coord_new, z_coord_new] > 0:
        return True
    else:
        return False

@njit()
def addVerticesAndTriangles(voxel_model_array: np.ndarray, verts_indices: np.ndarray, model_offsets: Tuple, x: int, y: int, z: int, vi: int):
    """
    Find the applicable mesh vertices and triangles for a target voxel.

    Args:
        voxel_model_array: VoxelModel.voxels
        verts_indices: verts indices array
        model_offsets: VoxelModel.coords
        x: Target voxel X location
        y: Target voxel Y location
        z: Target voxel Z location
        vi: Current vertex index

    Returns:
        New verts, Updated verts indices array, New tris, Updated current vert index
    """
    adjacent = [
        [check_adjacent(voxel_model_array, x, y, z, 1, 0, 0), check_adjacent(voxel_model_array, x, y, z, -1, 0, 0)],
        [check_adjacent(voxel_model_array, x, y, z, 0, 1, 0), check_adjacent(voxel_model_array, x, y, z, 0, -1, 0)],
        [check_adjacent(voxel_model_array, x, y, z, 0, 0, 1), check_adjacent(voxel_model_array, x, y, z, 0, 0, -1)]
    ]

    cube_verts_indices = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    verts = []
    tris = []

    if not adjacent[0][0] or not adjacent[1][0] or not adjacent[2][0]:
        vert_pos = (x+1, y+1, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[0] = verts_indices[vert_pos]

    if not adjacent[0][0] or not adjacent[1][1] or not adjacent[2][0]:
        vert_pos = (x+1, y, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[1] = verts_indices[vert_pos]

    if not adjacent[0][1] or not adjacent[1][0] or not adjacent[2][0]:
        vert_pos = (x, y+1, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[2] = verts_indices[vert_pos]

    if not adjacent[0][1] or not adjacent[1][1] or not adjacent[2][0]:
        vert_pos = (x, y, z+1)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[3] = verts_indices[vert_pos]

    if not adjacent[0][0] or not adjacent[1][0] or not adjacent[2][1]:
        vert_pos = (x+1, y+1, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[4] = verts_indices[vert_pos]

    if not adjacent[0][0] or not adjacent[1][1] or not adjacent[2][1]:
        vert_pos = (x+1, y, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[5] = verts_indices[vert_pos]

    if not adjacent[0][1] or not adjacent[1][0] or not adjacent[2][1]:
        vert_pos = (x, y+1, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[6] = verts_indices[vert_pos]

    if not adjacent[0][1] or not adjacent[1][1] or not adjacent[2][1]:
        vert_pos = (x, y, z)
        if verts_indices[vert_pos] < 1:
            verts_indices[vert_pos] = vi
            verts.append([vert_pos[0]+model_offsets[0], vert_pos[1]+model_offsets[1], vert_pos[2]+model_offsets[2]])
            vi = vi + 1
        cube_verts_indices[7] = verts_indices[vert_pos]

    if not adjacent[0][0]:
        tris.append([cube_verts_indices[0] - 1, cube_verts_indices[1] - 1, cube_verts_indices[5] - 1])
        tris.append([cube_verts_indices[5] - 1, cube_verts_indices[4] - 1, cube_verts_indices[0] - 1])

    if not adjacent[1][0]:
        tris.append([cube_verts_indices[2] - 1, cube_verts_indices[0] - 1, cube_verts_indices[4] - 1])
        tris.append([cube_verts_indices[4] - 1, cube_verts_indices[6] - 1, cube_verts_indices[2] - 1])

    if not adjacent[2][0]:
        tris.append([cube_verts_indices[1] - 1, cube_verts_indices[0] - 1, cube_verts_indices[2] - 1])
        tris.append([cube_verts_indices[2] - 1, cube_verts_indices[3] - 1, cube_verts_indices[1] - 1])

    if not adjacent[0][1]:
        tris.append([cube_verts_indices[3] - 1, cube_verts_indices[2] - 1, cube_verts_indices[6] - 1])
        tris.append([cube_verts_indices[6] - 1, cube_verts_indices[7] - 1, cube_verts_indices[3] - 1])

    if not adjacent[1][1]:
        tris.append([cube_verts_indices[1] - 1, cube_verts_indices[3] - 1, cube_verts_indices[7] - 1])
        tris.append([cube_verts_indices[7] - 1, cube_verts_indices[5] - 1, cube_verts_indices[1] - 1])

    if not adjacent[2][1]:
        tris.append([cube_verts_indices[4] - 1, cube_verts_indices[5] - 1, cube_verts_indices[7] - 1])
        tris.append([cube_verts_indices[7] - 1, cube_verts_indices[6] - 1, cube_verts_indices[4] - 1])

    return verts, verts_indices.astype(np.uint32), tris, vi

@njit()
def markInterior(vert_type: np.ndarray, x: int, y: int, z: int):
    """
    Determine if target voxel is an interior voxel.

    Args:
        vert_type: Array of vertex types
        x: Target voxel X
        y: Target voxel Y
        z: Target voxel Z

    Returns:
        Updated array of vertex types
    """
    if np.all(vert_type[x-1:x+2, y-1:y+2, z-1:z+2] != 0):
        vert_type[x, y, z] = 3 # Type 3 = interior/already included in a square
    return vert_type

@njit()
def markInsideCorner(vert_type, x, y, z):
    """
    Determine if target voxel is an inside corner.

    Args:
        vert_type: Array of vertex types
        x: Target voxel X
        y: Target voxel Y
        z: Target voxel Z

    Returns:
        Updated array of vertex types
    """
    x_len, y_len, z_len = vert_type.shape
    adjacent_empty = [True, True, True, True, True, True]

    if x > 0:
        adjacent_empty[0] = (vert_type[x - 1, y, z] == 0)
    if x < x_len-1:
        adjacent_empty[1] = (vert_type[x + 1, y, z] == 0)

    if y > 0:
        adjacent_empty[2] = (vert_type[x, y - 1, z] == 0)
    if y < y_len-1:
        adjacent_empty[3] = (vert_type[x, y + 1, z] == 0)

    if z > 0:
        adjacent_empty[4] = (vert_type[x, y, z - 1] == 0)
    if z < z_len-1:
        adjacent_empty[5] = (vert_type[x, y, z + 1] == 0)

    is_face = [adjacent_empty[0] != adjacent_empty[1],
               adjacent_empty[2] != adjacent_empty[3],
               adjacent_empty[4] != adjacent_empty[5]]

    # Inside corner  - 0 faces
    # Face           - 1 face
    # Outside edge   - 2 faces
    # Outside corner - 3 faces
    if vert_type[x, y, z] == 1 and np.sum(np.array(is_face)) == 0:
        vert_type[x, y, z] = 2  # Type 2 = blocking

    return vert_type

@njit()
def findSquare(vi: int, vert_type: np.ndarray, vert_index: np.ndarray, vert_color: np.ndarray, voxel_model_array: np.ndarray, x: int, y: int, z: int, dx: int, dy: int, dz: int):
    """
    Find the largest square starting from a given point and generate the corresponding points and tris.

    Args:
        vi: Current vertex index
        vert_type: Array of vertex types
        vert_index: Array of vertex indices
        vert_color: Array of vertex colors
        voxel_model_array: Voxel data array
        x: Target voxel X
        y: Target voxel Y
        z: Target voxel Z
        dx: Square search step in X
        dy: Square search step in Y
        dz: Square search step in Z

    Returns:
        vi, vert_type, vert_index, new_verts, new_colors, new_tris, new_quads
    """
    x_len, y_len, z_len = vert_type.shape
    new_verts = []
    new_colors = []
    new_tris = []
    new_quads = []

    vert_on_surface = (vert_type[x, y, z] == 1 or vert_type[x, y, z] == 2)
    if vert_on_surface and x+dx < x_len and y+dy < y_len and z+dz < z_len:  # Point is a face vertex and next point is in bounds
        xn = x
        yn = y
        zn = z

        for i in range(1, max(x_len - x, y_len - y, z_len - z)):  # See if a square can be found starting at this point
            xn = x + dx * i
            yn = y + dy * i
            zn = z + dz * i

            # Check if endpoint is in bounds
            in_range = [xn < x_len,
                        yn < y_len,
                        zn < z_len]

            if not np.all(np.array(in_range)):
                xn = x + dx * (i-1)
                yn = y + dy * (i-1)
                zn = z + dz * (i-1)
                break

            face = (vert_type[x:xn+1, y:yn+1,  z:zn+1] == 1)
            blocking = (vert_type[x:xn+1, y:yn+1,  z:zn+1] == 2)
            on_surface = np.logical_or(face, blocking)

            # Check if square includes only surface vertices
            if not np.all(on_surface):
                xn = x + dx * (i-1)
                yn = y + dy * (i-1)
                zn = z + dz * (i-1)
                break

            # Check if square includes any blocking vertices
            if np.any(blocking):
                break

        square = None

        # Determine vert coords based on search direction
        if xn > x and yn > y and zn == z:
            # vert_type[x:xn+1, y:yn+1, z] = 1 # Type 1 = occupied/exterior
            vert_type[x+1:xn, y+1:yn, z] = 3 # Type 3 = interior/already included in a square

            vx_pos = False
            vx_neg = False
            if z - 1 >= 0:
                vx_neg = (voxel_model_array[x, y, z - 1] != 0)
            if z < z_len-1:
                vx_pos = (voxel_model_array[x, y, z] != 0)

            if vx_pos and not vx_neg: # CW
                square = [[x, y, z],
                          [x, yn, z],
                          [xn, y, z],
                          [xn, yn, z]]
            elif vx_neg and not vx_pos: # CCW
                square = [[x, y, z],
                          [xn, y, z],
                          [x, yn, z],
                          [xn, yn, z]]
            else: # Interior face -- can occur with certain small features
                square = None

        elif xn > x and yn == y and zn > z:
            # vert_type[x:xn+1, y, z:zn+1] = 1 # Type 1 = occupied/exterior
            vert_type[x+1:xn, y, z+1:zn] = 3 # Type 3 = interior/already included in a square

            vx_pos = False
            vx_neg = False
            if y - 1 >= 0:
                vx_neg = (voxel_model_array[x, y - 1, z] != 0)
            if y < y_len-1:
                vx_pos = (voxel_model_array[x, y, z] != 0)

            if vx_pos and not vx_neg:  # CW
                square = [[x, y, z],
                          [xn, y, z],
                          [x, y, zn],
                          [xn, y, zn]]
            elif vx_neg and not vx_pos:  # CCW
                square = [[x, y, z],
                          [x, y, zn],
                          [xn, y, z],
                          [xn, y, zn]]
            else:  # Interior face -- can occur with certain small features
                square = None

        elif xn == x and yn > y and zn > z:
            # vert_type[x, y:yn+1, z:zn+1] = 1 # Type 1 = occupied/exterior
            vert_type[x, y+1:yn, z+1:zn] = 3 # Type 3 = interior/already included in a square

            vx_pos = False
            vx_neg = False
            if x - 1 >= 0:
                vx_neg = (voxel_model_array[x - 1, y, z] != 0)
            if x < x_len-1:
                vx_pos = (voxel_model_array[x, y, z] != 0)

            if vx_pos and not vx_neg:  # CW
                square = [[x, y, z],
                          [x, y, zn],
                          [x, yn, z],
                          [x, yn, zn]]
            elif vx_neg and not vx_pos:  # CCW
                square = [[x, y, z],
                          [x, yn, z],
                          [x, y, zn],
                          [x, yn, zn]]
            else:  # Interior face -- can occur with certain small features
                square = None

        # Add verts, tris, quads, and colors
        if square is not None:
            p = []
            for i in range(len(square)):
                new_vi = vert_index[square[i][0], square[i][1], square[i][2]]
                if new_vi == -1:
                    new_verts.append(square[i])
                    new_colors.append(vert_color[square[i][0], square[i][1], square[i][2]])
                    vert_index[square[i][0], square[i][1], square[i][2]] = vi
                    p.append(vi)
                    vi = vi + 1
                else:
                    p.append(new_vi)

            new_tris.append([p[0], p[1], p[2]])
            new_tris.append([p[3], p[2], p[1]])
            new_quads.append([p[0], p[1], p[3], p[2]])

    return vi, vert_type, vert_index, new_verts, new_colors, new_tris, new_quads