"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
import meshio
from materials import materials

def check_adjacent_x(input_model, x_coord, y_coord, z_coord, x_dir):
    x_len = len(input_model[0, 0, :, 0])

    if x_dir == 1:
        x_coord_new = x_coord+1
    else:
        x_coord_new = x_coord-1

    if (x_coord_new < x_len) and (x_coord_new >= 0) and not np.allclose(input_model[y_coord, z_coord, x_coord_new, :], input_model[y_coord, z_coord, x_coord, :]):
        return True
    elif (x_coord_new >= x_len) or (x_coord_new < 0):
        return True
    else:
        return False

def check_adjacent_y(input_model, x_coord, y_coord, z_coord, y_dir):
    y_len = len(input_model[:, 0, 0, 0])

    if y_dir == 1:
        y_coord_new = y_coord + 1
    else:
        y_coord_new = y_coord - 1

    if (y_coord_new < y_len) and (y_coord_new >= 0) and not np.allclose(input_model[y_coord_new, z_coord, x_coord, :], input_model[y_coord, z_coord, x_coord, :]):
        return True
    elif (y_coord_new >= y_len) or (y_coord_new < 0):
        return True
    else:
        return False

def check_adjacent_z(input_model, x_coord, y_coord, z_coord, z_dir):
    z_len = len(input_model[0, :, 0, 0])

    if z_dir == 1:
        z_coord_new = z_coord + 1
    else:
        z_coord_new = z_coord - 1

    if (z_coord_new < z_len) and (z_coord_new >= 0) and not np.allclose(input_model[y_coord, z_coord_new, x_coord, :], input_model[y_coord, z_coord, x_coord, :]):
        return True
    elif (z_coord_new >= z_len) or (z_coord_new < 0):
        return True
    else:
        return False

"""
Mesh Class

Initialized from a voxel model

Properties:
  verts - coordinates of surface vertices
  colors - color associated with each vertex
  tris - sets of vertices associated with triangular faces
"""
class Mesh:
    # Create mesh from voxel data
    def __init__(self, voxel_model):
        # Get voxel array
        input_model = np.copy(voxel_model.model)

        input_model[input_model < 0] = 0

        # Initialize arrays
        verts = []
        verts_colors = []
        tris = []
        vi = 1  # Tracks current vertex index

        x_len = len(input_model[0, 0, :, 0])
        y_len = len(input_model[:, 0, 0, 0])
        z_len = len(input_model[0, :, 0, 0])

        current_iter = 0
        max_iter = x_len * y_len * z_len

        # Loop through input_model data
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    current_iter = current_iter + 1
                    print("%s / %s" % (current_iter, max_iter))

                    material_count =  sum(input_model[y, z, x, :])

                    # If voxel is not empty
                    if material_count != 0:
                        r = 0
                        g = 0
                        b = 0

                        for i in range(len(materials)):
                            r = r + input_model[y, z, x, i] * materials[i]['r']
                            g = g + input_model[y, z, x, i] * materials[i]['g']
                            b = b + input_model[y, z, x, i] * materials[i]['b']

                        rgbSum = r + g + b

                        r = r / rgbSum #1 if r > 1 else r
                        g = g / rgbSum #1 if g > 1 else g
                        b = b / rgbSum #1 if b > 1 else b

                        a = 1  # if (r + g + b) > 1 else (r + g + b)

                        voxel_color = [r, g, b, a]

                        # Add voxel to mesh item arrays
                        verts_indices = [0, 0, 0, 0, 0, 0, 0, 0]

                        # Add cube vertices
                        if check_adjacent_x(input_model, x, y, z, 1) or check_adjacent_y(input_model, x, y, z, 1) or check_adjacent_z(input_model, x, y, z, 1):
                            verts.append([x + 0.5, y + 0.5, z + 0.5])
                            verts_indices[0] = vi
                            vi = vi + 1

                        if check_adjacent_x(input_model, x, y, z, 1) or check_adjacent_y(input_model, x, y, z, 0) or check_adjacent_z(input_model, x, y, z, 1):
                            verts.append([x + 0.5, y - 0.5, z + 0.5])
                            verts_indices[1] = vi
                            vi = vi + 1

                        if check_adjacent_x(input_model, x, y, z, 0) or check_adjacent_y(input_model, x, y, z, 1) or check_adjacent_z(input_model, x, y, z, 1):
                            verts.append([x - 0.5, y + 0.5, z + 0.5])
                            verts_indices[2] = vi
                            vi = vi + 1

                        if check_adjacent_x(input_model, x, y, z, 0) or check_adjacent_y(input_model, x, y, z, 0) or check_adjacent_z(input_model, x, y, z, 1):
                            verts.append([x - 0.5, y - 0.5, z + 0.5])
                            verts_indices[3] = vi
                            vi = vi + 1

                        if check_adjacent_x(input_model, x, y, z, 1) or check_adjacent_y(input_model, x, y, z, 1) or check_adjacent_z(input_model, x, y, z, 0):
                            verts.append([x + 0.5, y + 0.5, z - 0.5])
                            verts_indices[4] = vi
                            vi = vi + 1

                        if check_adjacent_x(input_model, x, y, z, 1) or check_adjacent_y(input_model, x, y, z, 0) or check_adjacent_z(input_model, x, y, z, 0):
                            verts.append([x + 0.5, y - 0.5, z - 0.5])
                            verts_indices[5] = vi
                            vi = vi + 1

                        if check_adjacent_x(input_model, x, y, z, 0) or check_adjacent_y(input_model, x, y, z, 1) or check_adjacent_z(input_model, x, y, z, 0):
                            verts.append([x - 0.5, y + 0.5, z - 0.5])
                            verts_indices[6] = vi
                            vi = vi + 1

                        if check_adjacent_x(input_model, x, y, z, 0) or check_adjacent_y(input_model, x, y, z, 0) or check_adjacent_z(input_model, x, y, z, 0):
                            verts.append([x - 0.5, y - 0.5, z - 0.5])
                            verts_indices[7] = vi
                            vi = vi + 1

                        # Apply color to all vertices
                        for i in range(0, np.count_nonzero(verts_indices)):
                            verts_colors.append(voxel_color)

                        # Add face triangles
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

        verts = np.array(verts)
        verts_colors = np.array(verts_colors)
        tris = np.array(tris)

        self.verts = verts
        self.colors = verts_colors
        self.tris = tris
        self.model = input_model

    # Export model from mesh data
    def export(self, filename):
        cells = {
            "triangle": self.tris
        }

        output_mesh = meshio.Mesh(self.verts, cells)
        meshio.write(filename, output_mesh)