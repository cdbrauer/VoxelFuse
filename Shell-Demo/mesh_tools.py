"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import numpy as np

# Create mesh from voxel data
def create_from_model(input_model):
    # Initialize arrays
    verts = []
    verts_colors = []
    tris = []
    vi = 0  # Tracks starting index for defining triangles

    x_len = len(input_model[0, 0, :])
    y_len = len(input_model[:, 0, 0])
    z_len = len(input_model[0, :, 0])

    # Loop through input_model data
    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                # If voxel is not empty
                if input_model[y, z, x] != 0:
                    # Set color based on material
                    if input_model[y, z, x] == 236:  # Blue
                        voxel_color = [0, 0, 1, 1]
                    elif input_model[y, z, x] == 217:  # Red
                        voxel_color = [1, 0, 0, 1]
                    elif input_model[y, z, x] == 226:  # Green
                        voxel_color = [0, 1, 0, 1]
                    else:  # Yellow - default if material not recognized
                        voxel_color = [1, 1, 0, 1]

                    # Add voxel to mesh item arrays
                    # Add cube vertices
                    verts.append([x + 0.5, y + 0.5, z + 0.5])
                    verts.append([x + 0.5, y - 0.5, z + 0.5])
                    verts.append([x - 0.5, y + 0.5, z + 0.5])
                    verts.append([x - 0.5, y - 0.5, z + 0.5])
                    verts.append([x + 0.5, y + 0.5, z - 0.5])
                    verts.append([x + 0.5, y - 0.5, z - 0.5])
                    verts.append([x - 0.5, y + 0.5, z - 0.5])
                    verts.append([x - 0.5, y - 0.5, z - 0.5])

                    # Apply color to all vertices
                    for i in range(0, 8):
                        verts_colors.append(voxel_color)

                    # Add face triangles
                    tris.append([vi + 0, vi + 1, vi + 2])
                    tris.append([vi + 1, vi + 2, vi + 3])
                    tris.append([vi + 0, vi + 1, vi + 4])
                    tris.append([vi + 1, vi + 4, vi + 5])
                    tris.append([vi + 0, vi + 2, vi + 4])
                    tris.append([vi + 2, vi + 4, vi + 6])
                    tris.append([vi + 2, vi + 3, vi + 6])
                    tris.append([vi + 3, vi + 6, vi + 7])
                    tris.append([vi + 1, vi + 3, vi + 5])
                    tris.append([vi + 3, vi + 5, vi + 7])
                    tris.append([vi + 4, vi + 5, vi + 6])
                    tris.append([vi + 5, vi + 6, vi + 7])

                    # Increment index by 8 vertices
                    vi = vi + 8

    verts = np.array(verts)
    verts_colors = np.array(verts_colors)
    tris = np.array(tris)

    return verts, verts_colors, tris