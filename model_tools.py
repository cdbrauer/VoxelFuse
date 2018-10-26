"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import numpy as np
from pyvox.parser import VoxParser

def import_vox(filename):
    # m1 = VoxParser(sys.argv[1]).parse()
    m1 = VoxParser(filename).parse()
    new_model = m1.to_dense()
    new_model = np.flip(new_model, 1)
    return new_model

# Selection operations #############################################################
# Get all voxels with a specified material
def isolate_material(base_model, material):
    new_model = np.copy(base_model)
    new_model[new_model != material] = 0
    return new_model

# Boolean operations, material from first argument takes priority ##################
def union(base_model, model_to_add):
    model_B = np.copy(model_to_add)
    model_B[base_model != 0] = 0
    model_B = model_B + base_model
    return model_B

def difference(base_model, model_to_subtract):
    model_A = np.copy(base_model)
    model_A[model_to_subtract != 0] = 0
    return model_A

def intersection(base_model, model_to_intersect):
    model_A = np.copy(base_model)
    model_A[model_to_intersect == 0] = 0
    return model_A

def invert(base_model):
    model_A = np.copy(base_model)
    model_A[model_A == 0] = 1
    model_A = model_A - base_model
    return model_A

def xor(base_model, model_2):
    model_A = union(base_model, model_2)
    model_B = intersection(base_model, model_2)
    model_A = model_A - model_B
    return model_A

def nor(base_model, model_2):
    model_A = base_model+model_2
    model_A[model_A == 0] = 1
    model_A = model_A - base_model
    model_A = model_A - model_2
    return model_A

# Generation operations ############################################################


# Dilate, erode, and shell operations ##############################################
def dilate(base_model, radius, output_material):
    # Initialize output array
    new_model = np.copy(base_model)

    ones = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    for i in range(radius):
        new_voxels = np.zeros_like(base_model)

        # Loop through model data
        for x in range(1, x_len-1):
            for y in range(1, y_len-1):
                for z in range(1, z_len-1):
                    # If voxel is not empty
                    if new_model[y, z, x] != 0:
                        new_voxels[y-1:y+2, z-1:z+2, x-1:x+2] = new_voxels[y-1:y+2, z-1:z+2, x-1:x+2] + ones

        new_model = new_model+new_voxels

    new_model[new_model != 0] = 1
    new_model = new_model*output_material

    return new_model

def erode(base_model, radius, output_material):
    # Initialize output array
    new_model = np.copy(base_model)

    ones = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    for i in range(radius):
        new_voxels = np.zeros_like(base_model)

        # Loop through model data
        for x in range(1, x_len-1):
            for y in range(1, y_len-1):
                for z in range(1, z_len-1):
                    # If voxel is empty
                    if new_model[y, z, x] == 0:
                        new_voxels[y-1:y+2, z-1:z+2, x-1:x+2] = new_voxels[y-1:y+2, z-1:z+2, x-1:x+2] + ones

        new_model = difference(new_model, new_voxels)

    new_model[new_model != 0] = 1
    new_model = new_model*output_material

    return new_model

# Create a shell around a model
def shell(base_model, thickness, inside_outside, output_material):
    # Initialize output array
    new_model = np.zeros_like(base_model)
    ones = np.ones((1+(2*thickness), 1+(2*thickness), 1+(2*thickness)))

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    # Loop through model data
    for x in range(thickness, x_len-thickness):
        for y in range(thickness, y_len-thickness):
            for z in range(thickness, z_len-thickness):
                # If voxel is empty
                if (inside_outside == 0) and (base_model[y, z, x] == 0):
                    new_model[y-thickness:y+thickness+1, z-thickness:z+thickness+1, x-thickness:x+thickness+1] = ones
                # If voxel is not empty
                elif (inside_outside == 1) and (base_model[y, z, x] != 0):
                    new_model[y-thickness:y+thickness+1, z-thickness:z+thickness+1, x-thickness:x+thickness+1] = ones

    if inside_outside == 0:
        new_model = difference(new_model, invert(base_model))
    else:
        new_model = difference(new_model, base_model)

    new_model = new_model*output_material

    return new_model