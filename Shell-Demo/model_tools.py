"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import numpy as np

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
# Create a shell around a model
def shell_outside(base_model):
    # Initialize output array
    new_model = np.zeros_like(base_model)
    ones = np.ones((3, 2, 3))

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    # Loop through model data
    for x in range(1, x_len-1):
        for y in range(1, y_len-1):
            for z in range(1, z_len):
                # If voxel is not empty
                if base_model[y, z, x] != 0:
                    new_model[y-1:y+2, z-1:z+1, x-1:x+2] = ones

    new_model = difference(new_model, base_model)
    return new_model

# Create a shell around a model
def shell_inside(base_model):
    # Initialize output array
    new_model = np.zeros_like(base_model)
    ones = np.ones((3, 2, 3))

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    # Loop through model data
    for x in range(1, x_len-1):
        for y in range(1, y_len-1):
            for z in range(0, z_len-1):
                # If voxel is empty
                if base_model[y, z, x] == 0:
                    new_model[y-1:y+2, z:z+2, x-1:x+2] = ones

    new_model = difference(new_model, invert(base_model))
    return new_model