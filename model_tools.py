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

# Get all voxels in a specified layer
def isolate_layer(base_model, layer):
    new_model = np.zeros_like(base_model)
    new_model[:, layer, :] = base_model[:, layer, :]
    return new_model

# Get R, G, and B color matrices from a model
def separate_colors(base_model):
    new_model = np.copy(base_model).astype(int)
    new_model[new_model != 0] = new_model[new_model != 0] - 1

    b = (new_model % 5)
    g = (((new_model - b) / 5) % 5)
    r = (((new_model - (g * 5) - b) / 25) % 5)

    b = b / 4.0
    g = g / 4.0
    r = r / 4.0

    return r, g, b

# Convert R, G, and B color matrices to a model
def combine_colors(r, g, b):
    r = (r * 4).astype(int)
    g = (g * 4).astype(int)
    b = (b * 4).astype(int)

    new_model = ((25*r) + (5*g) + b)
    new_model[new_model != 0] = new_model[new_model != 0] + 1

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

# Blur, Dilate, and Erode #########################################################
def blur(base_model, threshold = 0.0):
    # Initialize output arrays
    new_r = np.zeros_like(base_model).astype(float)
    new_g = np.zeros_like(base_model).astype(float)
    new_b = np.zeros_like(base_model).astype(float)

    kernel = np.array([[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                       [[2.0, 4.0, 2.0], [4.0, 8.0, 4.0], [2.0, 4.0, 2.0]],
                       [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]) * (1.0/64.0)

    r, g, b = separate_colors(base_model)

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    # Loop through model data
    for x in range(1, x_len-1):
        for y in range(1, y_len-1):
            for z in range(1, z_len-1):
                new_r[y, z, x] = np.sum(np.multiply(r[y-1:y+2, z-1:z+2, x-1:x+2], kernel))
                new_g[y, z, x] = np.sum(np.multiply(g[y-1:y+2, z-1:z+2, x-1:x+2], kernel))
                new_b[y, z, x] = np.sum(np.multiply(b[y-1:y+2, z-1:z+2, x-1:x+2], kernel))

    new_model = combine_colors(new_r, new_g, new_b)
    new_model_brightness = (new_r + new_g + new_b)

    if threshold != 0: # Input zero for no threshold effect
        new_model[new_model_brightness < threshold] = 0

    return new_model

def dilate(base_model, radius, effect = 'overlap'):
    # Initialize output arrays
    new_r = np.zeros_like(base_model).astype(float)
    new_g = np.zeros_like(base_model).astype(float)
    new_b = np.zeros_like(base_model).astype(float)

    r, g, b = separate_colors(base_model)

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    for i in range(radius):
        for x in range(1, x_len-1):
            for y in range(1, y_len-1):
                for z in range(1, z_len-1):
                    new_r[y, z, x] = np.max(r[y-1:y+2, z-1:z+2, x-1:x+2])
                    new_g[y, z, x] = np.max(g[y-1:y+2, z-1:z+2, x-1:x+2])
                    new_b[y, z, x] = np.max(b[y-1:y+2, z-1:z+2, x-1:x+2])

        r = np.copy(new_r)
        g = np.copy(new_g)
        b = np.copy(new_b)

    if effect == 'blur':
        new_model_brightness = (r + g + b)

        kernel = np.array([[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                           [[2.0, 4.0, 2.0], [4.0, 8.0, 4.0], [2.0, 4.0, 2.0]],
                           [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]) * (1.0 / 64.0)

        # Loop through model data
        for x in range(1, x_len - 1):
            for y in range(1, y_len - 1):
                for z in range(1, z_len - 1):
                    if new_model_brightness[y, z, x] > 1:
                        new_r[y, z, x] = np.sum(np.multiply(r[y - 1:y + 2, z - 1:z + 2, x - 1:x + 2], kernel))
                        new_g[y, z, x] = np.sum(np.multiply(g[y - 1:y + 2, z - 1:z + 2, x - 1:x + 2], kernel))
                        new_b[y, z, x] = np.sum(np.multiply(b[y - 1:y + 2, z - 1:z + 2, x - 1:x + 2], kernel))

        r = np.copy(new_r)
        g = np.copy(new_g)
        b = np.copy(new_b)

    if effect == 'avg' or effect == 'blur':
        new_model_brightness = (r + g + b)

        # Loop through model data
        for x in range(1, x_len - 1):
            for y in range(1, y_len - 1):
                for z in range(1, z_len - 1):
                    if new_model_brightness[y, z, x] > 1:
                        new_r[y, z, x] = (r[y, z, x] / new_model_brightness[y, z, x])
                        new_g[y, z, x] = (g[y, z, x] / new_model_brightness[y, z, x])
                        new_b[y, z, x] = (b[y, z, x] / new_model_brightness[y, z, x])

        r = np.copy(new_r)
        g = np.copy(new_g)
        b = np.copy(new_b)

    new_model = combine_colors(r, g, b)

    return new_model

def erode(base_model, radius):
    # Initialize output arrays
    new_r = np.zeros_like(base_model).astype(float)
    new_g = np.zeros_like(base_model).astype(float)
    new_b = np.zeros_like(base_model).astype(float)

    r, g, b = separate_colors(base_model)

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    for i in range(radius):
        for x in range(1, x_len-1):
            for y in range(1, y_len-1):
                for z in range(1, z_len-1):
                    new_r[y, z, x] = np.min(r[y-1:y+2, z-1:z+2, x-1:x+2])
                    new_g[y, z, x] = np.min(g[y-1:y+2, z-1:z+2, x-1:x+2])
                    new_b[y, z, x] = np.min(b[y-1:y+2, z-1:z+2, x-1:x+2])

        r = np.copy(new_r)
        g = np.copy(new_g)
        b = np.copy(new_b)

    new_model = combine_colors(r, g, b)

    return new_model

# Generation functions #############################################################
def bounding_box(base_model):
    new_model = np.zeros_like(base_model)

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    x_min = -1
    x_max = -1
    y_min = -1
    y_max = -1
    z_min = -1
    z_max = -1

    # Loop through model data
    for x in range(0, x_len):
        if x_min == -1:
            if np.sum(base_model[:, :, x]) > 0:
                x_min = x
        else:
            if np.sum(base_model[:, :, x]) == 0:
                x_max = x
                break

    for y in range(0, y_len):
        if y_min == -1:
            if np.sum(base_model[y, :, :]) > 0:
                y_min = y
        else:
            if np.sum(base_model[y, :, :]) == 0:
                y_max = y
                break

    for z in range(0, z_len):
        if z_min == -1:
            if np.sum(base_model[:, z, :]) > 0:
                z_min = z
        else:
            if np.sum(base_model[:, z, :]) == 0:
                z_max = z
                break

    new_model[y_min:y_max, z_min:z_max, x_min:x_max].fill(1)

    return new_model

def keepout(base_model, method):
    new_model = np.zeros_like(base_model)

    x_len = len(base_model[0, 0, :])
    y_len = len(base_model[:, 0, 0])
    z_len = len(base_model[0, :, 0])

    if method == 'laser':
        # Loop through model data
        for x in range(0, x_len):
            for y in range(0, y_len):
                if np.sum(base_model[y, :, x]) > 0:
                    new_model[y, :, x] = np.ones(z_len)

        for z in range(z_len-1, -1, -1):
            if np.sum(base_model[:, z, :]) == 0:
                new_model[:, z, :].fill(0)
            else:
                break

    elif method == 'mill':
        # Loop through model data
        for x in range(0, x_len):
            for y in range(0, y_len):
                for z in range(0, z_len):
                    if np.sum(base_model[y, z:, x]) > 0:
                        new_model[y, z, x] = 1
                    elif np.sum(base_model[y, z:, x]) == 0:
                        break

    return new_model

def clearance(base_model, method):
    # Initialize output array
    new_model = np.zeros_like(base_model)

    Kl = keepout(base_model, 'laser')
    Km = keepout(base_model, 'mill')

    if method == 'laser':
        new_model = difference(Kl, base_model)
    elif method == 'mill':
        new_model = difference(Kl, Km)

    return new_model

def web(base_model, method, layer, r1=1, r2=1):
    if method == 'laser':
        new_model = isolate_layer(keepout(base_model, 'laser'), layer)*2

    elif method == 'mill':
        new_model = isolate_layer(keepout(base_model, 'mill'), layer)*2

    model_B = isolate_layer(dilate(new_model, r1), layer)
    model_C = isolate_layer(dilate(model_B, r2), layer)
    model_D = bounding_box(model_C)
    new_model = difference(model_D, model_B)

    return new_model