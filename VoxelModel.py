"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
from pyvox.parser import VoxParser
from materials import materials

"""
VoxelModel Class

Initialized from a model array or file and position coordinates

Properties:
  model - array storing the material type present at each voxel
  x_coord, y_coord, z_coord - position of model origin
"""
class VoxelModel:
    def __init__(self, model, x_coord = 0, y_coord = 0, z_coord = 0):
        self.model = model
        self.x = x_coord
        self.y = y_coord
        self.z = z_coord

    @classmethod
    def fromFile(cls, filename, x_coord = 0, y_coord = 0, z_coord = 0):
        m1 = VoxParser(filename).parse()
        m2 = m1.to_dense()
        m2 = np.flip(m2, 1)

        x_len = len(m2[0, 0, :])
        y_len = len(m2[:, 0, 0])
        z_len = len(m2[0, :, 0])

        new_model = np.zeros((y_len, z_len, x_len, len(materials)))

        # Loop through input_model data
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    color_index = m2[y, z, x] - 1
                    if (color_index >= 0) and (color_index < len(materials)):
                        new_model[y, z, x, color_index] = 1
                    elif color_index >= len(materials):
                        print('Unrecognized material index: ' + str(color_index))

        return cls(new_model, x_coord, y_coord, z_coord)

    def union(self, model_to_add):
        new_model = self.model + model_to_add.model
        new_model[new_model > 1] = 1
        return VoxelModel(new_model)

    def __add__(self, other):
        return self.union(other)

    def difference(self, model_to_sub):
        new_model = self.model - model_to_sub.model
        new_model[new_model < 0] = 0
        return VoxelModel(new_model)

    def __sub__(self, other):
        return self.difference(other)
