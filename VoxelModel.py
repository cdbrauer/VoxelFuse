"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
from pyvox.parser import VoxParser

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
        new_model = m1.to_dense()
        new_model = np.flip(new_model, 1)
        return cls(new_model, x_coord, y_coord, z_coord)

    def union(self, model_to_add):
        new_model = np.copy(model_to_add.model)
        new_model[self.model != 0] = 0
        new_model = new_model + self.model
        return VoxelModel(new_model)

    def __add__(self, other):
        return self.union(other)

    def difference(self, model_to_sub):
        new_model = np.copy(self.model)
        new_model[model_to_sub.model != 0] = 0
        return VoxelModel(new_model)

    def __sub__(self, other):
        return self.difference(other)
