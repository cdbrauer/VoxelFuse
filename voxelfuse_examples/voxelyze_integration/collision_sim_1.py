"""
Export a simulation of a single falling object.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import cube, cuboid
from voxelfuse.simulation import Simulation, StopCondition

if __name__=='__main__':
    cubeModel = cube(5, (0, 0, 10), material=5)
    planeModel = cuboid((9, 9, 1), (-2, -2, 0), material=5)

    modelResult = planeModel | cubeModel
    modelResult = modelResult.scale(1)
    modelResult = modelResult.removeDuplicateMaterials()

    simulation = Simulation(modelResult)  # Initialize a simulation
    simulation.setCollision() # Enable self-collision
    # simulation.setStopCondition(StopCondition.TIME_VALUE, 0.01) # Set simulation time limit
    simulation.runSimVoxCad('collision_sim_1', delete_files=False) # Launch simulation, save simulation file
