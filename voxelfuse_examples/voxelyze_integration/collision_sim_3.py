"""
Export multiple simulations of a falling object.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import cube, cuboid
from voxelfuse.simulation import Simulation, StopCondition

if __name__=='__main__':
    cubeModel = cube(5, (0, 0, 4), material=5)
    planeModel = cuboid((9, 9, 1), (-2, -2, 0), material=5)

    modelResult = planeModel | cubeModel
    modelResult = modelResult.scale(1)
    modelResult = modelResult.removeDuplicateMaterials()

    simulation = Simulation(modelResult)  # Initialize a simulation
    simulation.setCollision() # Enable self-collision
    simulation.setDamping(environment=0.1)

    for f in range(4):
        simulation.clearBoundaryConditions()
        simulation.addBoundaryConditionBox()
        simulation.addBoundaryConditionVoxel((2, 2, 8), 0b000000, force=(0, 0, -0.1*f))
        simulation.saveVXA('collision_sim_3_' + str(f)) # Save simulation file