"""
Demonstrate temperature keyframes

Copyright 2021 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import cube
from  voxelfuse.simulation import Simulation, StopCondition

if __name__=='__main__':
    model = cube(3, (0, 0, 0), material=7)

    simulation = Simulation(model)  # Initialize a simulation
    simulation.setStopCondition(StopCondition.TIME_VALUE, 30)
    simulation.setGravity()
    simulation.addBoundaryConditionVoxel((1,1,0), fixed_dof=0b111100)

    simulation.addSensor((1, 1, 2))

    for x in range(model.voxels.shape[0]):
        for y in range(model.voxels.shape[1]):
            for z in range(model.voxels.shape[2]):
                simulation.addTempControl((x,y,z))

    simulation.saveVXA('cube_sim')