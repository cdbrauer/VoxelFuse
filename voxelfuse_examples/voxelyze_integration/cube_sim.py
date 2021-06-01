"""
Demonstrate temperature keyframes

Copyright 2021 - Cole Brauer, Dan Aukes
"""

import numpy as np
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import cube
from  voxelfuse.simulation import Simulation, StopCondition

if __name__=='__main__':
    model = cube(3, (0, 0, 0), material=7) | cube(3, (3, 0, 0), material=10)

    simulation = Simulation(model)  # Initialize a simulation
    simulation.setStopCondition(StopCondition.TIME_VALUE, 30)
    simulation.setGravity()
    simulation.addBoundaryConditionVoxel((1,1,0), fixed_dof=0b111100, name='Bottom Center')
    simulation.addSensor((1, 1, 2), name='Top Center')

    locs1 = [(1,1,0),
             (1,1,1)]

    locs2 = [(1,0,0),
             (0,1,0),
             (2,1,0),
             (1,2,0)]

    simulation.addTempControlGroup(locs1, 'Locations 1')
    simulation.addKeyframe(0, 15, temp_offset=15)
    simulation.addKeyframe(5, 30, const_temp=True)
    simulation.addTempControlGroup(locs2, 'Locations 2')
    simulation.addKeyframe(10, 20, square_wave=True)
    simulation.saveVXA('cube_sim_1')

    simulation.clearTempControlGroups()
    simulation.addTempControlGroup(name='All Voxels')
    simulation.addKeyframe(0, 15, temp_offset=15)
    simulation.addKeyframe(5, 30, period=3.0, const_temp=True)
    simulation.saveVXA('cube_sim_2')

    simulation.initializeTempMap()
    occupied = model.getOccupied().voxels
    occupied[2, 1, 0] = 0
    simulation.applyTempMap(0, np.multiply(occupied, 10), const_temp_map=occupied)
    simulation.applyTempMap(10, np.multiply(occupied, 20), const_temp_map=occupied)
    simulation.saveVXA('cube_sim_3')