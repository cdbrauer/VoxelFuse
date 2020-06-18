"""
Export a simulation that applies forces to two simple objects.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import cube
from  voxelfuse.simulation import Simulation

if __name__=='__main__':
    model1C = cube(5, (0, 0, 0), material=6)
    model2C = cube(5, (4, 0, 3), material=6)
    model3C = cube(5, (8, 0, 6), material=6)
    model1H = cube(5, (0, 8, 0), material=9)
    model2H = cube(5, (4, 8, 3), material=9)
    model3H = cube(5, (8, 8, 6), material=9)

    modelResult = model1C | model2C | model3C | model1H | model2H | model3H
    modelResult = modelResult.removeDuplicateMaterials()

    simulation = Simulation(modelResult)  # Initialize a simulation
    simulation.addBoundaryConditionBox() # Add a box boundary condition with default settings (fixed constraint, YZ plane at X=0)
    simulation.addBoundaryConditionBox(position=(0.99, 0, 0), size=(0.01, 0.5, 1.0), fixed_dof=0b111110, force=(30, 0, 0)) # Add a boundary condition at X = max, apply a 30N force
    simulation.addBoundaryConditionBox(position=(0.99, 0.5, 0), size=(0.01, 0.5, 1.0), fixed_dof=0b111110, force=(30, 0, 0)) # Add a boundary condition at X = max, apply a 30N force
    simulation.launchSim()  # Launch simulation
