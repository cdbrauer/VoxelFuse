"""
Export multiple simulations of a falling object.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import cube, cuboid
from voxelfuse.simulation import Simulation, StopCondition

if __name__=='__main__':
    # Generate primitives
    cubeModel = cube(5, (0, 0, 4), material=5)
    planeModel = cuboid((9, 9, 1), (-2, -2, 0), material=5)

    # Combine primitives
    modelResult = planeModel | cubeModel
    modelResult = modelResult.scale(1)
    modelResult = modelResult.removeDuplicateMaterials()

    # Initialize a simulation
    simulation = Simulation(modelResult)
    simulation.setCollision() # Enable self-collision
    simulation.setDamping(environment=0.1) # Viscous fluid environment
    simulation.addSensor((2, 2, 8)) # Sensor at top center of cube
    simulation.setStopCondition(StopCondition.MOTION_FLOOR, 1e-05)

    # Lists to store results
    position = []
    stress = []

    # Run simulation with 3 different force levels
    for f in range(1,4):
        simulation.clearBoundaryConditions() # Clear any previous boundary conditions
        simulation.addBoundaryConditionBox() # Fix bottom layer to prevent sliding
        simulation.addBoundaryConditionVoxel((2, 2, 8), 0b000000, force=(0, 0, -0.1*f)) # Apply a force to the top center voxel
        simulation.runSim() # Run simulation
        position.append(simulation.results[0]['Position']) # Save position reading from sensor
        stress.append(simulation.results[0]['BondStress']) # Save stress reading from sensor

    # Print results
    print('----------')
    print('Position: ' + str(position))
    print('Stress: ' + str(stress))
