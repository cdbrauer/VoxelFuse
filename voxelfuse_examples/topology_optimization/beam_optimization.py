"""
Perform topology optimization on a simple beam model.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

# Import Libraries
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.simulation import Simulation, StopCondition
from voxelfuse.optimization import optimizeTopology, ResultType

# Start Application
if __name__ == '__main__':
    # Import Model
    modelName = 'beam1'
    beamModel = VoxelModel.fromVoxFile(modelName + '.vox')
    beamModel = beamModel.translate((0, 0, 15))

    # Create simulation
    simulation = Simulation(beamModel)

    # Set simulation stop conditions
    simulation.setEquilibriumMode()
    simulation.setStopCondition(StopCondition.MOTION_FLOOR, 1e-05)

    # Apply loads and constraints
    simulation.addBoundaryConditionBox(position=(0, 0, 0), size=(0.01, 1, 1))
    simulation.addBoundaryConditionBox(position=(0.99, 0, 0), size=(0.01, 1, 1), fixed_dof=0b000000, force=(0, 0, -10))

    # Run optimization
    newBeamModel = optimizeTopology(simulation, ResultType.FACTOR_OF_SAFETY, 15, removal_tolerance=10, max_iter=1, protected=[1])

    # Export updated model
    simulation.setModel(newBeamModel)
    simulation.setEquilibriumMode(False)
    simulation.setStopCondition(StopCondition.NONE)
    simulation.launchSim(modelName, delete_files=False)