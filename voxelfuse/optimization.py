"""
Functions for performing various optimization algorithms.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import numpy as np
from enum import Enum
from typing import List
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.simulation import Simulation
from voxelfuse.materials import material_properties

class ResultType(Enum):
    """
    Options for simulation result measurements.
    """
    FACTOR_OF_SAFETY = 'SafetyFactor'
    STRESS = 'BondStress'
    PRESSURE = 'Pressure'

def optimizeTopology(simulation, condition: ResultType, value: float, removal_tolerance: float = 0, max_iter: int = 10, protected = None):
    """
    Remove material from a model until a threshold value is reached.

    This currently works only for measurements that must stay ABOVE the threshold (i.e. FOS).  Measurements that
    must stay BELOW the threshold (e.g. stress, pressure) still need to be added.

    :param simulation: Simulation object with desired loads and constraints
    :param condition: Measurement to monitor, set with ResultType class
    :param value: Measurement threshold value
    :param removal_tolerance: Sets how close voxels must be to the max/min value in order to be removed
    :param max_iter: Maximum number of material removal iterations
    :param protected: Material types that should not be removed
    :return: VoxelModel
    """
    if protected is None:
        protected = []

    # Copy simulation and model
    new_sim = Simulation.copy(simulation)
    new_model = VoxelModel.copy(simulation.getModel())

    x_len = len(new_model.voxels[:, 0, 0])
    y_len = len(new_model.voxels[0, :, 0])
    z_len = len(new_model.voxels[0, 0, :])

    firstIter = True
    last_max = 0.0
    last_min = 0.0

    for i in range(max_iter):
        # Clear any existing sensors
        new_sim.clearSensors()

        # Add sensors to all voxels that are not empty or protected
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    mat = new_model.voxels[x, y, z]
                    if mat != 0 and mat not in protected:
                        new_sim.addSensor((x, y, z))

        # Run simulation
        new_sim.runSim(voxelyze_on_path=True)

        # Generate array of result values
        result_values = np.zeros_like(new_model.voxels)
        for s in range(len(new_sim.results)):
            x = new_sim.results[s]['Location'][0]
            y = new_sim.results[s]['Location'][1]
            z = new_sim.results[s]['Location'][2]
            try:
                result_values[x, y, z] = new_sim.results[s][condition.value]
            except OverflowError:
                print('Overflow error')

        # Find max/min value
        max_result_value = np.max(result_values[np.nonzero(result_values)])
        min_result_value = np.min(result_values[np.nonzero(result_values)])

        if firstIter:
            firstIter = False
        else:
            max_result_value = min(max_result_value, last_max)
            min_result_value = min(min_result_value, last_min)

        print('Min value: ' + str(min_result_value))
        print('Max value: ' + str(max_result_value))

        # If max/min value has not reached the limiting value
        if min_result_value > value:
            new_model.voxels[result_values >= (max_result_value - removal_tolerance)] = 0 # Remove voxel with min/max value
            new_sim.setModel(new_model) # Update simulation
        else:
            print('Goal condition reached in ' + str(i+1) + ' iterations')
            break

        if i+1 == max_iter:
            print('Max iterations reached')

        last_max = max_result_value
        last_min = min_result_value

    return new_model