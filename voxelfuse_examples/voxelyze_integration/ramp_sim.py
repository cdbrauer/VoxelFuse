"""
START HERE: Run a simulation of a ball rolling down a ramp.

----

Copyright 2022 - Cole Brauer
"""

import voxelfuse as vf

if __name__=='__main__':
    # Generate primitives
    sphereModel = vf.sphere(3, (0, 0, 15), material=5) # Using rubber because soft materials simulate faster
    rampModel = vf.prism((30, 10, 10), -15, (-5, -5, 0), material=5)

    # Combine primitives
    modelResult = sphereModel | rampModel
    modelResult = modelResult.scale(1)
    modelResult = modelResult.removeDuplicateMaterials()

    # Initialize a simulation
    simulation = vf.Simulation(modelResult)
    simulation.setGravity()
    simulation.setCollision() # Enable self-collision
    simulation.setStopCondition(vf.StopCondition.TIME_VALUE, 0.6) # Run for 0.6 seconds

    # Constraints
    simulation.addBoundaryConditionBox() # Default options lock voxels touching the ground plane

    # Sensors
    simulation.addSensor((0, 0, 15)) # Sensor in sphere

    # Run simulation with UI -- collisions must be manually enabled from the UI
    # simulation.runSimVoxCad()

    # Run simulation in background -- WSL should be disabled if on Windows and WSL is not installed
    simulation.runSim('ramp', delete_files=False, log_interval=1000, history_interval=5000, wsl=True)
    print(simulation.results[0]['Position'])
