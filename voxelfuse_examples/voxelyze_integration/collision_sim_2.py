"""
Export a simulation of multiple falling objects.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.simulation import Simulation
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import empty, cube
from voxelfuse.materials import material_properties

workspaceX = 10
workspaceY = 10
workspaceZ = 25

particleSize = 3
particleSpacing = 5

if __name__=='__main__':
    # Initialize an empty model
    model = empty()
    mat = 1

    # Add multiple objects, changing the material for each
    for x in range(0,workspaceX, particleSpacing):
        for y in range(0, workspaceY, particleSpacing):
            for z in range(0, workspaceZ, particleSpacing):
                model = model | cube(particleSize, (x+round(z/particleSpacing), y+round(z/particleSpacing), z+5), mat, 1) # Slightly offset x and y so stacks will fall over
                mat = (mat % 9) + 1 # Increment material, note that the vxa file format is limited to 9 distinct materials.

    # Clean up result
    model = model.fitWorkspace() # Using this to show that Simulation correctly handles models with a coordinate offset
    model = model.removeDuplicateMaterials()

    # Create simulation file
    simulation = Simulation(model) # Initialize a simulation
    simulation.setCollision() # Enable self collisions with default settings
    simulation.addBoundaryConditionBox() # Add a box boundary with default settings (fixed constraint, YZ plane at X=0)
    simulation.addBoundaryConditionBox(position=(0.99, 0, 0)) # Add a boundary condition at x = max, leave other settings at default (fixed constraint, YZ plane)

    # Add some forces and sensors
    simulation.addSensor((3, 9, 16))
    simulation.addSensor((4, 10, 12))

    simulation.runSimVoxCad('collision_sim_2', delete_files=False) # Launch simulation, save simulation file