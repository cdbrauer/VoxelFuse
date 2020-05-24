"""
Copyright 2020
Dan Aukes, Cole Brauer
"""

import PyQt5.QtGui as qg
import sys
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.simulation import Simulation
from voxelfuse.mesh import Mesh
from voxelfuse.plot import Plot
from voxelfuse.primitives import cube
from voxelfuse.materials import material_properties

workspaceX = 10
workspaceY = 10
workspaceZ = 25

particleSize = 3
particleSpacing = 5

if __name__=='__main__':
    # app1 = qg.QApplication(sys.argv)

    # Initialize an empty model
    model = VoxelModel.empty((workspaceX, workspaceY, workspaceZ), 1)
    mat = 1

    # Add multiple objects, changing the material for each
    for x in range(0,workspaceX, particleSpacing):
        for y in range(0, workspaceY, particleSpacing):
            for z in range(0, workspaceZ, particleSpacing):
                model = model | cube(particleSize, (x+round(z/particleSpacing), y+round(z/particleSpacing), z), (mat % (len(material_properties) - 1)) + 1, 1) # Slightly offset x and y so stacks will fall over
                mat = mat+1 # Increment material

    # Clean up result
    model = model.removeDuplicateMaterials()

    # Create simulation file
    simulation = Simulation(model) # Initialize a simulation
    simulation.setCollision() # Enable self collisions with default settings
    simulation.addBoundaryConditionBox(position=(0.99,0,0)) # Add a boundary condition at (x = max, y = 0, z = 0), leave other settings at default (fixed constraint, YZ plane)
    simulation.launchSim('collision_sim', delete_files=False) # Launch simulation, save simulation file