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
    mat = 0

    # Add multiple objects
    for x in range(0,workspaceX, particleSpacing):
        for y in range(0, workspaceY, particleSpacing):
            for z in range(0, workspaceZ, particleSpacing):
                model = model | cube(particleSize, (x+round(z/particleSpacing), y+round(z/particleSpacing), z), (mat % (len(material_properties) - 1)) + 1, 1)
                mat = mat+1

    # Clean up result
    model = model.removeDuplicateMaterials()

    # Create simulation file
    simulation = Simulation(model)
    simulation.collisionEnable = True
    simulation.saveVXA('collision_sim')

    # Create mesh data
    # mesh1 = Mesh.fromVoxelModel(model)

    # Create plot
    # plot1 = Plot(mesh1)
    # plot1.show()
    # app1.processEvents()

    # app1.exec_()