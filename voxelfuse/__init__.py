# -*- coding: utf-8 -*-
"""
----

# VoxelFuse

The multi-material manufacturing toolkit

VoxelFuse provides a set of Python commands for importing, modifying, displaying, and exporting multi-material 3D model data.  This library allows scripts to be quickly created for processing different classes of models and generating the files needed to produce them.

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at ASU.

## Features

- Boolean operations for both volumes and materials
- Morphology operations
- Gaussian blurring
- Primitive solid generation
- Triply periodic structure generation
- Conversion of voxel data to mesh surfaces
- Model rendering with grids and axes
- VoxCad simulation configuration
- .vf, .vox, and .stl file import
- .vf, .vxc, .vxa, and .stl file export
- .gcode file modification

## Installation

The voxelfuse library can be installed using pip.

    pip3 install voxelfuse

To use the .stl file import commands/examples, [Gmsh](http://gmsh.info/) must also be installed and on the system path.

To use the simulation commands/examples, [VoxCad](https://sites.google.com/site/voxcadproject/) must also be installed and on the system path.

## Templates

Template for creating scripts using the built-in viewer:

    # Import Libraries
    import PyQt5.QtGui as qg
    import sys
    from voxelfuse.voxel_model import VoxelModel
    from voxelfuse.mesh import Mesh
    from voxelfuse.plot import Plot

    # Start Application
    if __name__=='__main__':
        app1 = qg.QApplication(sys.argv)

        # Import Models
        modelIn = VoxelModel.fromVoxFile('model1.vox')

        # Process Models
        modelResult = modelIn

        # Create and Export Mesh
        mesh1 = Mesh.fromVoxelModel(modelResult)
        mesh1.export('modelResult.stl')

        # Create and Export Plot
        plot1 = Plot(mesh1)
        plot1.show()
        app1.processEvents()
        plot1.export('result.png')

        app1.exec_()

Template for creating scripts using VoxCad simulation:

    # Import Libraries
    from voxelfuse.voxel_model import VoxelModel
    from voxelfuse.simulation import Simulation

    # Start Application
    if __name__=='__main__':
        # Import Models
        modelIn = VoxelModel.fromVoxFile('model1.vox')

        # Process Models
        modelResult = modelIn

        # Create simulation and launch
        simulation = Simulation(modelIn)
        simulation.launchSim()

## Examples

See [cdbrauer.github.io/VoxelFuse/voxelfuse_examples](https://cdbrauer.github.io/VoxelFuse/voxelfuse_examples/)
for a list of the example scripts.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

name = "voxelfuse"
__version__ = "1.2.5"