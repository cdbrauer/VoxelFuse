# ![VoxelFuse](../master/images/logo.png?raw=true)

The multi-material manufacturing toolkit.

![Example 1](../master/images/main.png?raw=true)

![Example 2](../master/images/main2.png?raw=true)

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
- Voxelyze simulation configuration/execution
- .vf, .vox, and .stl file import
- .vf, .vxc, .vxa, and .stl file export
- .gcode file modification

## Installation

The voxelfuse library can be installed using pip.

    pip3 install voxelfuse

[Gmsh](http://gmsh.info/) is used for .stl file import and Windows/Linux binaries are included with the library.

[VoxCad/Voxelyze](https://sites.google.com/site/voxcadproject/) are used for simulation features. Windows/Linux
binaries for VoxCad are included with the library. A Linux binary for Voxelyze is included which can also be used on
Windows if [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) is configured.

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
        simulation.runSimVoxCad()

## .vox File Generation
If desired, input models can be created in a .vox file format to allow different materials to be specified in a single model.  This also speeds up import times compared to .stl files. My process using [MagicaVoxel](https://ephtracy.github.io) is as follows:

1. Use the "Open" button under the "Palette" section to open the [color-palette-11mat.png](../master/images/color-palette-11mat.png) file. This will give you 11 colors that correspond to the materials defined in materials.py
2. Create your model. By default the library will use a scale of 1mm per voxel when importing/exporting.
3. Save the model as a .vox file using the "export" function (NOT the "save" function).

Using MagicaVoxel and the .vox format will limit you to using distinct voxel materials. The library's import function will convert these files to a data format that allows material mixing.

## Documentation

See [cdbrauer.github.io/VoxelFuse](https://cdbrauer.github.io/VoxelFuse/) for library documentation.

See [cdbrauer.github.io/VoxelFuse/voxelfuse_examples](https://cdbrauer.github.io/VoxelFuse/voxelfuse_examples/) for a list of the example scripts.

<br/><br/>

<a href="http://idealab.asu.edu/" target="_blank">![IDEAlab](../master/images/footer.png?raw=true)</a>
