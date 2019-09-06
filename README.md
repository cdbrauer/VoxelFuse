# ![VoxelFuse](../master/images/logo.png?raw=true)

The multi-material manufacturing toolkit.

![Example 1](../master/images/main.png?raw=true)

![Example 2](../master/images/main2.png?raw=true)

VoxelFuse provides a set of Python commands for importing, modifying, displaying, and exporting multi-material 3D model data.  This library allows scripts to be quickly created for processing different classes of models and generating the files needed to produce them.

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at ASU.

## Features
- .vox and .stl file import
- Isolation of specific materials and layers
- Boolean operations for both volumes and materials
- Dilate and Erode Operations
- Gaussian Blurring
- Conversion of voxel data to mesh surfaces
- Model rendering with grids and axes
- .stl file export
- .gcode file modification

## Installation

The voxelfuse library can be installed using pip.

    pip3 install voxelfuse

To use the .stl file import commands/examples, [Gmsh](http://gmsh.info/) must also be installed.

## .vox File Generation
If desired, input models can be created in a .vox file format to allow different materials to be specified in a single model.  This also speeds up import times. My process using [MagicaVoxel](https://ephtracy.github.io) is as follows:

1. Use the "Open" button under the "Palette" section to open the [color-palette-8mat.png](../master/images/color-palette-8mat.png) file. This will give you 8 colors that correspond to the materials defined in materials.py
2. Create your model. By default the library will use a scale of 1mm per voxel when importing/exporting.
3. Save the model as a .vox file using the "export" function  (NOT the "save" function).

Using MagicaVoxel and the .vox format will limit you to using distinct voxel materials. The library's import function will convert these files to a data format that allows material mixing.

## Documentation

Please see the [wiki](https://github.com/cdbrauer/VoxelFuse/wiki) for code documentation.

<br/><br/>

<a href="http://idealab.asu.edu/" target="_blank">![IDEAlab](../master/images/footer.png?raw=true)</a>
