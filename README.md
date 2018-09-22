# Multi-Material Manufacturing Process Planning Tools

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at ASU.

## voxel-tools.py
Demonstration of functions for manipulating voxel data. Should be run with a path to a .vox file as only parameter.

### Working
- .vox file import
- Isolation of specific materials
- Conversion of voxel data to mesh cubes
- Model rendering with grids and axes

### To-do
- Boolean operations (add, subtract, intersect)
- Function to import a model at a specified location

## .vox File Generation
I have been using [MagicaVoxel](https://ephtracy.github.io) to generate voxel files for testing. Files should be created using the "export" function rather than the "save" function. The [py-vox-io](https://github.com/gromgull/py-vox-io) library allows the .vox files to be imported into Python as NumPy arrays.
