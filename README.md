# Multi-Material Manufacturing Process Planning Tools

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at ASU.

## voxel-tools.py
Demonstration of functions for manipulating voxel data. Should be run with a path to a .vox file as only parameter.

### Working
- .vox file import
- Isolation of specific materials
- Conversion of voxel data to mesh cubes
- Model rendering with grids and axes
- Boolean operations

### To-do
- Function for generating shells
- Function to import a model at a specified location

![Sample output](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/voxel-tools-fig1.png "Sample output")

## voxel-tools-bool.py
Demonstration of boolean functions.  Should have paths to two .vox files as parameters.
### Union
![Union](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/voxel-tools-bool-fig1.png "Union")
### Difference
![Difference](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/voxel-tools-bool-fig2.png "Difference")
### Intersection
![Intersection](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/voxel-tools-bool-fig3.png "Intersection")
### XOR
![XOR](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/voxel-tools-bool-fig4.png "XOR")
### NOR
![NOR](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/voxel-tools-bool-fig5.png "NOR")
### NOT
![NOT](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/voxel-tools-bool-fig6.png "NOT")

## .vox File Generation
I have been using [MagicaVoxel](https://ephtracy.github.io) to generate voxel files for testing. Files should be created using the "export" function rather than the "save" function. The [py-vox-io](https://github.com/gromgull/py-vox-io) library allows the .vox files to be imported into Python as NumPy arrays.
