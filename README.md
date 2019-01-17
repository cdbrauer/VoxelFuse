# Multi-Material Manufacturing Process Planning Tools

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at ASU.

### Required Libraries
- numpy
- pyqt5
- pyqtgraph
- pyopengl
- py-vox-io
- meshio

### Working Features
- .vox file import
- Isolation of specific materials
- Conversion of voxel data to mesh surfaces
- Model rendering with grids and axes
- Boolean operations
- Dilate and Erode Operations
- Gaussian Blurring
- .stl file export

### To-do
- Larger workspace or dynamically adjusted workspace size
- Ability to import a model at a specified location
- Improvements to mesh conversion to remove duplicate vertices

## Import Demo
Demonstration of functions for importing, manipulating, and exporting voxel data.

<img src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Import-Demo/import-export.png?raw=true">

## Boolean Functions Demo
Demonstration of boolean functions.

<img src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Boolean-Functions-Demo/boolean-functions.png?raw=true">

## Shell Demo
Demonstrations of blur function and dilate/erode functions.


## .vox File Generation
I have been using [MagicaVoxel](https://ephtracy.github.io) to generate voxel files for testing. Files should be created using the "export" function rather than the "save" function. The [py-vox-io](https://github.com/gromgull/py-vox-io) library allows the .vox files to be imported into Python as NumPy arrays.
