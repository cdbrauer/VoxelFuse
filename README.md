# Multi-Material Manufacturing Process Planning Tools

<img src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/main.png?raw=true">

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
- Model rendering with grids and axes
- Conversion of voxel data to mesh surfaces
- Isolation of specific materials and layers
- Boolean operations for both volumes and materials
- Dilate and Erode Operations
- Gaussian Blurring
- .stl file export

### To-do
- Dithering option
- Improvements to mesh conversion to remove duplicate vertices

## .vox File Generation
I have been using [MagicaVoxel](https://ephtracy.github.io) to generate voxel files for testing. Files should be created using the "export" function rather than the "save" function. The [py-vox-io](https://github.com/gromgull/py-vox-io) library allows the .vox files to be imported into Python as NumPy arrays.
