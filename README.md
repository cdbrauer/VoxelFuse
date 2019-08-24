# Multi-Material Manufacturing Process Planning Tools

<img src="https://raw.githubusercontent.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/master/main.png">

This library provides a set of tools for processing 3D components requiring multiple materials and manufacturing processes.

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at ASU.

## Features
- .vox and .stl file import
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

## Installation

This project uses Python 3 and requires the following libraries:

- numpy
- scipy.ndimage
- pyqt5
- pyqtgraph
- pyopengl
- py-vox-io
- meshio
- numba

These can be installed from the requirements.txt file using pip by running the following command in the project directory:

    pip3 install -r requirements.txt

After cloning the repository, make sure that the following folder is accessible by Python. Depending on your IDE, this can be done by adding it to the system PATH variable or the project path settings.

	C:\<clone directory>\Multi-Material-Manufacturing-Process-Planning-Tools\python\voxelbots

## .vox File Generation
If desired, input models can be created in a .vox file format to allow different materials to be specified in a single model.  This also speeds up import times. My process using [MagicaVoxel](https://ephtracy.github.io) is as follows:

1. Use the "Open" button under the "Palette" section to open the [color-palette-8mat.png](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/raw/master/color-palette-8mat.png) file. This will give you 8 colors that correspond to the materials defined in materials.py
2. Create your model. By default the library will use a scale of 1mm per voxel when importing/exporting.
3. Save the model as a .vox file using the "export" function  (NOT the "save" function).

Using MagicaVoxel and the .vox format will limit you to using distinct voxel materials. The library's import function will convert these files to a data format that allows material mixing.

## Documentation

Please see the [wiki](https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/wiki) for code documentation.

<br/><br/>

<a href="http://idealab.asu.edu/" target="_blank"><img src="https://raw.githubusercontent.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/master/footer.png"/></a>
