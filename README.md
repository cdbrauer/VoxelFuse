# Multi-Material Manufacturing Process Planning Tools

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at ASU.

### Working Features
- .vox file import
- Isolation of specific materials
- Conversion of voxel data to mesh surfaces
- Model rendering with grids and axes
- Boolean operations
- Basic shell generation
- .stl file export

### To-do
- Improve functions for generating shells
- Larger workspace or dynamically adjusted workspace size
- Ability to import a model at a specified location
- Improvements to mesh conversion to remove duplicate vertices

## Import Demo
Demonstration of functions for importing, manipulating, and exporting voxel data.

<img width="600" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Import-Demo/voxel-tools-fig1.png">

<img width="600" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Import-Demo/Slic3r-1.png">

## Boolean Functions Demo
Demonstration of boolean functions.
### Union
<img width="300" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Boolean-Functions-Demo/voxel-tools-bool-fig1.png">

### Difference
<img width="300" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Boolean-Functions-Demo/voxel-tools-bool-fig2.png">

### Intersection
<img width="300" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Boolean-Functions-Demo/voxel-tools-bool-fig3.png">

### XOR
<img width="300" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Boolean-Functions-Demo/voxel-tools-bool-fig4.png">

### NOR
<img width="300" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Boolean-Functions-Demo/voxel-tools-bool-fig5.png">

### NOT
<img width="300" src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/Boolean-Functions-Demo/voxel-tools-bool-fig6.png">

## Shell Demo
Demonstrations of function for generating shells, blur function, and dilate/erode functions.


## .vox File Generation
I have been using [MagicaVoxel](https://ephtracy.github.io) to generate voxel files for testing. Files should be created using the "export" function rather than the "save" function. The [py-vox-io](https://github.com/gromgull/py-vox-io) library allows the .vox files to be imported into Python as NumPy arrays.
