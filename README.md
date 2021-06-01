# ![VoxelFuse](images/logo.png?raw=true)

VoxelFuse is a Python library for processing multi-material 3D model data. It includes the tools needed for a complete workflow from design to part fabrication, including part composition, manufacturing planning, design simulation, mesh generation, and file export. This library allows scripts to be quickly created for processing different classes of models and generating the files needed to produce them.  

Created as part of a research project with [IDEAlab](http://idealab.asu.edu) at [ASU](https://www.asu.edu/).

![Workflow](images/workflow.png?raw=true)

## Example Applications

![Example 1](images/embedded-joint.png?raw=true)

*Design of a part with embedded electronic components*

![Example 2](images/blurring.png?raw=true)

*Generation of blurred material transitions*

## Features

### Model Composition

- Primitive solid generation
- Triply periodic structure generation
- Mesh file import
- Import of the .vox voxel file format  
- Import of native .vf file format 

![Primitive solids](images/primitives.png?raw=true)

### Model Modification 
  
- Boolean operations for both volumes and materials
- Transformation operations
- Morphology operations
- Gaussian blurring
- Dithering

![Example CSG operations](images/csg.png?raw=true)

### Manufacturing Planning

![Manufacturing Features](images/manufacturing-features.png?raw=true)

### Simulation

- [VoxCad](https://www.creativemachineslab.com/voxcad.html) simulation configuration
- Integrated simulation engine based on [Voxelyze](https://github.com/jonhiller/Voxelyze)
- Simulation features include stress analysis, physics, and thermal actuation
- Automated execution of individual and multiprocess simulation tasks
- Logging of "sensor" voxels and model position throughout simulation
- Export to .vxc and .vxa files

### Mesh Generation

- Conversion of voxel faces to mesh surfaces
- Conversion of voxel data to a mesh using a marching cubes algorithm
- Mesh simplification
- Mesh rendering with grids and axes using PyQt and OpenGL
- Mesh plotting in [Jupyter Notebook](https://jupyter.org/)
- Mesh file export

![Mesh Examples](images/mesh_types.png?raw=true)

## Installation

The voxelfuse library can be installed using pip.

    pip3 install voxelfuse

[Gmsh](http://gmsh.info/) is used for mesh file import and Windows/Linux binaries are included with the library.

[VoxCad](https://www.creativemachineslab.com/voxcad.html) is used for running interactive simulations in a GUI. Windows/Linux binaries are included with the library.

A custom version of [Voxelyze](https://github.com/jonhiller/Voxelyze) is used for most simulation features. Windows/Linux binaries are included with the library. To use multiprocess simulation features on Windows, [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) must be configured.

[voxcraft-viz](https://github.com/voxcraft/voxcraft-viz) can be used to view simulation history files (must be installed separately).

[Jupyter Notebook](https://jupyter.org/) is used for some examples.

## Templates

Base template for creating scripts:

```python
# Import Library
import voxelfuse as vf

# Start Application
if __name__=='__main__':
    # Create Models
    model = vf.sphere(5)

    # Process Models
    modelResult = model.dilate(3, vf.Axes.XY)

    # Create and Export Mesh
    mesh = vf.Mesh.fromVoxelModel(modelResult)
    mesh.export('modelResult.stl')

    # Create Plot
    mesh.viewer(grids=True, name='mesh')
```

Template for creating scripts using VoxCad simulation:

```python
# Import Library
import voxelfuse as vf

# Start Application
if __name__=='__main__':
    # Create Models
    model = vf.sphere(5)

    # Process Models
    modelResult = model.dilate(3, vf.Axes.XY)
    modelResult = modelResult.translate((0, 0, 20))

    # Create simulation and launch
    simulation = vf.Simulation(modelResult)
    simulation.runSimVoxCad()
```

## Usage

See [cdbrauer.github.io/VoxelFuse](https://cdbrauer.github.io/VoxelFuse/) for library documentation.

See [cdbrauer.github.io/VoxelFuse/voxelfuse_examples](https://cdbrauer.github.io/VoxelFuse/voxelfuse_examples/)
for a list of the example scripts.

### .vox File Generation
If desired, input models can be created in a .vox file format to allow different materials to be specified in a single model.  This also speeds up import times compared to mesh files. My process using [MagicaVoxel](https://ephtracy.github.io) is as follows:

1. Use the "Open" button under the "Palette" section to open the [color-palette.png](../master/images/color-palette.png) file. This will give you a set of colors that correspond to the materials defined in materials.py
2. Create your model. By default, the library will use a scale of 1mm per voxel when importing/exporting, but this can be changed if necessary.
3. Save the model as a .vox file using the "export" function (NOT the "save" function).

Using MagicaVoxel and the .vox format will limit you to using distinct voxel materials. The library's import function will convert these files to a data format that allows material mixing.

## Papers

For more information about our work, please see:

> Brauer, C., & Aukes, D. M. (2020). Automated Generation of Multi-Material Structures Using the VoxelFuse Framework. Symposium on Computational Fabrication, 1–8. https://doi.org/10.1145/3424630.3425417

> Brauer, C. (2020). Automated Design of Graded Material Transitions for Educational Robotics Applications [PQDT-Global]. https://search.proquest.com/openview/3be6eafdf193c7b7271ccf714d51da9d

> Brauer, C., & Aukes, D. M. (2019). Voxel-Based CAD Framework for Planning Functionally Graded and Multi-Step Rapid Fabrication Processes. Volume 2A: 45th Design Automation Conference, 2A-2019. https://doi.org/10.1115/DETC2019-98103

To cite this library, consider using:

> Brauer, C., Aukes, D., Brauer, J., & Jeffries, C. (2020). VoxelFuse. https://github.com/cdbrauer/VoxelFuse

<br/><br/>

<a href="http://idealab.asu.edu/" target="_blank">![IDEAlab](../master/images/footer.png?raw=true)</a>
