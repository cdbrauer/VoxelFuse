# Multi-Material Manufacturing Process Planning Tools

<img src="https://github.com/Team-Automata/Multi-Material-Manufacturing-Process-Planning-Tools/blob/master/main.png?raw=true">

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

----------

# VoxelModel

Initialized from a model data array and xyz location coordinates. Each model also stores the number of components present and an array of component labels.

    def __init__(self, model, x_coord = 0, y_coord = 0, z_coord = 0):
        self.model = model
        self.x = x_coord
        self.y = y_coord
        self.z = z_coord
        self.numComponents = 0
        s = np.shape(model)
        self.components = np.zeros(s[:3])

## Class Methods

### fromVoxFile(cls, filename, x = 0, y = 0, z = 0)

Returns a VoxelModel object created from an imported .vox file. x, y, and z arguments will set the import location of the model.

	model1 = VoxelModel.fromVoxFile('cylinder-red.vox', 0, 5, 0)

### fromMeshFile(cls, filename, x = 0, y = 0, z = 0)

Returns a VoxelModel object created from an imported .stl file. x, y, and z arguments will set the import location of the model.

	model1 = VoxelModel.fromMeshFile('center.stl', 67, 3, 0)

### emptyLike(cls, voxel_model)

Returns an empty model with the same size as the input model

### copy(cls, voxel_model)

Returns a copy of a model

## Model Property Updates

Used to update model properties. These operations work directly on the model and do not return anything.

### fitWorkspace(self)

Resizes the workspace around a model to remove excess empty space.  Model coordinates are updated to reflect the change.

### getComponents(self, connectivity=1)

Updates the array of component labels for a model.  Connectivity can be set to 1-3 and defines the shape of the structuring element.

## Isolation/Masking Operations

### isolateMaterial(self, material)

Returns a model with only voxels that contain the specified material. Material input is an index corresponding to the table in materials.py

Example:

	model2 = model1.isolateMaterial(4)

### isolateLayer(self, layer)

Returns a model with only voxels that are included in a specific layer

	model2 = model1.isolateLayer(8)

### isolateComponent(self, component)

Returns a model with only voxels that have the specified component label. Labels must first be updated using the getComponents function.

	model1.getComponents()
    component3 = model1.isolateComponent(3)

### getUnoccupied(self)

Returns a mask of voxels that are unoccupied. A material can be assigned to the mask using the setMaterial or setMaterialVector commands.

### getOccupied(self)

Returns a mask of voxels that are occupied.

### getBoundingBox(self)

Returns a mask of voxels that within the bounding box of a model.

### setMaterial(self, material)

Returns a model with all occupied voxels in the input model set to the specified material index.

	model2 = model1.getBoundingBox()
	model3 = model2.setMaterial(2)

### setMaterialVector(self, material_vector)

Returns a model with all occupied voxels in the input model set to the specified material vector.

	material_vector = np.zeros(len(materials) + 1)	# Length of materials table +1
    material_vector[0] = 1							# Set a to 1
    material_vector[3] = 0.3						# Set material 3 to 30%
    material_vector[4] = 0.7						# Set material 4 to 70%
    model2 = model1.setMaterialVector(material_vector)

## Solid Geometry Operations

...

## Manufacturing Features

...