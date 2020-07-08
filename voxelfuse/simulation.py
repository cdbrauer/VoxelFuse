"""
Simulation Class

Initialized from a VoxelModel object. Used to configure VoxCad and Voxelyze simulations.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import os
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
from typing import Tuple, TextIO
from tqdm import tqdm
import numpy as np
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.primitives import empty, cuboid, sphere, cylinder

# Floating point error threshold for rounding to zero
FLOATING_ERROR = 0.0000000001

class StopCondition(Enum):
    """
    Options for simulation stop conditions.
    """
    NONE = 0
    TIME_STEP = 1
    TIME_VALUE = 2
    TEMP_CYCLES = 3
    ENERGY_CONST = 4
    ENERGY_KFLOOR = 5
    MOTION_FLOOR = 6

class BCShape(Enum):
    """
    Options for simulation boundary condition shapes.
    """
    BOX = 0
    CYLINDER = 1
    SPHERE = 2

class Simulation:
    """
    Simulation object that stores a VoxelModel and its associated simulation settings.
    """

    def __init__(self, voxel_model):
        """
        Initialize a Simulation object with default settings.

        Models located at positive coordinate values will have their workspace
        size adjusted to maintain their position in the exported simulation.
        Models located at negative coordinate values will be shifted to the origin.

        :param voxel_model: VoxelModel
        """
        # Fit workspace and union with an empty object at the origin to clear offsets if object is raised
        self.__model = (VoxelModel.copy(voxel_model).fitWorkspace()) | empty()

        # Simulator ##############
        # Integration
        self.__integrator = 0
        self.__dtFraction = 1.0

        # Damping
        self.__dampingBond = 1.0 # (0-1) Bulk material damping
        self.__dampingEnvironment = 0.0001 # (0-0.1) Damping caused by fluid environment

        # Collisions
        self.__collisionEnable = False
        self.__collisionDamping = 1.0 # (0-2) Elastic vs inelastic conditions
        self.__collisionSystem = 3
        self.__collisionHorizon = 3

        # Features
        self.__blendingEnable = False
        self.__xMixRadius = 0
        self.__yMixRadius = 0
        self.__zMixRadius = 0
        self.__blendingModel = 0
        self.__polyExp = 1
        self.__volumeEffectsEnable = False

        # Stop conditions
        self.__stopConditionType = StopCondition.NONE
        self.__stopConditionValue = 0.0

        # Equilibrium mode
        self.__equilibriumModeEnable = False

        # Environment ############
        # Boundary conditions
        self.__bcRegions = []
        self.__bcVoxels = []

        # Gravity
        self.__gravityEnable = True
        self.__gravityValue = -9.81
        self.__floorEnable = True

        # Thermal
        self.__temperatureEnable = False
        self.__temperatureBaseValue = 25.0
        self.__temperatureVaryEnable = False
        self.__temperatureVaryAmplitude = 0.0
        self.__temperatureVaryPeriod = 0.0

        # Sensors #######
        self.__sensors = []

        # Temperature Controls #######
        self.__tempControls = []

        # Results ################
        self.results = []
        self.valueMap = np.zeros_like(voxel_model.voxels, dtype=np.float32)

    @classmethod
    def copy(cls, simulation):
        """
        Create new Simulation object with the same settings as an existing Simulation object.

        :param simulation: Simulation to copy
        :return: Simulation
        """
        # Create new simulation object and copy attribute values
        new_simulation = cls(simulation.__model)
        new_simulation.__dict__ = simulation.__dict__.copy()

        # Make lists copies instead of references
        new_simulation.__bcRegions = simulation.__bcRegions.copy()
        new_simulation.__bcVoxels = simulation.__bcVoxels.copy()
        new_simulation.__sensors = simulation.__sensors.copy()

        return new_simulation

    # Configure settings ##################################
    def setModel(self, voxel_model):
        """
        Set the model for a simulation.

        Models located at positive coordinate values will have their workspace
        size adjusted to maintain their position in the exported simulation.
        Models located at negative coordinate values will be shifted to the origin.

        :param voxel_model: VoxelModel
        :return: None
        """
        # Fit workspace and union with an empty object at the origin to clear offsets if object is raised
        self.__model = (VoxelModel.copy(voxel_model).fitWorkspace()) | empty()

    def setDamping(self, bond: float = 1.0, environment: float = 0.0001):
        """
        Set simulation damping parameters.

        Environment damping can be used to simulate fluid environments. 0 represents
        a vacuum and larger values represent a viscous fluid.

        :param bond: Voxel bond damping (0-1)
        :param environment: Environment damping (0-0.1)
        :return: None
        """
        self.__dampingBond = bond
        self.__dampingEnvironment = environment

    def setCollision(self, enable: bool = True, damping: float = 1.0):
        """
        Set simulation collision parameters.

        A damping value of 0 represents completely elastic collisions and
        higher values represent inelastic collisions.

        :param enable: Enable/disable collisions
        :param damping: Collision damping (0-2)
        :return: None
        """
        self.__collisionEnable = enable
        self.__collisionDamping = damping

    def setStopCondition(self, condition: StopCondition = StopCondition.NONE, value: float = 0):
        """
        Set simulation stop condition.

        :param condition: Stop condition type, set using StopCondition class
        :param value: Stop condition value
        :return: None
        """
        self.__stopConditionType = condition
        self.__stopConditionValue = value

    def setEquilibriumMode(self, enable: bool = True):
        """
        Set simulation equilibrium mode.

        :param enable: Enable/disable equilibrium mode
        :return: None
        """
        self.__equilibriumModeEnable = enable

    def setGravity(self, enable: bool = True, value: float = -9.81, enable_floor: bool = True):
        """
        Set simulation gravity parameters.

        :param enable: Enable/disable gravity
        :param value: Acceleration due to gravity in m/sec^2
        :param enable_floor: Enable/disable ground plane
        :return: None
        """
        self.__gravityEnable = enable
        self.__gravityValue = value
        self.__floorEnable = enable_floor

    def setFixedThermal(self, enable: bool = True, base_temp: float = 25.0):
        """
        Set a fixed environment temperature.

        :param enable: Enable/disable temperature
        :param base_temp: Temperature in degrees C
        :return: None
        """
        self.__temperatureEnable = enable
        self.__temperatureBaseValue = base_temp
        self.__temperatureVaryEnable = False

    def setVaryingThermal(self, enable: bool = True, base_temp: float = 25.0, amplitude: float = 0.0, period: float = 0.0):
        """
        Set a varying environment temperature.

        :param enable: Enable/disable temperature
        :param base_temp: Base temperature in degrees C
        :param amplitude: Temperature fluctuation amplitude
        :param period: Temperature fluctuation period
        :return: None
        """
        self.__temperatureEnable = enable
        self.__temperatureBaseValue = base_temp
        self.__temperatureVaryEnable = enable
        self.__temperatureVaryAmplitude = amplitude
        self.__temperatureVaryPeriod = period

    # Read settings ##################################
    def getModel(self):
        """
        Get the simulation model.

        :return: VoxelModel
        """
        return self.__model

    def getDamping(self):
        """
        Get simulation damping parameters.

        :return: Voxel bond damping, Environment damping
        """
        return self.__dampingBond, self.__dampingEnvironment

    def getCollision(self):
        """
        Get simulation collision parameters.

        :return: Enable/disable collisions, Collision damping
        """
        return self.__collisionEnable, self.__collisionDamping

    def getStopCondition(self):
        """
        Get simulation stop condition.

        :return: Stop condition type, Stop condition value
        """
        return self.__stopConditionType, self.__stopConditionValue

    def getEquilibriumMode(self):
        """
        Get simulation equilibrium mode.

        :return: Enable/disable equilibrium mode
        """
        return self.__equilibriumModeEnable

    def getGravity(self):
        """
        Get simulation gravity parameters.

        :return: Enable/disable gravity, Acceleration due to gravity in m/sec^2, Enable/disable ground plane
        """
        return self.__gravityEnable, self.__gravityValue, self.__floorEnable

    def getThermal(self):
        """
        Get simulation temperature parameters.

        :return: Enable/disable temperature, Base temperature in degrees C, Enable/disable temperature fluctuation, Temperature fluctuation amplitude, Temperature fluctuation period
        """
        return self.__temperatureEnable, self.__temperatureBaseValue, self.__temperatureVaryEnable, self.__temperatureVaryAmplitude, self.__temperatureVaryPeriod

    # Add forces, constraints, and sensors ##################################
    # Boundary condition sizes and positions are expressed as percentages of the overall model size
    #   radius is a percentage of the largest model dimension
    # Fixed DOF bits correspond to: Rz, Ry, Rx, Z, Y, X
    #   0: Free, force will be applied
    #   1: Fixed, displacement will be applied
    # Displacement is expressed in mm

    def clearBoundaryConditions(self):
        """
        Remove all boundary conditions from a Simulation object.

        :return: None
        """
        self.__bcRegions = []
        self.__bcVoxels = []

    # Default box boundary condition is a fixed constraint in the YZ plane
    def addBoundaryConditionVoxel(self, position: Tuple[int, int, int] = (0, 0, 0),
                                  fixed_dof: int = 0b111111,
                                  force: Tuple[float, float, float] = (0, 0, 0),
                                  displacement: Tuple[float, float, float] = (0, 0, 0),
                                  torque: Tuple[float, float, float] = (0, 0, 0),
                                  angular_displacement: Tuple[float, float, float] = (0, 0, 0)):
        """
        Add a boundary condition at a specific voxel.

        The fixed DOF value should be set as a 6-bit binary value (e.g. 0b111111) and the bits
        correspond to: Rz, Ry, Rx, Z, Y, X. If a bit is set to 0, the corresponding force/torque
        will be applied. If a bit is set to 1, the DOF will be fixed and the displacement will be
        applied.

        :param position: Position in voxels
        :param fixed_dof: Fixed degrees of freedom
        :param force: Force vector in N
        :param displacement: Displacement vector in mm
        :param torque: Torque values in Nm
        :param angular_displacement: Angular displacement values in deg
        :return: None
        """
        x = position[0] - self.__model.coords[0]
        y = position[1] - self.__model.coords[1]
        z = position[2] - self.__model.coords[2]

        x_len = int(self.__model.voxels.shape[0])
        y_len = int(self.__model.voxels.shape[1])
        z_len = int(self.__model.voxels.shape[2])

        pos = ((x+0.5)/x_len, (y+0.5)/y_len, (z+0.5)/z_len)
        radius = 0.49/x_len

        self.__bcRegions.append([BCShape.SPHERE, pos, (0.0, 0.0, 0.0), radius, (0.6, 0.4, 0.4, .5), fixed_dof, force, torque, displacement, angular_displacement])
        self.__bcVoxels.append([x, y, z])

    # Default box boundary condition is a fixed constraint in the XY plane (bottom layer)
    def addBoundaryConditionBox(self, position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                size: Tuple[float, float, float] = (1.0, 1.0, 0.01),
                                fixed_dof: int = 0b111111,
                                force: Tuple[float, float, float] = (0, 0, 0),
                                displacement: Tuple[float, float, float] = (0, 0, 0),
                                torque: Tuple[float, float, float] = (0, 0, 0),
                                angular_displacement: Tuple[float, float, float] = (0, 0, 0)):
        """
        Add a box-shaped boundary condition.

        Boundary condition position and size are expressed as percentages of the
        overall model size. The fixed DOF value should be set as a 6-bit binary
        value (e.g. 0b111111) and the bits correspond to: Rz, Ry, Rx, Z, Y, X.
        If a bit is set to 0, the corresponding force/torque will be applied. If
        a bit is set to 1, the DOF will be fixed and the displacement will be
        applied.

        :param position: Box corner position (0-1)
        :param size: Box size (0-1)
        :param fixed_dof: Fixed degrees of freedom
        :param force: Force vector in N
        :param displacement: Displacement vector in mm
        :param torque: Torque values in Nm
        :param angular_displacement: Angular displacement values in deg
        :return: None
        """
        self.__bcRegions.append([BCShape.BOX, position, size, 0, (0.6, 0.4, 0.4, .5), fixed_dof, force, torque, displacement, angular_displacement])

        x_len = int(self.__model.voxels.shape[0])
        y_len = int(self.__model.voxels.shape[1])
        z_len = int(self.__model.voxels.shape[2])

        regionSize = np.ceil([size[0]*x_len, size[1]*y_len, size[2]*z_len]).astype(np.int32)
        regionPosition = np.floor([position[0] * x_len + self.__model.coords[0], position[1] * y_len + self.__model.coords[1], position[2] * z_len + self.__model.coords[2]]).astype(np.int32)
        bcRegion = cuboid(regionSize, regionPosition) & self.__model

        x_offset = int(bcRegion.coords[0])
        y_offset = int(bcRegion.coords[1])
        z_offset = int(bcRegion.coords[2])

        bcVoxels = []
        for x in tqdm(range(x_len), desc='Finding constrained voxels'):
            for y in range(y_len):
                for z in range(z_len):
                    if bcRegion.voxels[x, y, z] != 0:
                        bcVoxels.append([x+x_offset, y+y_offset, z+z_offset])

        self.__bcVoxels.append(bcVoxels)

    # Default sphere boundary condition is a fixed constraint centered in the model
    def addBoundaryConditionSphere(self, position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                                   radius: float = 0.05,
                                   fixed_dof: int = 0b111111,
                                   force: Tuple[float, float, float] = (0, 0, 0),
                                   displacement: Tuple[float, float, float] = (0, 0, 0),
                                   torque: Tuple[float, float, float] = (0, 0, 0),
                                   angular_displacement: Tuple[float, float, float] = (0, 0, 0)):
        """
        Add a spherical boundary condition.

        Boundary condition position and radius are expressed as percentages of the
        overall model size. The fixed DOF value should be set as a 6-bit binary
        value (e.g. 0b111111) and the bits correspond to: Rz, Ry, Rx, Z, Y, X.
        If a bit is set to 0, the corresponding force/torque will be applied. If
        a bit is set to 1, the DOF will be fixed and the displacement will be
        applied.

        :param position: Sphere center position (0-1)
        :param radius: Sphere radius (0-1)
        :param fixed_dof: Fixed degrees of freedom
        :param force: Force vector in N
        :param displacement: Displacement vector in mm
        :param torque: Torque values in Nm
        :param angular_displacement: Angular displacement values in deg
        :return: None
        """
        self.__bcRegions.append([BCShape.SPHERE, position, (0.0, 0.0, 0.0), radius, (0.6, 0.4, 0.4, .5), fixed_dof, force, torque, displacement, angular_displacement])

        x_len = int(self.__model.voxels.shape[0])
        y_len = int(self.__model.voxels.shape[1])
        z_len = int(self.__model.voxels.shape[2])

        regionRadius = np.ceil(np.max([x_len, y_len, z_len]) * radius).astype(np.int32)
        regionPosition = np.floor([position[0] * x_len + self.__model.coords[0], position[1] * y_len + self.__model.coords[1], position[2] * z_len + self.__model.coords[2]]).astype(np.int32)
        bcRegion = sphere(regionRadius, regionPosition) & self.__model

        x_offset = int(bcRegion.coords[0])
        y_offset = int(bcRegion.coords[1])
        z_offset = int(bcRegion.coords[2])

        bcVoxels = []
        for x in tqdm(range(x_len), desc='Finding constrained voxels'):
            for y in range(y_len):
                for z in range(z_len):
                    if bcRegion.voxels[x, y, z] != 0:
                        bcVoxels.append([x+x_offset, y+y_offset, z+z_offset])

        self.__bcVoxels.append(bcVoxels)

    # Default cylinder boundary condition is a fixed constraint centered in the model
    def addBoundaryConditionCylinder(self, position: Tuple[float, float, float] = (0.45, 0.5, 0.5), axis: int = 0,
                                     height: float = 0.1,
                                     radius: float = 0.05,
                                     fixed_dof: int = 0b111111,
                                     force: Tuple[float, float, float] = (0, 0, 0),
                                     displacement: Tuple[float, float, float] = (0, 0, 0),
                                     torque: Tuple[float, float, float] = (0, 0, 0),
                                     angular_displacement: Tuple[float, float, float] = (0, 0, 0)):
        """
        Add a cylindrical boundary condition.

        Boundary condition position and size are expressed as percentages of the
        overall model size. The fixed DOF value should be set as a 6-bit binary
        value (e.g. 0b111111) and the bits correspond to: Rz, Ry, Rx, Z, Y, X.
        If a bit is set to 0, the corresponding force/torque will be applied. If
        a bit is set to 1, the DOF will be fixed and the displacement will be
        applied.

        :param position: Boundary condition origin position (0-1)
        :param axis: Cylinder axis (0-2)
        :param height: Cylinder height (0-1)
        :param radius: Cylinder radius (0-1)
        :param fixed_dof: Fixed degrees of freedom
        :param force: Force vector in N
        :param displacement: Displacement vector in mm
        :param torque: Torque values in Nm
        :param angular_displacement: Angular displacement values in deg
        :return: None
        """
        size = [0.0, 0.0, 0.0]
        size[axis] = height
        self.__bcRegions.append([BCShape.CYLINDER, position, tuple(size), radius, (0.6, 0.4, 0.4, .5), fixed_dof, force, torque, displacement, angular_displacement])

        x_len = int(self.__model.voxels.shape[0])
        y_len = int(self.__model.voxels.shape[1])
        z_len = int(self.__model.voxels.shape[2])

        regionRadius = np.ceil(np.max([x_len, y_len, z_len]) * radius).astype(np.int32)
        regionHeight = np.ceil(int(self.__model.voxels.shape[axis] * height))
        regionPosition = np.floor([position[0] * x_len + self.__model.coords[0], position[1] * y_len + self.__model.coords[1], position[2] * z_len + self.__model.coords[2]]).astype(np.int32)
        bcRegion = cylinder(regionRadius, regionHeight, regionPosition)

        if axis == 0:
            bcRegion = bcRegion.rotate90(axis=1)
        elif axis == 1:
            bcRegion = bcRegion.rotate90(axis=0)

        bcRegion = bcRegion & self.__model

        x_offset = int(bcRegion.coords[0])
        y_offset = int(bcRegion.coords[1])
        z_offset = int(bcRegion.coords[2])

        bcVoxels = []
        for x in tqdm(range(x_len), desc='Finding constrained voxels'):
            for y in range(y_len):
                for z in range(z_len):
                    if bcRegion.voxels[x, y, z] != 0:
                        bcVoxels.append([x+x_offset, y+y_offset, z+z_offset])

        self.__bcVoxels.append(bcVoxels)

    def clearSensors(self):
        """
        Remove all sensors from a Simulation object.

        :return: None
        """
        self.__sensors = []

    def addSensor(self, location: Tuple[int, int, int] = (0, 0, 0)):
        """
        Add a sensor to a voxel.

        This feature is not currently supported by VoxCad

        :param location: Sensor location in voxels
        :return: None
        """
        x = location[0] - self.__model.coords[0]
        y = location[1] - self.__model.coords[1]
        z = location[2] - self.__model.coords[2]

        sensor = [x, y, z]
        self.__sensors.append(sensor)

    def clearTempControls(self):
        """
        Remove all temperature control elements from a Simulation object.

        :return: None
        """
        self.__tempControls = []

    def addTempControl(self, location: Tuple[int, int, int] = (0, 0, 0), temperature: float = 0):
        """
        Add a temperature control element to a voxel.

        This feature is not currently supported by VoxCad

        :param location: Control element location in voxels
        :param temperature: Control element temperature
        :return: None
        """
        x = location[0] - self.__model.coords[0]
        y = location[1] - self.__model.coords[1]
        z = location[2] - self.__model.coords[2]

        element = [x, y, z, temperature]
        self.__tempControls.append(element)

    def applyTempMap(self, valueMap):
        """
        Set the simulation temperature control elements based on a value map of target temperatures.

        :param valueMap: Array of target temperatures for each voxel
        :return:
        """

        # Clear any existing temp controls
        self.clearTempControls()

        # Get map size
        x_len = valueMap.shape[0]
        y_len = valueMap.shape[1]
        z_len = valueMap.shape[2]

        # Find required temperature change at each voxel
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    if abs(valueMap[x, y, z]) > FLOATING_ERROR:  # If voxel is not empty
                        self.addTempControl((x, y, z), valueMap[x, y, z])

    def applyVolumeMap(self, valueMap = None):
        """
        Set the simulation temperature control elements based on a value map of target volumes.

        If a valueMap is not specified, the Simulation object's value map attribute will
        be used. To update this attribute, first use ``runSim(value_map=10)`` to get the
        result volumes from running the simulation with any previous settings.

        :param valueMap: Array of target volumes for each voxel
        :return:
        """
        if valueMap is None:
            valueMap = self.valueMap

        # Clear any existing temp controls
        self.clearTempControls()

        # Get map size
        x_len = valueMap.shape[0]
        y_len = valueMap.shape[1]
        z_len = valueMap.shape[2]

        # Get initial volume
        v0 = (1 / self.__model.resolution) ** 3

        # Find required temperature change at each voxel
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    if abs(valueMap[x, y, z]) > FLOATING_ERROR:  # If voxel is not empty
                        # Get volume change
                        vol_delta = valueMap[x, y, z] - v0

                        # Get CTE value
                        avgProps = self.__model.getVoxelProperties((x, y, z))
                        cte = avgProps['CTE']

                        # Get required temperature change relative to base temperature
                        temp_delta = (vol_delta / (v0 * cte))

                        # Get base temperature
                        temp_base = (self.getThermal())[1]

                        # Add a temperature control element
                        self.addTempControl((x, y, z), temp_base+temp_delta)

    def saveTempControls(self, filename: str, figure: bool = False):
        """
        Save the temperature control elements applied to a model to a .csv file.

        :param filename: File name
        :param figure: Enable/disable exporting a figure as well
        :return: None
        """
        f = open(filename + '.csv', 'w+')
        print('Saving file: ' + f.name)
        f.write('X,Y,Z,Temperature (deg C)\n')
        for i in range(len(self.__tempControls)):
            f.write(str(self.__tempControls[i]).replace('[', '').replace(' ', '').replace(']', '') + '\n')
        f.close()

        if figure:
            # Get plot data
            points = np.array(self.__tempControls)
            xs = points[:, 0]
            ys = points[:, 1]
            zs = points[:, 2]
            temps = points[:, 3]
            colors = np.array(abs((temps - np.min(temps)) / (np.max(temps) - np.min(temps))), dtype=np.str)  # Grayscale range

            # Plot results
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.scatter(zs, ys, c=colors, marker='s')
            ax1.axis('equal')
            ax1.set_title('Side')
            ax2 = fig.add_subplot(122)
            ax2.scatter(xs, ys, c=colors, marker='s')
            ax2.axis('equal')
            ax2.set_title('Top')

            # Save figure
            print('Saving file: ' + filename + '.png')
            plt.savefig(filename + '.png')

    # Export simulation ##################################
    # Export simulation object to .vxa file for import into VoxCad or Voxelyze
    def saveVXA(self, filename: str, compression: bool = False):
        """
        Save model data to a .vxa file

        The VoxCad simulation file format stores all the data contained in
        a .vxc file (geometry, material palette) plus the simulation setup (simulation
        parameters, environment settings, boundary conditions).

        This format supports compression for the voxel data. Enabling compression allows
        for larger models, but it may introduce geometry errors that particularly affect
        small models.

        The .vxa file type can be opened using VoxCad simulation software. However, it
        cannot currently be reopened by a VoxelFuse script.

        :param filename: File name
        :param compression: Enable/disable voxel data compression
        :return: None
        """
        f = open(filename + '.vxa', 'w+')
        print('Saving file: ' + f.name)

        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        f.write('<VXA Version="' + str(1.1) + '">\n')
        self.writeSimData(f)
        self.writeEnvironmentData(f)
        self.writeSensors(f)
        self.writeTempControls(f)
        self.__model.writeVXCData(f, compression)
        f.write('</VXA>\n')

        f.close()

    # Write simulator settings to file
    def writeSimData(self, f: TextIO):
        """
        Write simulation parameters to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        # Simulator settings
        f.write('<Simulator>\n')
        f.write('  <Integration>\n')
        f.write('    <Integrator>' + str(self.__integrator) + '</Integrator>\n')
        f.write('    <DtFrac>' + str(self.__dtFraction) + '</DtFrac>\n')
        f.write('  </Integration>\n')
        f.write('  <Damping>\n')
        f.write('    <BondDampingZ>' + str(self.__dampingBond) + '</BondDampingZ>\n')
        f.write('    <ColDampingZ>' + str(self.__collisionDamping) + '</ColDampingZ>\n')
        f.write('    <SlowDampingZ>' + str(self.__dampingEnvironment) + '</SlowDampingZ>\n')
        f.write('  </Damping>\n')
        f.write('  <Collisions>\n')
        f.write('    <SelfColEnabled>' + str(int(self.__collisionEnable)) + '</SelfColEnabled>\n')
        f.write('    <ColSystem>' + str(self.__collisionSystem) + '</ColSystem>\n')
        f.write('    <CollisionHorizon>' + str(self.__collisionHorizon) + '</CollisionHorizon>\n')
        f.write('  </Collisions>\n')
        f.write('  <Features>\n')
        f.write('    <BlendingEnabled>' + str(int(self.__blendingEnable)) + '</BlendingEnabled>\n')
        f.write('    <XMixRadius>' + str(self.__xMixRadius) + '</XMixRadius>\n')
        f.write('    <YMixRadius>' + str(self.__yMixRadius) + '</YMixRadius>\n')
        f.write('    <ZMixRadius>' + str(self.__zMixRadius) + '</ZMixRadius>\n')
        f.write('    <BlendModel>' + str(self.__blendingModel) + '</BlendModel>\n')
        f.write('    <PolyExp>' + str(self.__polyExp) + '</PolyExp>\n')
        f.write('    <VolumeEffectsEnabled>' + str(int(self.__volumeEffectsEnable)) + '</VolumeEffectsEnabled>\n')
        f.write('  </Features>\n')
        f.write('  <StopCondition>\n')
        f.write('    <StopConditionType>' + str(self.__stopConditionType.value) + '</StopConditionType>\n')
        f.write('    <StopConditionValue>' + str(self.__stopConditionValue) + '</StopConditionValue>\n')
        f.write('  </StopCondition>\n')
        f.write('  <EquilibriumMode>\n')
        f.write('    <EquilibriumModeEnabled>' + str(int(self.__equilibriumModeEnable)) + '</EquilibriumModeEnabled>\n')
        f.write('  </EquilibriumMode>\n')
        f.write('</Simulator>\n')

    # Write environment settings to file
    def writeEnvironmentData(self, f: TextIO):
        """
        Write simulation environment parameters to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        # Environment settings
        f.write('<Environment>\n')
        f.write('  <Boundary_Conditions>\n')
        f.write('    <NumBCs>' + str(len(self.__bcRegions)) + '</NumBCs>\n')

        for r in tqdm(range(len(self.__bcRegions)), desc='Writing boundary conditions'):
            f.write('    <FRegion>\n')
            f.write('      <PrimType>' + str(int(self.__bcRegions[r][0].value)) + '</PrimType>\n')
            f.write('      <X>' + str(self.__bcRegions[r][1][0]) + '</X>\n')
            f.write('      <Y>' + str(self.__bcRegions[r][1][1]) + '</Y>\n')
            f.write('      <Z>' + str(self.__bcRegions[r][1][2]) + '</Z>\n')
            f.write('      <dX>' + str(self.__bcRegions[r][2][0]) + '</dX>\n')
            f.write('      <dY>' + str(self.__bcRegions[r][2][1]) + '</dY>\n')
            f.write('      <dZ>' + str(self.__bcRegions[r][2][2]) + '</dZ>\n')
            f.write('      <Radius>' + str(self.__bcRegions[r][3]) + '</Radius>\n')
            f.write('      <R>' + str(self.__bcRegions[r][4][0]) + '</R>\n')
            f.write('      <G>' + str(self.__bcRegions[r][4][1]) + '</G>\n')
            f.write('      <B>' + str(self.__bcRegions[r][4][2]) + '</B>\n')
            f.write('      <alpha>' + str(self.__bcRegions[r][4][3]) + '</alpha>\n')
            f.write('      <DofFixed>' + str(self.__bcRegions[r][5]) + '</DofFixed>\n')
            f.write('      <ForceX>' + str(self.__bcRegions[r][6][0]) + '</ForceX>\n')
            f.write('      <ForceY>' + str(self.__bcRegions[r][6][1]) + '</ForceY>\n')
            f.write('      <ForceZ>' + str(self.__bcRegions[r][6][2]) + '</ForceZ>\n')
            f.write('      <TorqueX>' + str(self.__bcRegions[r][7][0]) + '</TorqueX>\n')
            f.write('      <TorqueY>' + str(self.__bcRegions[r][7][1]) + '</TorqueY>\n')
            f.write('      <TorqueZ>' + str(self.__bcRegions[r][7][2]) + '</TorqueZ>\n')
            f.write('      <DisplaceX>' + str(self.__bcRegions[r][8][0] * 1e-3) + '</DisplaceX>\n')
            f.write('      <DisplaceY>' + str(self.__bcRegions[r][8][1] * 1e-3) + '</DisplaceY>\n')
            f.write('      <DisplaceZ>' + str(self.__bcRegions[r][8][2] * 1e-3) + '</DisplaceZ>\n')
            f.write('      <AngDisplaceX>' + str(self.__bcRegions[r][9][0]) + '</AngDisplaceX>\n')
            f.write('      <AngDisplaceY>' + str(self.__bcRegions[r][9][1]) + '</AngDisplaceY>\n')
            f.write('      <AngDisplaceZ>' + str(self.__bcRegions[r][9][2]) + '</AngDisplaceZ>\n')
            f.write('      <IntersectedVoxels>\n')

            for v in self.__bcVoxels[r]:
                f.write('        <Voxel>' + str(v).replace('[', '').replace(',', '').replace(']', '') + '</Voxel>\n')

            f.write('      </IntersectedVoxels>\n')
            f.write('    </FRegion>\n')

        f.write('  </Boundary_Conditions>\n')
        f.write('  <Gravity>\n')
        f.write('    <GravEnabled>' + str(int(self.__gravityEnable)) + '</GravEnabled>\n')
        f.write('    <GravAcc>' + str(self.__gravityValue) + '</GravAcc>\n')
        f.write('    <FloorEnabled>' + str(int(self.__floorEnable)) + '</FloorEnabled>\n')
        f.write('  </Gravity>\n')
        f.write('  <Thermal>\n')
        f.write('    <TempEnabled>' + str(int(self.__temperatureEnable)) + '</TempEnabled>\n')
        f.write('    <TempAmplitude>' + str(self.__temperatureVaryAmplitude) + '</TempAmplitude>\n')
        f.write('    <TempBase>' + str(self.__temperatureBaseValue) + '</TempBase>\n')
        f.write('    <VaryTempEnabled>' + str(int(self.__temperatureVaryEnable)) + '</VaryTempEnabled>\n')
        f.write('    <TempPeriod>' + str(self.__temperatureVaryPeriod) + '</TempPeriod>\n')
        f.write('  </Thermal>\n')
        f.write('</Environment>\n')

    def writeSensors(self, f: TextIO):
        """
        Write voxel sensors to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        f.write('<Sensors>\n')
        for sensor in self.__sensors:
            f.write('  <Sensor>\n')
            f.write('    <Location>' + str(sensor[0:3]).replace('[', '').replace(',', '').replace(']', '') + '</Location>\n')
            f.write('  </Sensor>\n')
        f.write('</Sensors>\n')

    def writeTempControls(self, f: TextIO):
        """
        Write temperature control element to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        f.write('<TempControls>\n')
        for element in self.__tempControls:
            f.write('  <Element>\n')
            f.write('    <Location>' + str(element[0:3]).replace('[', '').replace(',', '').replace(']', '') + '</Location>\n')
            f.write('    <Temperature>' + str(element[3]).replace('[', '').replace(',', '').replace(']', '') + '</Temperature>\n')
            f.write('  </Element>\n')
        f.write('</TempControls>\n')

    def runSim(self, filename: str = 'temp', value_map: int = 0, delete_files: bool = True, export_stl: bool = False, voxelyze_on_path: bool = False):
        """
        Run a Simulation object using Voxelyze.

        This function will create a .vxa file, run the file with Voxelyze, and then load the .xml results file into
        the results attribute of the Simulation object. Enabling delete_files will delete both the .vxa and .xml files
        once the results have been loaded.

        :param filename: File name for .vxa and .xml files
        :param value_map: Index of the desired value map type
        :param export_stl: Enable/disable exporting an stl file of the result
        :param delete_files: Enable/disable deleting simulation file when process is complete
        :param voxelyze_on_path: Enable/disable using system Voxelyze rather than bundled Voxelyze
        :return: None
        """
        # Create simulation file
        self.saveVXA(filename)

        if voxelyze_on_path:
            command_string = 'voxelyze '
        else:
            # Check OS type
            if os.name.startswith('nt'):
                # Windows - run Voxelyze with WSL
                command_string = 'wsl ' + os.path.dirname(os.path.realpath(__file__)).replace('C:', '/mnt/c').replace('\\', '/') + '/utils/voxelyze'
            else:
                # Linux - run Voxelyze directly
                command_string = os.path.dirname(os.path.realpath(__file__)) + '/utils/voxelyze'

        command_string = command_string + ' -f ' + filename + '.vxa -o ' + filename + '.xml -vm ' + str(value_map) + ' -p'

        if export_stl:
            command_string = command_string + ' -stl ' + filename + '.stl'

        print('Launching Voxelyze using: ' + command_string)
        p = subprocess.Popen(command_string, shell=True)
        p.wait()

        # Open simulation results
        f = open(filename + '.xml', 'r')
        print('Opening file: ' + f.name)
        data = f.readlines()
        f.close()

        # Clear any previous results
        self.results = []

        # Find start and end locations for individual sensors
        startLoc = []
        endLoc = []
        for row in tqdm(range(len(data)), desc='Finding sensor tags'):
            data[row] = data[row].replace('\t', '')
            if data[row][:-1] == '<Sensor>':
                startLoc.append(row)
            elif data[row][:-1] == '</Sensor>':
                endLoc.append(row)

        # Read the data from each sensor
        for sensor in tqdm(range(len(startLoc)), desc='Reading sensor results'):
            # Create a dictionary to hold the current sensor results
            sensorResults = {}

            # Read the data from each sensor tag
            for row in range(startLoc[sensor]+1, endLoc[sensor]):
                # Determine the current tag
                tag = ''
                for col in range(1, len(data[row])):
                    if data[row][col] == '>':
                        tag = data[row][1:col]
                        break

                # Remove the tags and newline to determine the current value
                data[row] = data[row].replace('<'+tag+'>', '').replace('</'+tag+'>', '')
                value = data[row][:-1]

                # Combine the current tag and value
                if tag == 'Location':
                    coords =  tuple(map(int, value.split(' ')))
                    x = coords[0] + self.__model.coords[0]
                    y = coords[1] + self.__model.coords[1]
                    z = coords[2] + self.__model.coords[2]
                    currentResult = {tag:(x, y, z)}
                elif ' ' in value:
                    currentResult = {tag:tuple(map(float, value.split(' ')))}
                else:
                    currentResult = {tag:float(value)}

                # Add the current tag and value to the sensor results dictionary
                sensorResults.update(currentResult)

            # Append the results dictionary for the current sensor to the simulation results list
            self.results.append(sensorResults)

        if os.path.exists('value_map.txt'):
            # Open simulation value map results
            f = open('value_map.txt', 'r')
            print('Opening file: ' + f.name)
            data = f.readlines()
            f.close()

            # Clear any previous results
            self.valueMap = np.zeros_like(self.__model.voxels, dtype=np.float32)

            # Get map size
            x_len = self.valueMap.shape[0]
            y_len = self.valueMap.shape[1]
            z_len = self.valueMap.shape[2]

            for z in tqdm(range(z_len), desc='Loading layers'):
                vals = np.array(data[z][:-2].split(","), dtype=np.float32)
                for y in range(y_len):
                    self.valueMap[:, y, z] = vals[y*x_len:(y+1)*x_len]

        # Remove temporary files
        if delete_files:
            print('Removing file: ' + filename + '.vxa')
            os.remove(filename + '.vxa')
            print('Removing file: ' + filename + '.xml')
            os.remove(filename + '.xml')

            if os.path.exists('value_map.txt'):
                print('Removing file: value_map.txt')
                os.remove('value_map.txt')

    def runSimVoxCad(self, filename: str = 'temp', delete_files: bool = True, voxcad_on_path: bool = False):
        """
        Run a Simulation object using the VoxCad GUI.

        ----

        Example:

        ``simulation = Simulation(modelResult)``

        ``simulation.setCollision()``

        ``simulation.setStopCondition(StopCondition.TIME_VALUE, 0.01)``

        ``simulation.runSimVoxCad('collision_sim_1', delete_files=False)``

        ----

        :param filename: File name
        :param delete_files: Enable/disable deleting simulation file when VoxCad is closed
        :param voxcad_on_path: Enable/disable using system VoxCad rather than bundled VoxCad
        :return: None
        """
        self.saveVXA(filename)

        if voxcad_on_path:
            command_string = 'voxcad '
        else:
            # Check OS type
            if os.name.startswith('nt'):
                # Windows
                command_string = os.path.dirname(os.path.realpath(__file__)) + '\\utils\\VoxCad.exe '
            else:
                # Linux
                command_string = os.path.dirname(os.path.realpath(__file__)) + '/utils/VoxCad '

        command_string = command_string + filename + '.vxa'

        print('Launching VoxCad using: ' + command_string)
        p = subprocess.Popen(command_string, shell=True)
        p.wait()

        if delete_files:
            print('Removing file: ' + filename + '.vxa')
            os.remove(filename + '.vxa')