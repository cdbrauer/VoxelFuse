"""
Simulation Class

Initialized from a VoxelModel object. Used to configure VoxCad and Voxelyze simulations.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import os
import time
import subprocess
import multiprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
from datetime import date
from typing import List, Tuple, TextIO
from tqdm import tqdm
import numpy as np
from voxelfuse.voxel_model import VoxelModel, writeHeader, writeData, writeOpen, writeClos
from voxelfuse.primitives import empty

# Floating point error threshold for rounding to zero
FLOATING_ERROR = 0.0000000001

class Axis(Enum):
    """
    Options for axes and planes.
    """
    NONE = -1
    X = 0
    Y = 1
    Z = 2

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

class _BCRegion:
    """
    Internal boundary condition class.
    """
    def __init__(self, name: str,
                 shape: BCShape,
                 position: Tuple[float, float, float],
                 size: Tuple[float, float, float],
                 radius: float,
                 fixed_dof: int,
                 force: Tuple[float, float, float],
                 displacement: Tuple[float, float, float],
                 torque: Tuple[float, float, float],
                 angular_displacement: Tuple[float, float, float]):

        self.name = name
        self.shape = shape
        self.position = position
        self.size = size
        self.radius = radius
        self.color = (0.6, 0.4, 0.4, 0.5)
        self.fixed_dof = fixed_dof
        self.force = force
        self.displacement = displacement
        self.torque = torque
        self.angular_displacement = angular_displacement

class _Sensor:
    """
    Internal sensor class.
    """
    def __init__(self, name: str, coords: Tuple[int, int, int], axis: Axis):
        self.name = name
        self.coords = coords
        self.axis = axis

class _Keyframe:
    """
    Internal keyframe class.
    """
    def __init__(self, time_value: float,
                 amplitude_pos: float,
                 amplitude_neg: float,
                 percent_pos: float,
                 period: float,
                 phase_offset: float,
                 temp_offset: float,
                 const_temp: bool,
                 square_wave: bool):

        self.time_value = time_value
        self.amplitude_pos = amplitude_pos
        self.amplitude_neg = amplitude_neg
        self.percent_pos = percent_pos
        self.period = period
        self.phase_offset = phase_offset
        self.temp_offset = temp_offset
        self.const_temp = const_temp
        self.square_wave = square_wave

class _TempControlGroup:
    """
    Internal temperature control group class.
    """
    def __init__(self, name: str, locations: List[Tuple[int, int, int]], keyframes: List[_Keyframe]):
        self.name = name
        self.locations = locations
        self.keyframes = keyframes

class _Disconnection:
    """
    Internal disconnection class.
    """
    def __init__(self, voxel_1: Tuple[int, int, int], voxel_2: Tuple[int, int, int]):
        self.vx1 = voxel_1
        self.vx2 = voxel_2

class Simulation:
    """
    Simulation object that stores a VoxelModel and its associated simulation settings.
    """

    def __init__(self, voxel_model, id_number: int = 0):
        """
        Initialize a Simulation object with default settings.

        Models located at positive coordinate values will have their workspace
        size adjusted to maintain their position in the exported simulation.
        Models located at negative coordinate values will be shifted to the origin.

        :param voxel_model: VoxelModel
        """
        # Simulation ID and start date
        self.id = id_number
        self.date = date.today()

        # Fit workspace and union with an empty object at the origin to clear offsets if object is raised
        self.__model = ((VoxelModel.copy(voxel_model).fitWorkspace()) | empty(num_materials=(voxel_model.materials.shape[1] - 1), resolution=voxel_model.resolution)).removeDuplicateMaterials()

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
        self.__hydrogelModelEnable = False

        # Stop conditions
        self.__stopConditionType = StopCondition.NONE
        self.__stopConditionValue = 0.0

        # Equilibrium mode
        self.__equilibriumModeEnable = False

        # Environment ############
        # Boundary conditions
        self.__bcRegions = []

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
        self.__temperatureVaryOffset = 0.0
        self.__growthAmplitude = 0.0

        # Sensors #######
        self.__sensors = []

        # Temperature Controls #######
        self.__currentTempControlGroup = 0
        self.__localTempControls = []

        # Disconnected Bonds #######
        self.__disconnections = []

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

        # Update ID and date
        new_simulation.id = simulation.id + 1
        new_simulation.date = date.today()

        # Make lists copies instead of references
        new_simulation.__bcRegions = simulation.__bcRegions.copy()
        new_simulation.__sensors = simulation.__sensors.copy()
        new_simulation.__localTempControls = simulation.__localTempControls.copy()
        new_simulation.__disconnections = simulation.__disconnections.copy()
        new_simulation.results = simulation.results.copy()
        new_simulation.valueMap = simulation.valueMap.copy()

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
        self.__model = ((VoxelModel.copy(voxel_model).fitWorkspace()) | empty(num_materials=(voxel_model.materials.shape[1] - 1))).removeDuplicateMaterials()

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

    def setHydrogelModel(self, enable: bool = True):
        """
        Set hydrogel model parameters.

        :param enable: Enable/disable hydrogel model
        :return: None
        """
        self.__hydrogelModelEnable = enable

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

    def setFixedThermal(self, enable: bool = True, base_temp: float = 25.0, growth_amplitude: float = 0.0):
        """
        Set a fixed environment temperature.

        :param enable: Enable/disable temperature
        :param base_temp: Temperature in degrees C
        :param growth_amplitude: Set to 1 to enable expansion from base size
        :return: None
        """
        self.__temperatureEnable = enable
        self.__temperatureBaseValue = base_temp
        self.__temperatureVaryEnable = False
        self.__growthAmplitude = growth_amplitude

    def setVaryingThermal(self, enable: bool = True, base_temp: float = 25.0, amplitude: float = 0.0, period: float = 1.0, offset: float = 0.0, growth_amplitude: float = 1.0):
        """
        Set a varying environment temperature.


        :param enable: Enable/disable temperature
        :param base_temp: Base temperature in degrees C
        :param amplitude: Temperature fluctuation amplitude
        :param period: Temperature fluctuation period
        :param offset: Temperature offset (not currently supported)
        :param growth_amplitude: Set to 1 to enable expansion from base size
        :return: None
        """
        self.__temperatureEnable = enable
        self.__temperatureBaseValue = base_temp
        self.__temperatureVaryEnable = enable
        self.__temperatureVaryAmplitude = amplitude
        self.__temperatureVaryPeriod = period
        self.__temperatureVaryOffset = offset
        self.__growthAmplitude = growth_amplitude

    # Read settings ##################################
    def getModel(self):
        """
        Get the simulation model.

        :return: VoxelModel
        """
        return self.__model

    def getVoxelDim(self):
        """
        Get the side dimension of a voxel in mm.

        :return: Float
        """
        res = self.__model.resolution
        return (1.0/res) * 0.001

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

    def getHydrogelModel(self):
        """
        Get hydrogel model parameters.

        :return: Enable/disable hydrogel model
        """
        return self.__hydrogelModelEnable

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

        :return: Enable/disable temperature, Base temperature in degrees C, Enable/disable temperature fluctuation, Temperature fluctuation amplitude, Temperature fluctuation period, Temperature offset
        """
        return self.__temperatureEnable, self.__temperatureBaseValue, self.__temperatureVaryEnable, self.__temperatureVaryAmplitude, self.__temperatureVaryPeriod, self.__temperatureVaryOffset

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
        # self.__bcVoxels = []

    # Default box boundary condition is a fixed constraint in the YZ plane
    def addBoundaryConditionVoxel(self, position: Tuple[int, int, int] = (0, 0, 0),
                                  fixed_dof: int = 0b111111,
                                  force: Tuple[float, float, float] = (0, 0, 0),
                                  displacement: Tuple[float, float, float] = (0, 0, 0),
                                  torque: Tuple[float, float, float] = (0, 0, 0),
                                  angular_displacement: Tuple[float, float, float] = (0, 0, 0),
                                  name: str = None):
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
        :param name: Boundary condition name
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

        bc = _BCRegion(name, BCShape.SPHERE, pos, (0.0, 0.0, 0.0), radius, fixed_dof, force, displacement, torque, angular_displacement)
        self.__bcRegions.append(bc)

    # Default box boundary condition is a fixed constraint in the XY plane (bottom layer)
    def addBoundaryConditionBox(self, position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                size: Tuple[float, float, float] = (1.0, 1.0, 0.01),
                                fixed_dof: int = 0b111111,
                                force: Tuple[float, float, float] = (0, 0, 0),
                                displacement: Tuple[float, float, float] = (0, 0, 0),
                                torque: Tuple[float, float, float] = (0, 0, 0),
                                angular_displacement: Tuple[float, float, float] = (0, 0, 0),
                                name: str = None):
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
        :param name: Boundary condition name
        :return: None
        """
        bc = _BCRegion(name, BCShape.BOX, position, size, 0, fixed_dof, force, displacement, torque, angular_displacement)
        self.__bcRegions.append(bc)

    # Default sphere boundary condition is a fixed constraint centered in the model
    def addBoundaryConditionSphere(self, position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                                   radius: float = 0.05,
                                   fixed_dof: int = 0b111111,
                                   force: Tuple[float, float, float] = (0, 0, 0),
                                   displacement: Tuple[float, float, float] = (0, 0, 0),
                                   torque: Tuple[float, float, float] = (0, 0, 0),
                                   angular_displacement: Tuple[float, float, float] = (0, 0, 0),
                                   name: str = None):
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
        :param name: Boundary condition name
        :return: None
        """
        bc = _BCRegion(name, BCShape.SPHERE, position, (0.0, 0.0, 0.0), radius, fixed_dof, force, displacement, torque, angular_displacement)
        self.__bcRegions.append(bc)

    # Default cylinder boundary condition is a fixed constraint centered in the model
    def addBoundaryConditionCylinder(self, position: Tuple[float, float, float] = (0.45, 0.5, 0.5), axis: Axis = Axis.X,
                                     height: float = 0.1,
                                     radius: float = 0.05,
                                     fixed_dof: int = 0b111111,
                                     force: Tuple[float, float, float] = (0, 0, 0),
                                     displacement: Tuple[float, float, float] = (0, 0, 0),
                                     torque: Tuple[float, float, float] = (0, 0, 0),
                                     angular_displacement: Tuple[float, float, float] = (0, 0, 0),
                                     name: str = None):
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
        :param name: Boundary condition name
        :return: None
        """
        size = [0.0, 0.0, 0.0]
        size[axis.value] = height
        bc = _BCRegion(name, BCShape.CYLINDER, position, (size[0], size[1], size[2]), radius, fixed_dof, force, displacement, torque, angular_displacement)
        self.__bcRegions.append(bc)

    def clearSensors(self):
        """
        Remove all sensors from a Simulation object.

        :return: None
        """
        self.__sensors = []

    def addSensor(self, location: Tuple[int, int, int] = (0, 0, 0), axis: Axis = Axis.NONE, name: str = None): # TODO: Make Voxelyze use axis parameter
        """
        Add a sensor to a voxel.

        This feature is not currently supported by VoxCad

        :param location: Sensor location in voxels
        :param axis: Sensor measurement axis
        :param name: Sensor name
        :return: None
        """
        x = location[0] - self.__model.coords[0]
        y = location[1] - self.__model.coords[1]
        z = location[2] - self.__model.coords[2]

        sensor = _Sensor(name, (x, y, z), axis)
        self.__sensors.append(sensor)

        if not self.__model.isOccupied((x, y, z)):
            print('WARNING: No material present at sensor voxel ' + str((x, y, z)))

    def clearDisconnections(self):
        """
        Clear all disconnected voxel bonds.

        :return: None
        """
        self.__disconnections = []

    def addDisconnection(self, voxel_1: Tuple[int, int, int], voxel_2: Tuple[int, int, int]):
        """
        Specify a pair of voxels which should be disconnected

        :param voxel_1: Coordinates in voxels
        :param voxel_2: Coordinates in voxels
        :return: None
        """
        dc = _Disconnection(voxel_1, voxel_2)
        self.__disconnections.append(dc)

    def addTempControlGroup(self, locations: List[Tuple[int, int, int]] = None, name: str = None):
        """
        Add a new temperature control group and select it.

        :param locations: Control element locations in voxels as a list of tuples
        :param name: Group name
        :return: None
        """
        # Enable temperature control
        self.__temperatureEnable = True
        self.__temperatureVaryEnable = True

        if locations is None:
            print('No locations provided - applying temperature control group to entire model')

            x_len = self.__model.voxels.shape[0]
            y_len = self.__model.voxels.shape[1]
            z_len = self.__model.voxels.shape[2]

            locations = []
            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        locations.append((x, y, z))

        group = _TempControlGroup(name, locations, [])
        self.__localTempControls.append(group)
        self.__currentTempControlGroup = len(self.__localTempControls)-1

    def removeTempControlGroup(self, index: int = 0):
        """
        Remove a temperature control group by index.

        :param index: Temperature control group index
        :return: Name of removed group
        """
        group = self.__localTempControls.pop(index)
        self.__currentTempControlGroup = len(self.__localTempControls)-1
        return group.name

    def selectTempControlGroup(self, index: int = 0):
        """
        Select which keyframe new temperature control elements should be added to.

        :param index: Temperature control group index
        :return: None
        """
        self.__currentTempControlGroup = index

    def clearTempControlGroups(self):
        """
        Clear all temperature control groups.

        :return: None
        """
        self.__currentTempControlGroup = 0
        self.__localTempControls = []

    def cleanTempControlGroups(self):
        """
        Remove invalid temperature control groups.

        Invalid groups have no keyframes, or have no target voxels.

        :return: Number of groups removed
        """
        removed_groups = []
        for g in range(len(self.__localTempControls)):
            g = g - len(removed_groups)
            group = self.__localTempControls[g]
            if len(group.keyframes) == 0 or len(group.locations) == 0:
                removed_groups.append(self.removeTempControlGroup(g))

        self.__currentTempControlGroup = len(self.__localTempControls)-1

        named_groups = []
        for n in removed_groups:
            if n is not None:
                named_groups.append(n)

        if len(removed_groups) == 0:
            pass
        elif len(named_groups) == 0:
            print('Removed ' + str(len(removed_groups)) + ' temperature control groups')
        else:
            print('Removed ' + str(len(removed_groups)) + ' temperature control groups, including: ' + str(named_groups))

        return len(removed_groups)

    def clearKeyframes(self):
        """
        Remove all keyframes assigned to the current temperature control group.

        :return: None
        """
        g = self.__currentTempControlGroup
        self.__localTempControls[g].keyframes = []

    def addKeyframe(self, time_value: float = 0, amplitude_pos: float = 0, amplitude_neg: float = -1, percent_pos: float = 0.5,
                    period: float = 1.0, phase_offset: float = 0, temp_offset: float = 0, const_temp: bool = False, square_wave: bool = False):
        """
        Add a keyframe to a temperature control group.

        :param time_value: Time at which keyframe should take effect (sec)
        :param amplitude_pos: Control element positive temperature amplitude (deg C)
        :param amplitude_neg: Control element negative temperature amplitude (deg C)
        :param percent_pos: Percent of period spanned by positive temperature amplitude (0-1)
        :param period: Period of the control signal (sec)
        :param phase_offset: Control element phase offset for time-varying thermal (rad)
        :param temp_offset: Control element temperature offset for time-varying thermal (deg C)
        :param const_temp: Enable/disable setting a constant target temperature that respects heating/cooling rates
        :param square_wave: Enable/disable converting signal to a square wave (positive -> a = amplitude1, negative -> a = 0)
        :return: None
        """
        amplitude_pos = max(amplitude_pos, 0) # amplitude_pos must be positive

        if amplitude_neg < 0: # If amplitude neg is not given (or not negative) use amplitude_pos instead
            amplitude_neg = amplitude_pos

        g = self.__currentTempControlGroup
        kf = _Keyframe(time_value, amplitude_pos, amplitude_neg, percent_pos, period, phase_offset, temp_offset, const_temp, square_wave)
        self.__localTempControls[g].keyframes.append(kf)

    def initializeTempMap(self):
        """
        Initialize temperature control groups to which keyframes will be stored when using applyTempMap.

        :return: None
        """
        # Clear any existing temperature controls
        self.clearTempControlGroups()

        # Get map size
        x_len = self.__model.voxels.shape[0]
        y_len = self.__model.voxels.shape[1]
        z_len = self.__model.voxels.shape[2]

        # Generate empty control element for each voxel
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    self.addTempControlGroup([(x, y, z)])

    def applyTempMap(self, time_value, amp_pos_map, amp_neg_map=None, percent_pos_map=None, period_map=None,
                     phase_map=None, offset_map=None, const_temp_map=None, square_wave_map=None):
        """
        Set the simulation temperature control elements based on a value maps of target temperature settings.

        This function relies on the temperature control groups being in a specific order. initializeTempMap should be called prior to running this function.

        :param time_value: Time at which keyframe should take effect (sec)
        :param amp_pos_map: Array of target positive temperature amplitudes for each voxel (deg C)
        :param amp_neg_map: Array of target negative temperature amplitudes for each voxel (deg C)
        :param percent_pos_map: Array containing percent of period spanned by positive temperature amplitude for each voxel
        :param period_map: Array of signal periods for each voxel (sec)
        :param phase_map: Array of phase offsets for each voxel (rad)
        :param offset_map: Array of temperature offsets for each voxel (deg C)
        :param const_temp_map: Array of boolean values to enable/disable constant temperature mode
        :param square_wave_map: Array of boolean values to enable/disable square wave mode
        :return: None
        """
        # Get map size
        x_len = amp_pos_map.shape[0]
        y_len = amp_pos_map.shape[1]
        z_len = amp_pos_map.shape[2]

        # Generate the control element for each voxel
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    if abs(amp_pos_map[x, y, z]) > FLOATING_ERROR or ((amp_neg_map is not None) and (abs(amp_neg_map[x, y, z]) > FLOATING_ERROR)):  # If voxel is not empty
                        amplitude_pos = amp_pos_map[x, y, z]

                        if amp_neg_map is None:
                            amplitude_neg = amp_pos_map[x, y, z]
                        else:
                            amplitude_neg = amp_neg_map[x, y, z]

                        if percent_pos_map is None:
                            percent_pos = 0.5
                        else:
                            percent_pos = percent_pos_map[x, y, z]

                        if period_map is None:
                            period = 1.0
                        else:
                            period = period_map[x, y, z]

                        if phase_map is None:
                            phase_offset = 0
                        else:
                            phase_offset = phase_map[x, y, z]

                        if offset_map is None:
                            temp_offset = 0
                        else:
                            temp_offset = offset_map[x, y, z]

                        if const_temp_map is None:
                            const_temp = False
                        else:
                            const_temp = const_temp_map[x, y, z]

                        if square_wave_map is None:
                            square_wave = False
                        else:
                            square_wave = square_wave_map[x, y, z]

                        kf = _Keyframe(time_value, amplitude_pos, amplitude_neg, percent_pos, period, phase_offset, temp_offset, const_temp, square_wave)
                        g = z + y*z_len + x*y_len*z_len
                        self.__localTempControls[g].keyframes.append(kf)

    # TODO: Remove or update to be compatible with keyframes
    # def saveTempControls(self, filename: str, figure: bool = False):
    #     """
    #     Save the temperature control elements applied to a model to a .csv file.
    #
    #     :param filename: File name
    #     :param figure: Enable/disable exporting a figure as well
    #     :return: None
    #     """
    #     f = open(filename + '.csv', 'w+')
    #     print('Saving file: ' + f.name)
    #     f.write('X,Y,Z,Amplitude 1 (deg C),Amplitude 2 (deg C),Change X,Phase Offset (rad),Temperature Offset (deg C),Constant Temperature Enabled,Square Wave Enabled\n')
    #     for i in range(len(self.__tempControls)):
    #         f.write(str(self.__tempControls[i]).replace('[', '').replace(' ', '').replace(']', '') + '\n')
    #     f.close()
    #
    #     if figure:
    #         # Get plot data
    #         points = np.array(self.__tempControls)
    #         xs = points[:, 0]
    #         ys = points[:, 1]
    #         zs = points[:, 2]
    #         temps = points[:, 3]
    #         colors = np.array(abs((temps - np.min(temps)) / (np.max(temps) - np.min(temps))), dtype=np.str)  # Grayscale range
    #
    #         # Plot results
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot(121)
    #         ax1.scatter(zs, ys, c=colors, marker='s')
    #         ax1.axis('equal')
    #         ax1.set_title('Side')
    #         ax2 = fig.add_subplot(122)
    #         ax2.scatter(xs, ys, c=colors, marker='s')
    #         ax2.axis('equal')
    #         ax2.set_title('Top')
    #
    #         # Save figure
    #         print('Saving file: ' + filename + '.png')
    #         plt.savefig(filename + '.png')

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
        self.cleanTempControlGroups()
        f = open(filename + '.vxa', 'w+')
        print('Saving file: ' + f.name)

        writeHeader(f, '1.0', 'ISO-8859-1')
        writeOpen(f, 'VXA Version="' + str(1.1) + '"', 0)
        self.writeSimData(f)
        self.writeEnvironmentData(f)
        self.writeSensors(f)
        self.writeTempControls(f)
        self.writeDisconnections(f)
        self.__model.writeVXCData(f, compression)
        writeClos(f, 'VXA', 0)
        f.close()

    # Write simulator settings to file
    def writeSimData(self, f: TextIO):
        """
        Write simulation parameters to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        # Simulator settings
        writeOpen(f, 'Simulator', 0)
        writeOpen(f, 'Integration', 1)
        writeData(f, 'Integrator', self.__integrator, 2)
        writeData(f, 'DtFrac', self.__dtFraction, 2)
        writeClos(f, 'Integration', 1)

        writeOpen(f, 'Damping', 1)
        writeData(f, 'BondDampingZ', self.__dampingBond, 2)
        writeData(f, 'ColDampingZ', self.__collisionDamping, 2)
        writeData(f, 'SlowDampingZ', self.__dampingEnvironment, 2)
        writeClos(f, 'Damping', 1)

        writeOpen(f, 'Collisions', 1)
        writeData(f, 'SelfColEnabled', int(self.__collisionEnable), 2)
        writeData(f, 'ColSystem', self.__collisionSystem, 2)
        writeData(f, 'CollisionHorizon', self.__collisionHorizon, 2)
        writeClos(f, 'Collisions', 1)

        writeOpen(f, 'Features', 1)
        writeData(f, 'BlendingEnabled', int(self.__blendingEnable), 2)
        writeData(f, 'XMixRadius', self.__xMixRadius, 2)
        writeData(f, 'YMixRadius', self.__yMixRadius, 2)
        writeData(f, 'ZMixRadius', self.__zMixRadius, 2)
        writeData(f, 'BlendModel', self.__blendingModel, 2)
        writeData(f, 'PolyExp', self.__polyExp, 2)
        writeData(f, 'VolumeEffectsEnabled', int(self.__volumeEffectsEnable), 2)
        writeClos(f, 'Features', 1)

        writeOpen(f, 'StopCondition', 1)
        writeData(f, 'StopConditionType', self.__stopConditionType.value, 2)
        writeData(f, 'StopConditionValue', self.__stopConditionValue, 2)
        writeClos(f, 'StopCondition', 1)

        writeOpen(f, 'EquilibriumMode', 1)
        writeData(f, 'EquilibriumModeEnabled', int(self.__equilibriumModeEnable), 2)
        writeClos(f, 'EquilibriumMode', 1)
        writeClos(f, 'Simulator', 0)

    # Write environment settings to file
    def writeEnvironmentData(self, f: TextIO):
        """
        Write simulation environment parameters to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        # Environment settings
        writeOpen(f, 'Environment', 0)
        writeOpen(f, 'Boundary_Conditions', 1)
        writeData(f, 'NumBCs', len(self.__bcRegions), 2)
        for bc in self.__bcRegions:
            writeOpen(f, 'FRegion', 2)
            if bc.name is not None:
                writeData(f, 'Name', bc.name, 3)
            writeData(f, 'PrimType', int(bc.shape.value), 3)
            writeData(f, 'X', bc.position[0], 3)
            writeData(f, 'Y', bc.position[1], 3)
            writeData(f, 'Z', bc.position[2], 3)
            writeData(f, 'dX', bc.size[0], 3)
            writeData(f, 'dY', bc.size[1], 3)
            writeData(f, 'dZ', bc.size[2], 3)
            writeData(f, 'Radius', bc.radius, 3)
            writeData(f, 'R', bc.color[0], 3)
            writeData(f, 'G', bc.color[1], 3)
            writeData(f, 'B', bc.color[2], 3)
            writeData(f, 'alpha', bc.color[3], 3)
            writeData(f, 'DofFixed', bc.fixed_dof, 3)
            writeData(f, 'ForceX', bc.force[0], 3)
            writeData(f, 'ForceY', bc.force[1], 3)
            writeData(f, 'ForceZ', bc.force[2], 3)
            writeData(f, 'TorqueX', bc.torque[0], 3)
            writeData(f, 'TorqueY', bc.torque[1], 3)
            writeData(f, 'TorqueZ', bc.torque[2], 3)
            writeData(f, 'DisplaceX', bc.displacement[0] * 1e-3, 3)
            writeData(f, 'DisplaceY', bc.displacement[1] * 1e-3, 3)
            writeData(f, 'DisplaceZ', bc.displacement[2] * 1e-3, 3)
            writeData(f, 'AngDisplaceX', bc.angular_displacement[0], 3)
            writeData(f, 'AngDisplaceY', bc.angular_displacement[1], 3)
            writeData(f, 'AngDisplaceZ', bc.angular_displacement[2], 3)
            writeClos(f, 'FRegion', 2)
        writeClos(f, 'Boundary_Conditions', 1)

        writeOpen(f, 'Gravity', 1)
        writeData(f, 'GravEnabled', int(self.__gravityEnable), 2)
        writeData(f, 'GravAcc', self.__gravityValue, 2)
        writeData(f, 'FloorEnabled', int(self.__floorEnable), 2)
        writeClos(f, 'Gravity', 1)

        writeOpen(f, 'Thermal', 1)
        writeData(f, 'TempEnabled', int(self.__temperatureEnable), 2)
        writeData(f, 'TempAmplitude', self.__temperatureVaryAmplitude, 2)
        writeData(f, 'TempBase', self.__temperatureBaseValue, 2)
        writeData(f, 'VaryTempEnabled', int(self.__temperatureVaryEnable), 2)
        writeData(f, 'TempPeriod', self.__temperatureVaryPeriod, 2)
        writeData(f, 'TempOffset', self.__temperatureVaryOffset, 2)
        writeClos(f, 'Thermal', 1)

        writeData(f, 'GrowthAmplitude', self.__growthAmplitude, 1)
        writeClos(f, 'Environment', 0)

    def writeSensors(self, f: TextIO):
        """
        Write voxel sensors to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        writeOpen(f, 'Sensors', 0)
        for sensor in self.__sensors:
            writeOpen(f, 'Sensor', 1)
            if sensor.name is not None:
                writeData(f, 'Name', sensor.name, 2)
            writeData(f, 'Location', str(sensor.coords).replace('(', '').replace(',', '').replace(')', ''), 2)
            writeData(f, 'Axis', sensor.axis.value, 2)
            writeClos(f, 'Sensor', 1)
        writeClos(f, 'Sensors', 0)

    def writeTempControls(self, f: TextIO):
        """
        Write temperature control element to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        writeData(f, 'EnableHydrogelModel', int(self.__hydrogelModelEnable), 0)

        writeOpen(f, 'TempControls', 0)
        for group in self.__localTempControls:
            writeOpen(f, 'Element', 1)

            if group.name is not None:
                writeData(f, 'Name', group.name, 2)

            writeOpen(f, 'Locations', 2)
            for loc in group.locations:
                writeData(f, 'Location', str(loc).replace('(', '').replace(',', '').replace(')', ''), 3)
            writeClos(f, 'Locations', 2)

            writeOpen(f, 'Keyframes', 2)
            for keyframe in group.keyframes:
                writeOpen(f, 'Keyframe', 3)
                writeData(f, 'TimeValue', keyframe.time_value, 4)
                writeData(f, 'AmplitudePos', keyframe.amplitude_pos, 4)
                writeData(f, 'AmplitudeNeg', keyframe.amplitude_neg, 4)
                writeData(f, 'PercentPos', keyframe.percent_pos, 4)
                writeData(f, 'Period', keyframe.period, 4)
                writeData(f, 'PhaseOffset', keyframe.phase_offset, 4)
                writeData(f, 'TempOffset', keyframe.temp_offset, 4)
                writeData(f, 'ConstantTemp', int(keyframe.const_temp), 4)
                writeData(f, 'SquareWave', int(keyframe.square_wave), 4)
                writeClos(f, 'Keyframe', 3)
            writeClos(f, 'Keyframes', 2)

            writeClos(f, 'Element', 1)
        writeClos(f, 'TempControls', 0)

    def writeDisconnections(self, f: TextIO):
        """
        Write bond disconnections to a text file using the .vxa format.

        :param f: File to write to
        :return: None
        """
        writeOpen(f, 'Disconnections', 0)
        for dc in self.__disconnections:
            writeOpen(f, 'Break', 1)
            writeData(f, 'Voxel1', str(dc.vx1).replace('(', '').replace(',', '').replace(')', ''), 2)
            writeData(f, 'Voxel2', str(dc.vx2).replace('(', '').replace(',', '').replace(')', ''), 2)
            writeClos(f, 'Break', 1)
        writeClos(f, 'Disconnections', 0)

    def runSim(self, filename: str = 'temp', value_map: int = 0, delete_files: bool = True, log_interval: int = -1, history_interval: int = -1, voxelyze_on_path: bool = False, wsl: bool = False):
        """
        Run a Simulation object using Voxelyze.

        This function will create a .vxa file, run the file with Voxelyze, and then load the .xml results file into
        the results attribute of the Simulation object. Enabling delete_files will delete both the .vxa and .xml files
        once the results have been loaded.

        :param filename: File name for .vxa and .xml files
        :param value_map: Index of the desired value map type
        :param log_interval: Set the step interval at which sensor log entries should be recorded, -1 to disable log
        :param history_interval: Set the step interval at which history file entries should be recorded, -1 for default interval
        :param delete_files: Enable/disable deleting simulation file when process is complete
        :param voxelyze_on_path: Enable/disable using system Voxelyze rather than bundled Voxelyze
        :param wsl: Enable/disable using Windows Subsystem for Linux with bundled Voxelyze
        :return: None
        """
        # Generate results file/directory names
        filename = filename + '_' + str(self.id)
        dirname = 'sim_results_' + str(self.date)

        # Create results directory
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Create simulation file
        self.saveVXA(dirname + '/' + filename)

        if voxelyze_on_path:
            command_string = 'voxelyze'
        else:
            # Check OS type
            if os.name.startswith('nt'): # Windows
                if wsl:
                    command_string = 'wsl "' + os.path.dirname(os.path.realpath(__file__)).replace('C:', '/mnt/c').replace('\\', '/') + '/utils/voxelyze"'
                else:
                    command_string = f'"{os.path.dirname(os.path.realpath(__file__))}\\utils\\voxelyze.exe"'
            else: # Linux
                command_string = f'"{os.path.dirname(os.path.realpath(__file__))}/utils/voxelyze"'

        command_string = command_string + ' -f ' + dirname + '/' + filename + '.vxa -o ' + dirname + '/' + filename + ' -vm ' + str(value_map) + ' -p'

        if log_interval > 0:
            command_string = command_string + ' -log-interval ' + str(log_interval)

        if history_interval > 0:
            command_string = command_string + ' -history-interval ' + str(history_interval)

        print('Launching Voxelyze using: ' + command_string)
        p = subprocess.Popen(command_string, shell=True)
        p.wait()

        # Open simulation results
        f = open(dirname + '/' + filename + '.xml', 'r')
        #print('Opening file: ' + f.name)
        data = f.readlines()
        f.close()

        # Clear any previous results
        self.results = []

        # Find start and end locations for individual sensors
        startLoc = []
        endLoc = []
        for row in range(len(data)): # tqdm(range(len(data)), desc='Finding sensor tags'):
            data[row] = data[row].replace('\t', '')
            if data[row][:-1] == '<Sensor>':
                startLoc.append(row)
            elif data[row][:-1] == '</Sensor>':
                endLoc.append(row)

        # Read the data from each sensor
        sensor_count = len(startLoc)
        if sensor_count > 0:
            for sensor in range(sensor_count): # tqdm(range(len(startLoc)), desc='Reading sensor results'):
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
        else:
            print('No sensors found')

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

            for z in range(z_len): # tqdm(range(z_len), desc='Loading layers'):
                vals = np.array(data[z][:-2].split(","), dtype=np.float32)
                for y in range(y_len):
                    self.valueMap[:, y, z] = vals[y*x_len:(y+1)*x_len]

        # Remove temporary files
        if delete_files:
            os.remove(dirname + '/' + filename + '.vxa')
            os.remove(dirname + '/' + filename + '.xml')

            if os.path.exists(dirname + '/value_map.txt'):
                os.remove(dirname + '/value_map.txt')

    def runSimVoxCad(self, filename: str = 'temp', delete_files: bool = True, voxcad_on_path: bool = False, wsl: bool = False):
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
        :param wsl: Enable/disable using Windows Subsystem for Linux with bundled VoxCad
        :return: None
        """
        # Generate file name
        filename = filename + '_' + str(self.id)

        # Create simulation file
        self.saveVXA(filename)

        if voxcad_on_path:
            command_string = 'voxcad '
        else:
            # Check OS type
            if os.name.startswith('nt'): # Windows
                if wsl:
                    command_string = 'wsl "' + os.path.dirname(os.path.realpath(__file__)).replace('C:', '/mnt/c').replace('\\', '/') + '/utils/VoxCad" '
                else:
                    command_string = f'"{os.path.dirname(os.path.realpath(__file__))}\\utils\\VoxCad.exe" '
            else: # Linux
                command_string =  f'"{os.path.dirname(os.path.realpath(__file__))}/utils/VoxCad" '

        command_string = command_string + filename + '.vxa'

        print('Launching VoxCad using: ' + command_string)
        p = subprocess.Popen(command_string, shell=True)
        p.wait()

        if delete_files:
            print('Removing file: ' + filename + '.vxa')
            os.remove(filename + '.vxa')

    def saveResults(self, filename):
        """
        Saves a simulation's results dictionary to a .csv file.

        :param filename: Name of output file
        :return:
        """
        # Get result table keys and size
        keys = list(self.results[0].keys())
        rows = len(self.results)
        cols = len(keys)

        # Create results file
        f = open(filename + '.csv', 'w+')
        print('Saving file: ' + f.name)

        # Write headings
        f.write('Sensor')
        for c in range(cols):
            f.write(',' + str(keys[c]))
        f.write('\n')

        # Write values
        for r in range(rows):
            f.write(str(r))
            vals = list(self.results[r].values())
            for c in range(cols):
                f.write(',' + str(vals[c]).replace(',', ' '))
            f.write('\n')

        # Close file
        f.close()

class MultiSimulation:
    """
    MultiSimulation object that holds settings for generating a simulation and running multiple parallel trials of it using different parameters.
    """

    def __init__(self, setup_fcn, setup_params: List[Tuple], thread_count: int = -1):
        """
        Initialize a MultiSimulation object.

        :param setup_fcn: Function to use for initializing Simulation objects. Should take a single tuple as an input and return a single Simulation object.
        :param setup_params: List containing the desired input tuples for setup_fcn
        :param thread_count: Maximum number of CPU threads, -1 to auto-detect
        """
        self.__setup_fcn = setup_fcn
        self.__setup_params = setup_params

        if thread_count > 0:
            self.__thread_count = thread_count
        else:
            max_threads = os.cpu_count()
            self.__thread_count = max(1,
                                      max_threads - 2)  # Default to leaving 1 core (2 threads) free, minimum of 1 thread

            # Initialize result arrays
        self.total_time = 0
        self.displacement_result = multiprocessing.Array('d', len(setup_params))
        self.time_result = multiprocessing.Array('d', len(setup_params))

    def getParams(self):
        """
        Get the current simulation setup parameters.

        :return: List containing the current input tuples for setup_fcn
        """
        return self.__setup_params

    def setParams(self, setup_params: List[Tuple]):
        """
        Update the simulation setup parameters.

        :param setup_params: List containing the desired input tuples for setup_fcn
        :return: None
        """
        self.__setup_params = setup_params
        self.displacement_result = multiprocessing.Array('d', len(setup_params))
        self.time_result = multiprocessing.Array('d', len(setup_params))

    def confirmSimCount(self):
        """
        Print the number of simulations to be run and confirm that the user would like to continue.

        :return: None
        """
        # Check if results directory already exists
        dirname = 'sim_results_' + str(date.today())
        if os.path.exists(dirname):
            print("WARNING: Previous results exist and may be overwritten: " + str(dirname))

        print("Trials to run: " + str(len(self.__setup_params)))
        print("Max CPU threads: " + str(self.__thread_count))
        input("Press Enter to continue...")

    def run(self, enable_log : bool = False, fine_log: bool = False):
        """
        Run all simulation configurations and save the results.

        :param enable_log: Enable saving sensor log files
        :param fine_log: If enabled, save entries in sensor logs and history files 100x as frequently
        :return: None
        """
        # Save start time
        time_started = time.time()

        # Set up simulations
        sim_id = 0
        sim_array = []
        for config in tqdm(self.__setup_params, desc='Initializing simulations'):
            sim = self.__setup_fcn(config)

            # Use automatic sim id if one is not already set
            if sim.id == 0:
                sim.id = sim_id
                sim_id = sim_id + 1

            sim_array.append(sim)

        # Initialize processing pool
        p = multiprocessing.Pool(self.__thread_count, initializer=poolInit, initargs=(self.displacement_result, self.time_result))
        if enable_log:
            if fine_log:
                p.map(simProcessLogFine, sim_array)
            else:
                p.map(simProcessLog, sim_array)
        else:
            p.map(simProcess, sim_array)

        # Get elapsed time
        time_finished = time.time()
        self.total_time = time_finished - time_started

    def export(self, filename: str, labels: List[str]):
        """
        Export a CSV file containing simulation setup parameters and the corresponding simulation results.

        :param filename: File name
        :param labels: Column headers for simulation setup parameters
        :return: None
        """
        # Add labels for results
        labels.append('Displacement (m)')
        labels.append('Simulation Time (s)')

        # Get result table size
        rows = len(self.__setup_params)
        cols = len(labels)

        # Create results file
        f = open(filename + '.csv', 'w+')
        print('Saving file: ' + f.name)

        # Write sim info
        f.write('Simulation Elapsed Time (mins),' + str(self.total_time / 60.0) + '\n')
        f.write('Trial Count,' + str(rows) + '\n')
        f.write('Max Thread Count,' + str(self.__thread_count) + '\n')
        f.write('\n')

        # Write headings
        for c in range(cols):
            f.write(labels[c] + ',')
        f.write('\n')

        # Write values
        for r in range(rows):
            for c in range(cols - 2):
                f.write(str(self.__setup_params[r][c]) + ',')
            f.write(str(self.displacement_result[r]) + ',')
            f.write(str(self.time_result[r]))
            f.write('\n')

        # Close file
        f.close()

    def exportVXA(self, filename: str, config_number: int = -1):
        """
        Export VXA files for all or specified simulation configurations.

        :param filename: File name
        :param config_number: Config to export, -1 to export all configs
        :return: None
        """
        if config_number == -1:
            for config in tqdm(self.__setup_params, desc='Saving simulations'):
                sim = self.__setup_fcn(config)
                sim.saveVXA(filename + '_' + str(config[0]))
        else:
            config = self.__setup_params[config_number]
            sim = self.__setup_fcn(config)
            sim.saveVXA(filename)

# Helper functions
def poolInit(disp_result_array, t_result_array):
    """
    Initialize shared result variables.

    :param disp_result_array: Multiprocessing array to hold displacement results
    :param t_result_array: Multiprocessing array to hold time results
    :return: None
    """
    global disp_result, t_result
    disp_result = disp_result_array
    t_result = t_result_array

def simProcess(simulation: Simulation):
    """
    Simulation process.

    :param simulation: Simulation object to run
    :return: None
    """
    print('Process ' + str(simulation.id) + ' starting')

    # Run simulation
    time_process_started = time.time()
    simulation.runSim('multisim', log_interval=-1, history_interval=100000, wsl=True)
    time_process_finished = time.time()

    # Read results
    try:
        disp_x = float(simulation.results[0]['Position'][0]) - float(simulation.results[0]['InitialPosition'][0])
        disp_y = float(simulation.results[0]['Position'][1]) - float(simulation.results[0]['InitialPosition'][1])
        disp_z = float(simulation.results[0]['Position'][2]) - float(simulation.results[0]['InitialPosition'][2])
        disp_result[simulation.id] = np.sqrt((disp_x**2) + (disp_y**2) + (disp_z**2))
        t_result[simulation.id] = time_process_finished - time_process_started
    except IndexError:
        print('Unable to load sensor results')

    # Finished
    print('Process ' + str(simulation.id) + ' finished')

def simProcessLog(simulation: Simulation):
    """
    Simulation process.

    :param simulation: Simulation object to run
    :return: None
    """
    print('Process ' + str(simulation.id) + ' starting')

    # Run simulation
    time_process_started = time.time()
    simulation.runSim('multisim', log_interval=100000, history_interval=100000, wsl=True)
    time_process_finished = time.time()

    # Read results
    try:
        disp_x = float(simulation.results[0]['Position'][0]) - float(simulation.results[0]['InitialPosition'][0])
        disp_y = float(simulation.results[0]['Position'][1]) - float(simulation.results[0]['InitialPosition'][1])
        disp_z = float(simulation.results[0]['Position'][2]) - float(simulation.results[0]['InitialPosition'][2])
        disp_result[simulation.id] = np.sqrt((disp_x**2) + (disp_y**2) + (disp_z**2))
        t_result[simulation.id] = time_process_finished - time_process_started
    except IndexError:
        print('Unable to load sensor results')

    # Finished
    print('Process ' + str(simulation.id) + ' finished')

def simProcessLogFine(simulation: Simulation):
    """
    Simulation process.

    :param simulation: Simulation object to run
    :return: None
    """
    print('Process ' + str(simulation.id) + ' starting')

    # Run simulation
    time_process_started = time.time()
    simulation.runSim('multisim', log_interval=1000, history_interval=1000, wsl=True)
    time_process_finished = time.time()

    # Read results
    try:
        disp_x = float(simulation.results[0]['Position'][0]) - float(simulation.results[0]['InitialPosition'][0])
        disp_y = float(simulation.results[0]['Position'][1]) - float(simulation.results[0]['InitialPosition'][1])
        disp_z = float(simulation.results[0]['Position'][2]) - float(simulation.results[0]['InitialPosition'][2])
        disp_result[simulation.id] = np.sqrt((disp_x**2) + (disp_y**2) + (disp_z**2))
        t_result[simulation.id] = time_process_finished - time_process_started
    except IndexError:
        print('Unable to load sensor results')

    # Finished
    print('Process ' + str(simulation.id) + ' finished')