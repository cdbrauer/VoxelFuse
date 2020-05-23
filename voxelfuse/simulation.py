"""
Copyright 2020
Dan Aukes, Cole Brauer
"""

import numpy as np
from enum import Enum
from tqdm import tqdm
from voxelfuse.voxel_model import VoxelModel

class StopCondition(Enum):
    NONE = 0
    TIME_STEP = 1
    TIME_VALUE = 2
    TEMP_CYCLES = 3
    ENERGY_CONST = 4
    ENERGY_KFLOOR = 5
    MOTION_FLOOR = 6

class Simulation:
    # Initialize a simulation object with default settings
    def __init__(self, voxel_model):
        self.model = VoxelModel.copy(voxel_model)

        # Simulator ##################################
        # Integration
        self.integrator = 0
        self.dtFraction = 1.0

        # Damping
        self.dampingBond = 1.0 # (0-1) Bulk material damping
        self.dampingCollision = 1.0 # (0-2) Elastic vs inelastic conditions
        self.dampingEnvironment = 0.0 # (0-0.1) Damping caused by fluid environment

        # Collisions
        self.collisionEnable = False
        self.collisionSystem = 3
        self.collisionHorizon = 3

        # Features
        self.blendingEnable = False
        self.xMixRadius = 0
        self.yMixRadius = 0
        self.zMixRadius = 0
        self.blendingModel = 0
        self.polyExp = 1
        self.volumeEffectsEnable = False

        # Stop conditions
        self.stopConditionType = StopCondition.NONE
        self.stopConditionValue = 0

        # Equilibrium mode
        self.equilibriumModeEnable = False

        # Environment ################################
        # Boundary conditions
        self.bcNumber = 0
        self.bcRegions = np.zeros((1, 25))

        # Gravity
        self.gravityEnable = True
        self.gravityValue = -9.81
        self.floorEnable = True

        # Thermal
        self.temperatureEnable = True
        self.temperatureBaseValue = 25.0
        self.temperatureVaryEnable = False
        self.temperatureVaryAmplitude = 0.0
        self.temperatureVaryPeriod = 0.0

    # Configure settings


    # Export simulation object to .vxa file for import into simulation engine
    def saveVXA(self, filename, compression=False):
        f = open(filename + '.vxa', 'w+')
        print('Saving file: ' + f.name)

        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        f.write('<VXA Version="' + str(1.1) + '">\n')
        self.writeSimData(f)
        self.writeEnvironmentData(f)
        self.model.writeVXCData(f, compression)
        f.write('</VXA>\n')

        f.close()

    def writeSimData(self, f):
        # Simulator settings
        f.write('<Simulator>\n')
        f.write('  <Integration>\n')
        f.write('    <Integrator>' + str(self.integrator) + '</Integrator>\n')
        f.write('    <DtFrac>' + str(self.dtFraction) + '</DtFrac>\n')
        f.write('  </Integration>\n')
        f.write('  <Damping>\n')
        f.write('    <BondDampingZ>' + str(self.dampingBond) + '</BondDampingZ>\n')
        f.write('    <ColDampingZ>' + str(self.dampingCollision) + '</ColDampingZ>\n')
        f.write('    <SlowDampingZ>' + str(self.dampingEnvironment) + '</SlowDampingZ>\n')
        f.write('  </Damping>\n')
        f.write('  <Collisions>\n')
        f.write('    <SelfColEnabled>' + str(int(self.collisionEnable)) + '</SelfColEnabled>\n')
        f.write('    <ColSystem>' + str(self.collisionSystem) + '</ColSystem>\n')
        f.write('    <CollisionHorizon>' + str(self.collisionHorizon) + '</CollisionHorizon>\n')
        f.write('  </Collisions>\n')
        f.write('  <Features>\n')
        f.write('    <BlendingEnabled>' + str(int(self.blendingEnable)) + '</BlendingEnabled>\n')
        f.write('    <XMixRadius>' + str(self.xMixRadius) + '</XMixRadius>\n')
        f.write('    <YMixRadius>' + str(self.yMixRadius) + '</YMixRadius>\n')
        f.write('    <ZMixRadius>' + str(self.zMixRadius) + '</ZMixRadius>\n')
        f.write('    <BlendModel>' + str(self.blendingModel) + '</BlendModel>\n')
        f.write('    <PolyExp>' + str(self.polyExp) + '</PolyExp>\n')
        f.write('    <VolumeEffectsEnabled>' + str(int(self.volumeEffectsEnable)) + '</VolumeEffectsEnabled>\n')
        f.write('  </Features>\n')
        f.write('  <StopCondition>\n')
        f.write('    <StopConditionType>' + str(self.stopConditionType.value) + '</StopConditionType>\n')
        f.write('    <StopConditionValue>' + str(self.stopConditionValue) + '</StopConditionValue>\n')
        f.write('  </StopCondition>\n')
        f.write('  <EquilibriumMode>\n')
        f.write('    <EquilibriumModeEnabled>' + str(int(self.equilibriumModeEnable)) + '</EquilibriumModeEnabled>\n')
        f.write('  </EquilibriumMode>\n')
        f.write('</Simulator>\n')

    def writeEnvironmentData(self, f):
        # Environment settings
        f.write('<Environment>\n')
        f.write('  <Boundary_Conditions>\n')
        f.write('    <NumBCs>' + str(self.bcNumber) + '</NumBCs>\n')
        f.write('    <FRegion>\n')

        f.write('    </FRegion>\n')
        f.write('  </Boundary_Conditions>\n')
        f.write('  <Gravity>\n')
        f.write('    <GravEnabled>' + str(int(self.gravityEnable)) + '</GravEnabled>\n')
        f.write('    <GravAcc>' + str(self.gravityValue) + '</GravAcc>\n')
        f.write('    <FloorEnabled>' + str(int(self.floorEnable)) + '</FloorEnabled>\n')
        f.write('  </Gravity>\n')
        f.write('  <Thermal>\n')
        f.write('    <TempEnabled>' + str(int(self.temperatureEnable)) + '</TempEnabled>\n')
        f.write('    <TempAmplitude>' + str(self.temperatureVaryAmplitude) + '</TempAmplitude>\n')
        f.write('    <TempBase>' + str(self.temperatureBaseValue) + '</TempBase>\n')
        f.write('    <VaryTempEnabled>' + str(int(self.temperatureVaryEnable)) + '</VaryTempEnabled>\n')
        f.write('    <TempPeriod>' + str(self.temperatureVaryPeriod) + '</TempPeriod>\n')
        f.write('  </Thermal>\n')
        f.write('</Environment>\n')



