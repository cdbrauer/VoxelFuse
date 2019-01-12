"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import numpy as np

def import_gcode(filename):
    file = open(filename, 'r')
    print("File opened: "+file.name)
    lines = file.readlines()
    file.close()
    return lines

def export(filename, lines):
    file = open(filename, 'w')
    print("File created: "+file.name)
    file.writelines(lines)
    file.close()
    return 1

def remove_to_string(gcode, string):
    for i in range(len(gcode)):
        if gcode[i][0] != ';':
            gcode[i] = ''
        elif gcode[i] == string + '\n':
            break
    return 1

def pause_before_voxel(gcode, voxel):
    for i in range(len(gcode)):
        if gcode[i] == (';V' + str(voxel) + '\n'):
            gcode.insert(i, 'M601 ;Pause print\n')
            break
    return 1

def find_voxels(gcode, voxel_size = 1):
    voxel = 0

    for i in range(len(gcode)):
        z_index = gcode[i].find(" Z")
        if z_index != -1:
            z = float(gcode[i][z_index+2:-1])
            if z > voxel*voxel_size:
                gcode.insert(i, ';V' + str(voxel) + '\n')
                gcode.insert(i+1, ';Start of voxel ' + str(voxel) + '\n')
                voxel = voxel+1
    return 1