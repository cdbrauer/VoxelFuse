"""
Functions for manipulating gcode files

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import numpy as np
from typing import List

def import_gcode(filename: str):
    """
    Import the lines of a gcode file to a list

    :param filename: File name with extension
    :return: List of gcode lines
    """
    file = open(filename, 'r')
    print("File opened: "+file.name)
    lines = file.readlines()
    file.close()
    return lines

def export(filename: str, lines: List[str]):
    """
    Export a list of strings to a gcode file.

    :param filename: File name with extension
    :param lines: List of gcode lines
    :return: None
    """
    file = open(filename, 'w')
    print("File created: "+file.name)
    file.writelines(lines)
    file.close()

def remove_to_string(gcode: List[str], string: str):
    """
    Remove all gcode commands before a specified comment string.

    find_voxels can be used before this command to add comments before the
    start of each voxel layer.

    :param gcode: List of gcode lines
    :param string: Comment string to find
    :return: None
    """
    for i in range(len(gcode)):
        if gcode[i][0] != ';':
            gcode[i] = ''
        elif (gcode[i] == string + '\n') or (gcode[i] == ';' + string + '\n'):
            break

def pause_before_voxel(gcode: List[str], voxel: int):
    """
    Insert a pause command (M601) before a specified voxel layer.

    Before using this command, use find_voxels to add comments before the start
    of each voxel layer.

    :param gcode: List of gcode lines
    :param voxel: Voxel layer
    :return: None
    """
    for i in range(len(gcode)):
        if gcode[i] == (';V' + str(voxel) + '\n'):
            gcode.insert(i, 'M601 ;Pause print\n')
            break

def find_voxels(gcode: List[str], voxel_size: float = 1):
    """
    Insert comments before each voxel layer.

    Voxels are determined based on the position of the Z-axis and the specified
    voxel dimension. Comments are formatted as follows:

    ``;V0``

    ``;Start of voxel 0``

    pause_before_voxel references the ``;V0`` line of the comment. This line can
    also be used with remove_to_string to remove initialization code before the
    first voxel layer.

    :param gcode: List of gcode lines
    :param voxel_size: Size of voxels in mm
    :return: None
    """
    voxel = 0

    for i in range(len(gcode)):
        z_index = gcode[i].find(" Z")
        if z_index != -1:
            z = float(gcode[i][z_index+2:-1])
            if z > voxel*voxel_size:
                gcode.insert(i, ';V' + str(voxel) + '\n')
                gcode.insert(i+1, ';Start of voxel ' + str(voxel) + '\n')
                voxel = voxel+1