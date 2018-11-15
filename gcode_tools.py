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

def pause_at_layer(gcode, layer):
    for i in range(len(gcode)):
        if gcode[i] == (';LAYER:' + str(layer) + '\n'):
            gcode.insert(i + 1, 'M601 ;Pause print\n')
            break
    return 1