"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import gcode_tools as gcode

gc1 = gcode.import_gcode('flex.gcode')
header = gcode.import_gcode('header-template.gcode')

# Insert tags at the start of each voxel layer
gcode.find_voxels(gc1)

# Remove initialization commands
gcode.remove_to_string(gc1, ';V0')

# Insert pause before voxel 2
gcode.pause_before_voxel(gc1, 2)

# Insert header template
gc1[0:0] = header

gcode.export('new-file.gcode', gc1)

print("Finished")