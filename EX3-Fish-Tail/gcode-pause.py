"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import gcode_tools as gcode

gc1 = gcode.import_gcode('joint.gcode')

# Insert tags at the start of each voxel layer
gcode.find_voxels(gc1)

# Insert pause before voxel 51
gcode.pause_before_voxel(gc1, 51)

gcode.export('joint-pause.gcode', gc1)

print("Finished")