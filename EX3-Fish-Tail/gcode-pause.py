"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import gcode_tools as gcode

#gc1 = gcode.import_gcode('joint-2.gcode')
gc1 = gcode.import_gcode('tail-holder.gcode')

# Insert tags at the start of each voxel layer
gcode.find_voxels(gc1)

# Insert pause before voxel
gcode.pause_before_voxel(gc1, 43)

#gcode.export('joint-2-pause.gcode', gc1)
gcode.export('tail-holder-pause.gcode', gc1)

print("Finished")