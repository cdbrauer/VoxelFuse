"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import gcode_tools as gcode

gc1 = gcode.import_gcode('flex.gcode')
header = gcode.import_gcode('header-template.gcode')

# Remove initialization commands
gcode.remove_to_string(gc1, ';LAYER:0')

# Insert pause before layer 2
gcode.pause_at_layer(gc1, 2)

# Insert header template
gc1[0:0] = header

gcode.export('new-file.gcode', gc1)

print("Finished")