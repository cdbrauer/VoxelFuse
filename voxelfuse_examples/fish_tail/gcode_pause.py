"""
Copyright 2019
Dan Aukes, Cole Brauer

Program adds a pause to the input gcode file at the specified voxel layer
"""

import voxelfuse.gcode_tools as gcode

if __name__=='__main__':
    #gc1 = gcode.import_gcode('joint-2.gcode')
    gc1 = gcode.import_gcode('tail-holder.gcode')

    # Insert tags at the start of each voxel layer
    gcode.find_voxels(gc1)

    # Insert pause before voxel
    gcode.pause_before_voxel(gc1, 43)

    #gcode.export('joint-2-pause.gcode', gc1)
    gcode.export('tail-holder-pause.gcode', gc1)

    print("Finished")