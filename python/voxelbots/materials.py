"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""
"""
Material data array

Properties:
  r, g, b - color used to represent material
  process - method used to create parts made of this material
  blur - defines whether material and process support blurring

Processes:
  3dp: 3D Printing
  laser: Laser Cutting
  ins: Inserted component
"""

materials = [{'r': 0,   'g': 0,   'b': 0,   'process': 'nul'},  # 0 - Null
             {'r': 1,   'g': 0,   'b': 0,   'process': '3dp'},  # 1 - 3D printing
             {'r': 0,   'g': 1,   'b': 0,   'process': '3dp'},  # 2 - 3D printing
             {'r': 0,   'g': 0,   'b': 1,   'process': '3dp'},  # 3 - 3D printing
             {'r': 0.5, 'g': 0.5, 'b': 0.5, 'process': 'ins'},  # 4 - inserted component
             {'r': 1,   'g': 1,   'b': 0,   'process': '3dp'}]  # 5 - 3D printing