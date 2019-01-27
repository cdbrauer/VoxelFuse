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
  3dpl: 3D Printing with Laser Cleanup - laser step to improve tolerances
  ins: Inserted component
"""

materials = [{'r': 1, 'g': 0, 'b': 0, 'process': '3dp', 'blur': True},
             {'r': 0, 'g': 1, 'b': 0, 'process': '3dpl', 'blur': True},
             {'r': 0, 'g': 0, 'b': 1, 'process': '3dp', 'blur': True},
             {'r': 0.5, 'g': 0.5, 'b': 0.5, 'process': 'ins', 'blur': False},
             {'r': 1, 'g': 1, 'b': 0, 'process': '3dp', 'blur': False}]