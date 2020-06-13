"""
Material data array

Each material should be formatted as a dictionary item with
the following keys.

### Keys

- name - display name of the material
- process - method used to create parts made of this material
- r, g, b - color used to represent material
- p - density, g/cm^3
- v - poisson's ratio
- E - elastic modulus, Pa
- G - shear modulus, Pa
- Z - plastic modulus, Pa
- eY - yield stress, Pa
- eF - fail stress, Pa
- SY - yield strain, m/m
- SF - fail strain, m/m
- CTE - coefficient of thermal expansion, 1/deg C
- TP - temp phase, rad
- uS - coefficient of static friction
- uD - coefficient of dynamic friction
- MM - material model, 0=linear, 1=linear+failure, 2=bilinear
- FM - failure model, 0=stress, 1=strain

### Supported Process Values

- '3dp' -- 3D Printing
- 'laser' -- Laser Cutting
- 'ins' -- Inserted component

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

material_properties =\
    [{'name':'Empty',        'process':'nul', 'r':0.00, 'g':0.00, 'b':0.00, 'p':0.000, 'v':0, 'E':0, 'G':0, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':0, 'TP':0, 'uS':0, 'uD':0, 'MM':0, 'FM':0},
     {'name':'R PLA',        'process':'3dp', 'r':1.00, 'g':0.00, 'b':0.00, 'p':1.240, 'v':0.36, 'E':3.5e9, 'G':1.287e9, 'Z':0, 'eY':70e6, 'eF':73e6, 'SY':0.02, 'SF':0.04, 'CTE':41e-6, 'TP':0, 'uS':0.5, 'uD':0.4, 'MM':0, 'FM':0}, # u is for pla-steel
     {'name':'G PLA',        'process':'3dp', 'r':0.00, 'g':1.00, 'b':0.00, 'p':1.240, 'v':0.36, 'E':3.5e9, 'G':1.287e9, 'Z':0, 'eY':70e6, 'eF':73e6, 'SY':0.02, 'SF':0.04, 'CTE':41e-6, 'TP':0, 'uS':0.5, 'uD':0.4, 'MM':0, 'FM':0},
     {'name':'B PLA',        'process':'3dp', 'r':0.00, 'g':0.00, 'b':1.00, 'p':1.240, 'v':0.36, 'E':3.5e9, 'G':1.287e9, 'Z':0, 'eY':70e6, 'eF':73e6, 'SY':0.02, 'SF':0.04, 'CTE':41e-6, 'TP':0, 'uS':0.5, 'uD':0.4, 'MM':0, 'FM':0},
     {'name':'Aluminum',     'process':'ins', 'r':0.60, 'g':0.60, 'b':0.60, 'p':2.710, 'v':0.33, 'E':68.9e9, 'G':26.2e9, 'Z':0, 'eY':241e6, 'eF':262e6, 'SY':0.1, 'SF':0.17, 'CTE':23.4e-6, 'TP':0, 'uS':0.61, 'uD':0.47, 'MM':0, 'FM':0}, # 6061 T6, u is for alu-steel
     {'name':'Rubber',       'process':'3dp', 'r':0.20, 'g':0.20, 'b':0.20, 'p':0.930, 'v':0.5, 'E':0.518e6, 'G':1.0e9, 'Z':0, 'eY':3.223e6, 'eF':3.164e6, 'SY':1.025, 'SF':1.028, 'CTE':0.1e-6, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':1}, # u is for rubber-asphalt
     {'name':'C H-Gel Cold', 'process':'ins', 'r':0.00, 'g':0.60, 'b':0.60, 'p':1.000, 'v':0.4, 'E':5e6, 'G':1.8e6, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':-0.02, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0}, # TODO: confirm CTE
     {'name':'M H-Gel Cold', 'process':'ins', 'r':0.60, 'g':0.00, 'b':0.60, 'p':1.000, 'v':0.4, 'E':5e6, 'G':1.8e6, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':-0.02, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'Y H-Gel Cold', 'process':'ins', 'r':0.60, 'g':0.60, 'b':0.00, 'p':1.000, 'v':0.4, 'E':5e6, 'G':1.8e6, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':-0.02, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'C H-Gel Hot',  'process':'ins', 'r':0.00, 'g':1.00, 'b':1.00, 'p':1.000, 'v':0.4, 'E':30e6, 'G':10.7e6, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':-0.02, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'M H-Gel Hot',  'process':'ins', 'r':1.00, 'g':0.00, 'b':1.00, 'p':1.000, 'v':0.4, 'E':30e6, 'G':10.7e6, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':-0.02, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'Y H-Gel Hot',  'process':'ins', 'r':1.00, 'g':1.00, 'b':0.00, 'p':1.000, 'v':0.4, 'E':30e6, 'G':10.7e6, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':-0.02, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0}]