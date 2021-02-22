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
- MM - material model, 0=linear, 1=linear+failure, 2=bilinear, 3=data
- FM - failure model, 0=stress, 1=strain

### Supported Process Values

- '3dp' -- 3D Printing
- 'laser' -- Laser Cutting
- 'ins' -- Inserted component

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

material_properties =\
    [# Empty space
     {'name':'Empty',    'process':'nul', 'r':0.00, 'g':0.00, 'b':0.00, 'p':0.000, 'v':0, 'E':0, 'G':0, 'Z':0, 'eY':0, 'eF':0, 'SY':0, 'SF':0, 'CTE':0, 'TP':0, 'uS':0, 'uD':0, 'MM':0, 'FM':0},

     # Common materials
     {'name':'R PLA',    'process':'3dp', 'r':1.00, 'g':0.00, 'b':0.00, 'p':1.240, 'v':0.36, 'E':3.5000e9, 'G':1.287e9, 'Z':0, 'eY':70e6, 'eF':73e6, 'SY':0.02, 'SF':0.04, 'CTE':41e-6, 'TP':0, 'uS':0.5, 'uD':0.4, 'MM':0, 'FM':0}, # u is for pla-steel
     {'name':'G PLA',    'process':'3dp', 'r':0.00, 'g':1.00, 'b':0.00, 'p':1.240, 'v':0.36, 'E':3.5000e9, 'G':1.287e9, 'Z':0, 'eY':70e6, 'eF':73e6, 'SY':0.02, 'SF':0.04, 'CTE':41e-6, 'TP':0, 'uS':0.5, 'uD':0.4, 'MM':0, 'FM':0},
     {'name':'B PLA',    'process':'3dp', 'r':0.00, 'g':0.00, 'b':1.00, 'p':1.240, 'v':0.36, 'E':3.5000e9, 'G':1.287e9, 'Z':0, 'eY':70e6, 'eF':73e6, 'SY':0.02, 'SF':0.04, 'CTE':41e-6, 'TP':0, 'uS':0.5, 'uD':0.4, 'MM':0, 'FM':0},
     {'name':'Aluminum', 'process':'ins', 'r':0.60, 'g':0.60, 'b':0.60, 'p':2.710, 'v':0.33, 'E':68.900e9, 'G':26.20e9, 'Z':0, 'eY':241e6, 'eF':262e6, 'SY':0.1, 'SF':0.17, 'CTE':23.4e-6, 'TP':0, 'uS':0.61, 'uD':0.47, 'MM':0, 'FM':0}, # 6061 T6, u is for alu-steel
     {'name':'Rubber',   'process':'3dp', 'r':0.20, 'g':0.20, 'b':0.20, 'p':0.930, 'v':0.50, 'E':1.8180e6, 'G':1.000e9, 'Z':0, 'eY':3.223e6, 'eF':3.164e6, 'SY':1.025, 'SF':1.028, 'CTE':0.1e-6, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':3, 'FM':1}, # u is for rubber-asphalt

     # Hydrogel starting from a cold state
     {'name':'C H-Gel Cold', 'process':'ins', 'r':0.00, 'g':0.60, 'b':0.60, 'p':1.000, 'v':0.4, 'E':5e3,  'G':1.8e3, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':-0.016, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'M H-Gel Cold', 'process':'ins', 'r':0.60, 'g':0.00, 'b':0.60, 'p':1.000, 'v':0.4, 'E':5e3,  'G':1.8e3, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':-0.016, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'Y H-Gel Cold', 'process':'ins', 'r':0.60, 'g':0.60, 'b':0.00, 'p':1.000, 'v':0.4, 'E':5e3,  'G':1.8e3, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':-0.016, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},

     # Hydrogel starting from a hot state
     {'name':'C H-Gel Hot',  'process':'ins', 'r':0.00, 'g':1.00, 'b':1.00, 'p':1.000, 'v':0.4, 'E':30e3, 'G':10.7e3, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':-0.027, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'M H-Gel Hot',  'process':'ins', 'r':1.00, 'g':0.00, 'b':1.00, 'p':1.000, 'v':0.4, 'E':30e3, 'G':10.7e3, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':-0.027, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},
     {'name':'Y H-Gel Hot',  'process':'ins', 'r':1.00, 'g':1.00, 'b':0.00, 'p':1.000, 'v':0.4, 'E':30e3, 'G':10.7e3, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':-0.027, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':0, 'FM':0},

     # Similar properties to rubber, but with lower yield and failure points
     {'name':'Adhesive', 'process':'3dp', 'r':0.30, 'g':0.30, 'b':0.30, 'p':0.930, 'v':0.50, 'E':1.8180e6, 'G':1.000e9, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':0.1e-6, 'TP':0, 'uS':0.9, 'uD':0.8, 'MM':3, 'FM':1},

     # For use as a sensor element - similar hardness to hydrogel, low mass, no thermal expansion
     {'name':'Low Mass', 'process':'3dp', 'r':0.80, 'g':0.60, 'b':0.00, 'p':0.01, 'v':0.50, 'E':5e3, 'G':1.8e3, 'Z':0, 'eY':1.4e6, 'eF':1.301e6, 'SY':0.6606, 'SF':0.6776, 'CTE':0, 'TP':0, 'uS':0, 'uD':0, 'MM':0, 'FM':0}]

# Stress-strain curves for use when 'MM' = 3
ss_data = [{'name':'Empty'},
           {'name':'R PLA'},
           {'name':'G PLA'},
           {'name':'B PLA'},
           {'name':'Aluminum'},
           {'name':'Rubber',
            'strain':[0, 0.1004,   0.2009,   0.3026,   0.4000,   0.4988,   0.6015,   0.7001,   0.8010,   0.9035,   0.9991,   1.0250],
            'stress':[0, 0.3884e6, 0.5753e6, 0.7757e6, 0.9612e6, 1.1710e6, 1.4500e6, 1.7560e6, 2.1300e6, 2.5600e6, 3.0820e6, 3.2230e6]},
           {'name':'C H-Gel Cold'},
           {'name':'M H-Gel Cold'},
           {'name':'Y H-Gel Cold'},
           {'name':'C H-Gel Hot'},
           {'name':'M H-Gel Hot'},
           {'name':'Y H-Gel Hot'},
           {'name':'Adhesive',
            'strain':[0, 0.1006,   0.2004,   0.3000,   0.4003,   0.5005,   0.6004,   0.6606,   0.6738,   0.6776],
            'stress':[0, 0.3609e6, 0.5338e6, 0.6872e6, 0.8617e6, 1.0520e6, 1.2700e6, 1.4000e6, 1.3800e6, 1.3010e6]}]