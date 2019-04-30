# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:57:14 2019

@author: daukes
"""
import idealab_tools.plot_tris
import meshio
#import meshio.gmsh_io

asdf = meshio.read('cylinder.msh')
points = asdf.points
tris = asdf.cells['triangle']
idealab_tools.plot_tris.plot_tris(points,tris)