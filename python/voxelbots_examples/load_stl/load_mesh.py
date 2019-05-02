# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:57:14 2019

@author: daukes
"""
import idealab_tools.plot_tris
import meshio
import numpy
#import meshio.gmsh_io

asdf = meshio.read('cylinder.msh')
points = asdf.points
ii_tri = asdf.cells['triangle']
ii_tet = asdf.cells['tetra']

tets= points[ii_tet]
T = numpy.array([tets[:,1]-tets[:,0],tets[:,2]-tets[:,0],tets[:,3]-tets[:,0]])
T = T.transpose(1,0,2)

T_inv = numpy.zeros(T.shape)
for ii,t in enumerate(T):
    T_inv[ii] = numpy.linalg.inv(t)
#idealab_tools.plot_tris.plot_tris(points,tris)

res = 0

points_min = points.min(0)
points_max = points.max(0)

points_min_r = numpy.round(points_min,res)
points_max_r = numpy.round(points_max,res)

xx = numpy.r_[points_min_r[0]:points_max_r[0]:10**(-res)]
yy = numpy.r_[points_min_r[1]:points_max_r[1]:10**(-res)]
zz = numpy.r_[points_min_r[2]:points_max_r[2]:10**(-res)]

xx_mid = xx[1:]-xx[:-1]
yy_mid = yy[1:]-yy[:-1]
zz_mid = zz[1:]-zz[:-1]
#xx = numpy.r_[1:3]
#yy = numpy.r_[4:7]
#zz = numpy.r_[8:12]



xyz_mid = numpy.array(numpy.meshgrid(xx_mid,yy_mid,zz_mid,indexing='ij'))
xyz_mid = xyz_mid.transpose(1,2,3,0)
xyz_mid = xyz_mid.reshape(-1,3)

u = T.dot(xyz_mid.T)
