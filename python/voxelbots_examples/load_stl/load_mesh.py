# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:57:14 2019

@author: daukes
"""
import idealab_tools.plot_tris
import meshio
import numpy
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal

def make_mesh(filename,delete_files=True):

    geo_string=template.format(filename)
    with open('output.geo','w') as f:
        f.writelines(geo_string)
    
    command_string = 'gmsh output.geo -3 -format msh'
    p = subprocess.Popen(command_string)
    p.wait()
    mesh_file = 'output.msh'
    data = meshio.read(mesh_file)
    if delete_files:
        os.remove('output.msh')
        os.remove('output.geo')
    return data

def generate_outer(v):
    kernel = numpy.zeros((3,3,3))
    kernel[:,1,1]=1
    kernel[1,:,1]=1
    kernel[1,1,:]=1
    
    v2 = scipy.signal.convolve(voxels*1,kernel,mode='same')
    v2 = v2.round()
    v2 = v&(v2<len(kernel.nonzero()[0]))
    return v2,kernel

template = '''
Merge "{0}";
Surface Loop(1) = {{1}};
//+
Volume(1) = {{1}};
'''

res = 0

data = make_mesh('tube.stl',True)

points = data.points
ii_tri = data.cells['triangle']
ii_tet = data.cells['tetra']

tris= points[ii_tri]
tets= points[ii_tet]
T=numpy.concatenate((tets,tets[:,:,0:1]*0+1),2)
T_inv = numpy.zeros(T.shape)
for ii,t in enumerate(T):
    T_inv[ii] = numpy.linalg.inv(t).T


points_min = points.min(0)
points_max = points.max(0)

points_min_r = numpy.round(points_min,res)
points_max_r = numpy.round(points_max,res)

xx = numpy.r_[points_min_r[0]:points_max_r[0]:10**(-res)]
yy = numpy.r_[points_min_r[1]:points_max_r[1]:10**(-res)]
zz = numpy.r_[points_min_r[2]:points_max_r[2]:10**(-res)]

xx_mid = (xx[1:]+xx[:-1])/2
yy_mid = (yy[1:]+yy[:-1])/2
zz_mid = (zz[1:]+zz[:-1])/2

xyz_mid = numpy.array(numpy.meshgrid(xx_mid,yy_mid,zz_mid,indexing='ij'))
xyz_mid = xyz_mid.transpose(1,2,3,0)
xyz_mid = xyz_mid.reshape(-1,3)
xyz_mid = numpy.concatenate((xyz_mid,xyz_mid[:,0:1]*0+1),1)

ijk_mid = numpy.array(numpy.meshgrid(numpy.r_[:len(xx_mid)],numpy.r_[:len(yy_mid)],numpy.r_[:len(zz_mid)],indexing='ij'))
ijk_mid = ijk_mid.transpose(1,2,3,0)
ijk_mid2 = ijk_mid.reshape(-1,3)
#
u2 = T_inv.dot(xyz_mid.T)

f1 = ((u2[:,:,:]>=0).sum(1)==4)
f2 = ((u2[:,:,:]<=1).sum(1)==4)
f3 = f1&f2
ii,jj = f3.nonzero()

#print(len(jj))
#print(len(numpy.unique(jj)))

lmn = ijk_mid2[numpy.unique(jj)]

voxels = numpy.zeros(ijk_mid.shape[:3],dtype=numpy.bool)
voxels[lmn[:,0],lmn[:,1],lmn[:,2]]=True
v2,kernel = generate_outer(voxels)
colors = np.empty(v2.shape, dtype=object)
colors[v2] = 'red'

colors2 = np.empty(kernel.shape, dtype=object)
colors2[kernel==1] = 'red'

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(kernel, facecolors=colors2, edgecolor='k')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(v2, facecolors=colors, edgecolor='k')

#for tri in tris:
#    ax.plot3D(tri[(0,1,2,0),0],tri[(0,1,2,0),1],tri[(0,1,2,0),2],'k-')