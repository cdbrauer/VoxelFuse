# -*- coding: utf-8 -*-
"""
Copyright 2018
Dan Aukes, Cole Brauer

Run with path to .vox file as parameter
"""

import PyQt5.QtGui as qg
import pyqtgraph.opengl as pgo
import sys
import numpy as np
import coloredlogs
from pyvox.parser import VoxParser

def prep():
    application = qg.QApplication(sys.argv)
    widget = pgo.GLViewWidget()    
    return application, widget

def make_mi(vertices, triangles, vertex_colors = None, face_colors = None, drawEdges = False, edgeColor = (1,1,1,1)):
    mesh_data = pgo.MeshData(vertexes = vertices, faces = triangles, vertexColors = vertex_colors, faceColors = face_colors)
    mesh_item = pgo.GLMeshItem(meshdata = mesh_data, shader='balloon', drawEdges=drawEdges, edgeColor = edgeColor, smooth=False, computeNormals = False, glOptions='translucent')
    #mesh_item = pgo.GLMeshItem(meshdata = mesh_data, shader='shaded', drawEdges=False, smooth=True, computeNormals = True, glOptions='opaque')
    return mesh_item

def isolate_material(input_model, material):
    model = np.copy(input_model)

    # Loop through model data
    for x in range(len(model[0, 0, :])):
        for y in range(len(model[:, 0, 0])):
            for z in range(len(model[0, :, 0])):
                # If voxel does not contain the requested material
                if model[y, z, x] != material:
                    # Clear voxel
                    model[y, z, x] = 0

    return model

def model_to_mesh(model):
    # Initialize arrays
    verts = []
    verts_colors = []
    tris = []
    vi = 0  # Tracks starting index for defining triangles

    # Loop through model data
    for x in range(len(model[0, 0, :])):
        for y in range(len(model[:, 0, 0])):
            for z in range(len(model[0, :, 0])):
                # If voxel is not empty
                if model[y, z, x] != 0:
                    # Set color based on material
                    if model[y, z, x] == 236:  # Blue
                        voxel_color = [0, 0, 1, 1]
                    elif model[y, z, x] == 217:  # Red
                        voxel_color = [1, 0, 0, 1]
                    elif model[y, z, x] == 226:  # Green
                        voxel_color = [0, 1, 0, 1]
                    else:  # Yellow - default if material not recognized
                        voxel_color = [1, 1, 0, 1]

                    # Add voxel to mesh item arrays
                    # Add cube vertices
                    verts.append([x + 0.5, y + 0.5, z + 0.5])
                    verts.append([x + 0.5, y - 0.5, z + 0.5])
                    verts.append([x - 0.5, y + 0.5, z + 0.5])
                    verts.append([x - 0.5, y - 0.5, z + 0.5])
                    verts.append([x + 0.5, y + 0.5, z - 0.5])
                    verts.append([x + 0.5, y - 0.5, z - 0.5])
                    verts.append([x - 0.5, y + 0.5, z - 0.5])
                    verts.append([x - 0.5, y - 0.5, z - 0.5])

                    # Apply color to all vertices
                    for i in range(0, 8):
                        verts_colors.append(voxel_color)

                    # Add face triangles
                    tris.append([vi + 0, vi + 1, vi + 2])
                    tris.append([vi + 1, vi + 2, vi + 3])
                    tris.append([vi + 0, vi + 1, vi + 4])
                    tris.append([vi + 1, vi + 4, vi + 5])
                    tris.append([vi + 0, vi + 2, vi + 4])
                    tris.append([vi + 2, vi + 4, vi + 6])
                    tris.append([vi + 2, vi + 3, vi + 6])
                    tris.append([vi + 3, vi + 6, vi + 7])
                    tris.append([vi + 1, vi + 3, vi + 5])
                    tris.append([vi + 3, vi + 5, vi + 7])
                    tris.append([vi + 4, vi + 5, vi + 6])
                    tris.append([vi + 5, vi + 6, vi + 7])

                    # Increment index by 8 vertices
                    vi = vi + 8

    verts = np.array(verts)
    verts_colors = np.array(verts_colors)
    tris = np.array(tris)

    return verts, verts_colors, tris

def show_plot(widget):
    # Add grids
    gx = pgo.GLGridItem()
    gx.setSize(x=50, y=50, z=50)
    gx.rotate(90, 0, 1, 0)
    gx.translate(-0.5, 24.5, 24.5)
    widget.addItem(gx)
    gy = pgo.GLGridItem()
    gy.setSize(x=50, y=50, z=50)
    gy.rotate(90, 1, 0, 0)
    gy.translate(24.5, -0.5, 24.5)
    widget.addItem(gy)
    gz = pgo.GLGridItem()
    gz.setSize(x=50, y=50, z=50)
    gz.translate(24.5, 24.5, -0.5)
    widget.addItem(gz)

    # Add axes
    ptsx = np.array([[-0.5, -0.5, -0.5], [50, -0.5, -0.5]])
    pltx = pgo.GLLinePlotItem(pos=ptsx, color=(1, 0, 0, 1), width=1, antialias=True)
    widget.addItem(pltx)
    ptsy = np.array([[-0.5, -0.5, -0.5], [-0.5, 50, -0.5]])
    plty = pgo.GLLinePlotItem(pos=ptsy, color=(0, 1, 0, 1), width=1, antialias=True)
    widget.addItem(plty)
    ptsz = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, 50]])
    pltz = pgo.GLLinePlotItem(pos=ptsz, color=(0, 0, 1, 1), width=1, antialias=True)
    widget.addItem(pltz)

    # Set plot options
    widget.opts['center'] = qg.QVector3D((len(model1[0, 0, :])) / 2, (len(model1[:, 0, 0])) / 2, (len(model1[0, :, 0])) / 2)
    widget.opts['elevation'] = 40
    widget.opts['azimuth'] = 30
    widget.opts['distance'] = 60
    widget.resize(1920 / 2, 1080 / 2)

    # Show plot
    widget.show()
    return 1

    
if __name__=='__main__':
    coloredlogs.install(level='DEBUG')

    # Import model ##############################################################
    m = VoxParser(sys.argv[1]).parse()
    model1 = m.to_dense()
    model1 = np.flip(model1, 1)

    # Initialize application 1
    app1, w1 = prep()

    # Convert model to mesh data
    v, vc, t = model_to_mesh(model1)

    # Create mesh item and add to plot
    mi = make_mi(v, t, vc, drawEdges = True, edgeColor=(1,1,1,0.5))
    w1.addItem(mi)

    # Show plot 1
    show_plot(w1)

    # Save screenshot of plot 1
    w1.paintGL()
    w1.grabFrameBuffer().save('voxel-tools-fig1.png')


    # Isolate flexible components ###############################################
    modelR = isolate_material(model1, 217)

    # Initialize application 2
    app2, w2 = prep()

    # Convert model to mesh data
    v, vc, t = model_to_mesh(modelR)

    # Create mesh item and add to plot
    mi = make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w2.addItem(mi)

    # Show plot 2
    show_plot(w2)

    # Save screenshot of plot 2
    w2.paintGL()
    w2.grabFrameBuffer().save('voxel-tools-fig2.png')

    app2.exec_()
