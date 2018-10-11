# -*- coding: utf-8 -*-
"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import numpy as np
import PyQt5.QtGui as qg
import pyqtgraph.opengl as pgo
import sys
import coloredlogs
from pyvox.parser import VoxParser

import model_tools as model
import mesh_tools as mesh

def prep():
    application = qg.QApplication(sys.argv)
    widget = pgo.GLViewWidget()    
    return application, widget

def make_mi(vertices, triangles, vertex_colors = None, face_colors = None, drawEdges = False, edgeColor = (1,1,1,1)):
    mesh_data = pgo.MeshData(vertexes = vertices, faces = triangles, vertexColors = vertex_colors, faceColors = face_colors)
    mesh_item = pgo.GLMeshItem(meshdata = mesh_data, shader='balloon', drawEdges=drawEdges, edgeColor = edgeColor, smooth=False, computeNormals = False, glOptions='translucent')
    #mesh_item = pgo.GLMeshItem(meshdata = mesh_data, shader='shaded', drawEdges=False, smooth=True, computeNormals = True, glOptions='opaque')
    return mesh_item

def show_plot(widget, center_model):
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
    widget.opts['center'] = qg.QVector3D((len(center_model[0, 0, :])) / 2, (len(center_model[:, 0, 0])) / 2, (len(center_model[0, :, 0])) / 2)
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
    # m = VoxParser(sys.argv[1]).parse()
    m = VoxParser('joint5.vox').parse()
    joint1 = m.to_dense()
    joint1 = np.flip(joint1, 1)

    outsideShell = model.shell_outside(joint1)

    # Initialize application 1
    app1, w1 = prep()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model.union(joint1, outsideShell))

    # Create mesh item and add to plot
    mi = make_mi(v, t, vc, drawEdges = True, edgeColor=(1,1,1,0.5))
    w1.addItem(mi)

    # Show plot 1
    show_plot(w1, joint1)

    # Save screenshot of plot 1
    #w1.paintGL()
    #w1.grabFrameBuffer().save('shell-fig1.png')

    # Isolate flexible components ###############################################
    flexComponents = model.isolate_material(joint1, 217)
    outsideShell = model.shell_outside(flexComponents)

    # Initialize application 2
    app2, w2 = prep()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model.union(flexComponents, outsideShell))

    # Create mesh item and add to plot
    mi = make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w2.addItem(mi)

    # Show plot 2
    show_plot(w2, flexComponents)

    # Save screenshot of plot 2
    #w2.paintGL()
    #w2.grabFrameBuffer().save('shell-fig2.png')

    app2.exec_()
