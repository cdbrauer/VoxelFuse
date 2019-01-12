"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
import PyQt5.QtGui as qg
import pyqtgraph.opengl as pgo
import sys

def prep():
    application = qg.QApplication(sys.argv)
    widget = pgo.GLViewWidget()
    return application, widget

def add_widget():
    widget = pgo.GLViewWidget()
    return widget

def make_mi(vertices, triangles, vertex_colors = None, face_colors = None, drawEdges = False, edgeColor = (1,1,1,0.5)):
    mesh_data = pgo.MeshData(vertexes = vertices, faces = triangles, vertexColors = vertex_colors, faceColors = face_colors)
    mesh_item = pgo.GLMeshItem(meshdata = mesh_data, shader='balloon', drawEdges=drawEdges, edgeColor = edgeColor, smooth=False, computeNormals = False, glOptions='translucent')
    #mesh_item = pgo.GLMeshItem(meshdata = mesh_data, shader='shaded', drawEdges=False, smooth=True, computeNormals = True, glOptions='opaque')
    return mesh_item

def show(widget, center_model, grids = False):
    if grids:
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