"""
Copyright 2018-2019
Dan Aukes, Cole Brauer
"""

import numpy as np
import PyQt5.QtGui as qg
import pyqtgraph.opengl as pgo

"""
Plot Class

Initialized from mesh data
"""
class Plot:
    def __init__(self, mesh, grids = False, drawEdges=True, edgeColor=(0, 0, 0, 0.5), positionOffset = (0, 0, 0), viewAngle = (40, 30, 300), resolution = (1280, 720), name = 'Plot 1'):
        self.mesh = mesh
        self.grids = grids
        self.drawEdges = drawEdges
        self.edgeColor = edgeColor
        self.pos = positionOffset
        self.angle = viewAngle
        self.res = resolution
        self.name = name
        self.widget = None

    def show(self):
        mesh_data = pgo.MeshData(vertexes=self.mesh.verts, faces=self.mesh.tris, vertexColors=self.mesh.colors, faceColors=None)
        mesh_item = pgo.GLMeshItem(meshdata=mesh_data, shader='balloon', drawEdges=self.drawEdges, edgeColor=self.edgeColor,
                                   smooth=False, computeNormals=False, glOptions='translucent')
        # mesh_item = pgo.GLMeshItem(meshdata = mesh_data, shader='shaded', drawEdges=False, smooth=True, computeNormals = True, glOptions='opaque')

        widget = pgo.GLViewWidget()
        widget.setBackgroundColor('w')
        widget.addItem(mesh_item)

        if self.grids:
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
        widget.opts['center'] = qg.QVector3D(((self.mesh.model.shape[0])/2)+self.pos[0], ((self.mesh.model.shape[1])/2)+self.pos[1], ((self.mesh.model.shape[2])/2)+self.pos[2])
        widget.opts['elevation'] = self.angle[0]
        widget.opts['azimuth'] = self.angle[1]
        widget.opts['distance'] = self.angle[2]
        widget.resize(self.res[0], self.res[1])

        # Show plot
        widget.setWindowTitle(str(self.name))
        widget.show()

        self.widget = widget

    def export(self, filename):
        if self.widget is not None:
            self.widget.paintGL()
            self.widget.grabFrameBuffer().save(filename)
