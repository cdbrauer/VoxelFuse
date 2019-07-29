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
    def __init__(self, mesh, grids = False, drawEdges=True, edgeColor=(0, 0, 0, 0.5)):
        self.mesh = mesh
        self.grids = grids
        self.drawEdges = drawEdges
        self.edgeColor = edgeColor
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
        widget.opts['center'] = qg.QVector3D((len(self.mesh.model[0, 0, :, 0])) / 2, (len(self.mesh.model[:, 0, 0, 0])) / 2, ((len(self.mesh.model[0, :, 0, 0])) / 2))
        widget.opts['elevation'] = 40 #30
        widget.opts['azimuth'] = 30 #30
        widget.opts['distance'] = 50 #50
        widget.resize(1440 / 1.5, 1080 / 1.5)

        # Show plot
        #widget.setWindowTitle(str())
        widget.show()

        self.widget = widget

    def export(self, filename):
        if self.widget is not None:
            self.widget.paintGL()
            self.widget.grabFrameBuffer().save(filename)
