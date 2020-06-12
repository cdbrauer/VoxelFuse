"""
Plot Class

Initialized from mesh data

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import numpy as np
import PyQt5.QtGui as qg
import pyqtgraph.opengl as pgo
from typing import Tuple

class Plot:
    """
    Plot object that can be displayed or exported.
    """

    def __init__(self, mesh, grids: bool = False, drawEdges: bool = True, edgeColor: Tuple[float, float, float, float] = (0, 0, 0, 0.5), positionOffset: Tuple[int, int, int] = (0, 0, 0), viewAngle: Tuple[int, int, int] = (40, 30, 300), resolution: Tuple[int, int] = (1280, 720), name: str = 'Plot 1'):
        """
        Initialize a Plot object from a Mesh object.

        :param mesh: Mesh object to be plotted
        :param grids: Enable/disable display of XYZ axes and grids
        :param drawEdges: Enable/disable display of voxel edges
        :param edgeColor: Set display color of voxel edges
        :param positionOffset: Offset of the camera target from the center of the model in voxels
        :param viewAngle: Elevation, Azimuth, and Distance of the camera
        :param resolution: Number of voxels per mm
        :param name: Plot window name
        """
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
        """
        Generate an display widget based on the plot settings and display it.

        ----

        Example:

        ``app1 = qg.QApplication(sys.argv)``

        ``plot1 = Plot(mesh1)``

        ``plot1.show()``

        ``app1.processEvents()``

        ``app1.exec_()``

        ----

        :return: None
        """
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
        widget.opts['center'] = qg.QVector3D(((self.mesh.model.shape[0]/self.mesh.res)/2)+self.pos[0], ((self.mesh.model.shape[1]/self.mesh.res)/2)+self.pos[1], ((self.mesh.model.shape[2]/self.mesh.res)/2)+self.pos[2])
        widget.opts['elevation'] = self.angle[0]
        widget.opts['azimuth'] = self.angle[1]
        widget.opts['distance'] = self.angle[2]
        widget.resize(self.res[0], self.res[1])

        # Show plot
        widget.setWindowTitle(str(self.name))
        widget.show()

        self.widget = widget

    def export(self, filename: str):
        """
        Save a screenshot of the last generated plot widget with the specified name and file format.

        ----

        Example:

        ``app1 = qg.QApplication(sys.argv)``

        ``plot1 = Plot(mesh1)``

        ``plot1.show()``

        ``app1.processEvents()``

        ``plot1.export('result.png')``

        ``app1.exec_()``

        ----

        :param filename: File name with extension
        :return: None
        """
        if self.widget is not None:
            self.widget.paintGL()
            self.widget.grabFrameBuffer().save(filename)
