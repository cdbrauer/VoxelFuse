"""
Plot Class

Initialized from mesh data

** Plot class is deprecated and will be removed in later versions. Please use 'plot' or 'viewer' from the Mesh class instead. **

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import numpy as np
from typing import Tuple
import PyQt5.QtGui as qg
import pyqtgraph.opengl as pgo

class Plot:
    """
    Plot object that can be displayed or exported.
    """

    def __init__(self, mesh, grids: bool = False, drawEdges: bool = True, edgeColor: Tuple[float, float, float, float] = (0, 0, 0, 0.5), positionOffset: Tuple[int, int, int] = (0, 0, 0), viewAngle: Tuple[int, int, int] = (40, 30, 300), resolution: Tuple[int, int] = (1280, 720), name: str = 'Plot 1'):
        """
        Initialize a Plot object from a Mesh object.

        Args:
            mesh: Mesh object to be plotted
            grids: Enable/disable display of XYZ axes and grids
            drawEdges: Enable/disable display of voxel edges
            edgeColor: Set display color of voxel edges
            positionOffset: Offset of the camera target from the center of the model in voxels
            viewAngle: Elevation, Azimuth, and Distance of the camera
            resolution: Window resolution in px
            name: Plot window name
        """
        print("WARNING: Plot class is deprecated and will be removed in later versions. Please use 'plot' or 'viewer' from the Mesh class instead.")
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

        Returns:
            None
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

        Args:
            filename: File name with extension
        
        Returns:
            None
        """
        if self.widget is not None:
            self.widget.paintGL()
            self.widget.grabFrameBuffer().save(filename)
