# -*- coding: utf-8 -*-
"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import coloredlogs
import model_tools as model
import mesh_tools as mesh
import opengl_plot_tools as plot
import numpy as np
  
if __name__=='__main__':
    coloredlogs.install(level='DEBUG')

    # Import model
    joint1 = model.import_vox('Shell-Demo/sample-joint-1.vox')

    # Dilate
    model1 = model.isolate_layer(model.dilate(joint1, 1), 24)

    # Initialize application 1
    app1, w1 = plot.prep()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model1)

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges = True)
    w1.addItem(mi)

    # Show plot 1
    plot.show(w1, model1, grids=True)

    # Save screenshot of plot 1
    w1.paintGL()
    w1.grabFrameBuffer().save('Shell-Demo/dilate-fig1.png')

    # Erode
    model2 = model.isolate_layer(model.erode(joint1, 1), 24)

    # Initialize application 2
    w2 = plot.add_widget()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model2)

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges=True)
    w2.addItem(mi)

    # Show plot 2
    plot.show(w2, model2, grids=True)

    # Save screenshot of plot 2
    w2.paintGL()
    w2.grabFrameBuffer().save('Shell-Demo/erode-fig1.png')

    app1.exec_()
