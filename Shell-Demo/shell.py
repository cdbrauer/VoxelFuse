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
    joint1 = model.import_vox('sample-joint-1.vox')

    # Isolate flexible components and generate outer shell
    flexComponents = model.isolate_material(joint1, 101)
    shell1 = model.difference(model.dilate(flexComponents, 1), flexComponents)

    # Initialize application 1
    app1, w1 = plot.prep()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model.isolate_layer(shell1, 24))

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges = True)
    w1.addItem(mi)

    # Show plot 1
    plot.show(w1, joint1)

    # Save screenshot of plot 1
    #w1.paintGL()
    #w1.grabFrameBuffer().save('shell-fig3.png')

    # Generate inner shell
    shell2 = model.difference(flexComponents, model.erode(flexComponents, 1))

    # Initialize application 2
    w2 = plot.add_widget()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model.isolate_layer(shell2, 24))

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges=True)
    w2.addItem(mi)

    # Show plot 2
    plot.show(w2, flexComponents)

    # Save screenshot of plot 2
    #w2.paintGL()
    #w2.grabFrameBuffer().save('shell-fig4.png')

    app1.exec_()
