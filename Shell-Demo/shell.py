# -*- coding: utf-8 -*-
"""
Copyright 2018
Dan Aukes, Cole Brauer
"""

import coloredlogs
import model_tools as model
import mesh_tools as mesh
import opengl_plot_tools as plot
  
if __name__=='__main__':
    coloredlogs.install(level='DEBUG')

    # Import model ##############################################################
    joint1 = model.import_vox('joint5.vox')

    shell1 = model.shell_xy(joint1, 1, 226)

    # Initialize application 1
    app1, w1 = plot.prep()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model.union(shell1, joint1))

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges = True)
    w1.addItem(mi)

    # Show plot 1
    plot.show(w1, joint1)

    # Save screenshot of plot 1
    #w1.paintGL()
    #w1.grabFrameBuffer().save('shell-fig1.png')

    # Isolate flexible components and generate shell ###############################################
    flexComponents = model.isolate_material(joint1, 217)
    shell2 = model.shell(flexComponents, 1, 226)

    # Initialize application 2
    w2 = plot.add_widget()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model.union(shell2, flexComponents))

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges=True)
    w2.addItem(mi)

    # Show plot 2
    plot.show(w2, flexComponents)

    # Save screenshot of plot 2
    #w2.paintGL()
    #w2.grabFrameBuffer().save('shell-fig4.png')

    app1.exec_()
