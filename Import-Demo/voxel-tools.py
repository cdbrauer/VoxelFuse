# -*- coding: utf-8 -*-
"""
Copyright 2018
Dan Aukes, Cole Brauer

Run with path to .vox file as parameter
"""

import coloredlogs
import model_tools as model
import mesh_tools as mesh
import opengl_plot_tools as plot

if __name__=='__main__':
    coloredlogs.install(level='DEBUG')

    # Import model ##############################################################
    model1 = model.import_vox('sample-joint-1.vox')

    # Initialize application 1
    app1, w1 = plot.prep()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(model1)

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges = True, edgeColor=(1,1,1,0.5))
    w1.addItem(mi)

    # Show plot 1
    plot.show(w1, model1)

    # Save screenshot of plot 1
    #w1.paintGL()
    #w1.grabFrameBuffer().save('voxel-tools-fig1.png')


    # Isolate flexible components ###############################################
    modelFlex = model.isolate_material(model1, 101)

    # Initialize application 2
    app2, w2 = plot.prep()

    # Convert model to mesh data
    v, vc, t = mesh.create_from_model(modelFlex)

    mesh.export("flex.stl", v, t)

    # Create mesh item and add to plot
    mi = plot.make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w2.addItem(mi)

    # Show plot 2
    plot.show(w2, modelFlex)

    # Save screenshot of plot 2
    #w2.paintGL()
    #w2.grabFrameBuffer().save('voxel-tools-fig2.png')

    app2.exec_()
