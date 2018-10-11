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

    # Import models ##############################################################
    model1 = model.import_vox('square-Le.vox')
    model2 = model.import_vox('square-Re.vox')

    # Perform Boolean Operations ###############################################

    # Initialize application
    app1, w1 = plot.prep()
    w2 = plot.add_widget()
    w3 = plot.add_widget()
    w4 = plot.add_widget()
    w5 = plot.add_widget()
    w6 = plot.add_widget()

    # Union ###############################################
    modelBool = model.union(model1, model2)

    # Convert model to mesh data and add to plot
    v, vc, t = mesh.create_from_model(modelBool)
    mi = plot.make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w1.addItem(mi)

    # Show plot 1
    plot.show(w1, modelBool)

    # Save screenshot of plot 1
    w1.paintGL()
    #w1.grabFrameBuffer().save('voxel-tools-bool-fig1.png')

    # Difference ###############################################
    modelBool = model.difference(model1, model2)

    # Convert model to mesh data and add to plot
    v, vc, t = mesh.create_from_model(modelBool)
    mi = plot.make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w2.addItem(mi)

    # Show plot 2
    plot.show(w2, modelBool)

    # Save screenshot of plot 2
    w2.paintGL()
    #w2.grabFrameBuffer().save('voxel-tools-bool-fig2.png')

    # Intersection ###############################################
    modelBool = model.intersection(model1, model2)

    # Convert model to mesh data and add to plot
    v, vc, t = mesh.create_from_model(modelBool)
    mi = plot.make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w3.addItem(mi)

    # Show plot 2
    plot.show(w3, modelBool)

    # Save screenshot of plot 3
    w3.paintGL()
    #w3.grabFrameBuffer().save('voxel-tools-bool-fig3.png')

    # XOR ###############################################
    modelBool = model.xor(model1, model2)

    # Convert model to mesh data and add to plot
    v, vc, t = mesh.create_from_model(modelBool)
    mi = plot.make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w4.addItem(mi)

    # Show plot 2
    plot.show(w4, modelBool)

    # Save screenshot of plot 4
    w4.paintGL()
    #w4.grabFrameBuffer().save('voxel-tools-bool-fig4.png')

    # NOR ###############################################
    modelBool = model.nor(model1, model2)

    # Convert model to mesh data and add to plot
    v, vc, t = mesh.create_from_model(modelBool)
    mi = plot.make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w5.addItem(mi)

    # Show plot 2
    plot.show(w5, modelBool)

    # Save screenshot of plot 3
    w5.paintGL()
    #w5.grabFrameBuffer().save('voxel-tools-bool-fig5.png')

    # NOT ###############################################
    modelBool = model.invert(model1)

    # Convert model to mesh data and add to plot
    v, vc, t = mesh.create_from_model(modelBool)
    mi = plot.make_mi(v, t, vc, drawEdges=True, edgeColor=(1, 1, 1, 0.5))
    w6.addItem(mi)

    # Show plot 2
    plot.show(w6, modelBool)

    # Save screenshot of plot 4
    w6.paintGL()
    #w6.grabFrameBuffer().save('voxel-tools-bool-fig6.png')

    app1.exec_()
