# Import Library
import voxelfuse as vf

# Start Application
if __name__=='__main__':
    # Create Models
    model = vf.sphere(5)

    # Process Models
    modelResult = model.dilate(3, vf.Axes.XY)

    # Create and Export Mesh
    mesh = vf.Mesh.fromVoxelModel(modelResult)
    mesh.export('model-result.stl')

    # Create Plot
    mesh.viewer(grids=True, name='mesh')