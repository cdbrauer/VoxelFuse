# Import Library
import voxelfuse as vf

# Start Application
if __name__=='__main__':
    # Create Models
    model = vf.sphere(5)

    # Process Models
    modelResult = model.dilate(3, vf.Axes.XY)
    modelResult = modelResult.translate((0, 0, 20))

    # Create simulation and launch
    simulation = vf.Simulation(modelResult)
    simulation.runSimVoxCad()