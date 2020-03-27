import boltzgas.visualizer

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles
from boltzgas.visual import VelocityHistogram, WireBox

dimension = 2
grid_width = 30
radius = 0.002
char_u = 1120

position, velocity = grid_of_random_velocity_particles(dimension, grid_width, radius, char_u)
velocity[:,:] = 0
velocity[0,0] = 5*char_u
velocity[0,1] = 4*char_u
if dimension == 3:
    velocity[0,2] = 3*char_u

config = HardSphereSetup(radius, char_u, position, velocity)
gas = HardSphereSimulation(config, opengl = True, t_scale = 0.5)

histogram = VelocityHistogram(gas, [1.1,0], [1,1])

instruments = [ WireBox(0,1,0,1,0,dimension == 3 if 1 else 0), histogram ]

boltzgas.visualizer.simulate(config, gas, instruments)
