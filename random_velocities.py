import boltzgas.visualizer

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles
from boltzgas.visual import VelocityHistogram, WireBox, Tracer

dimension = 3
grid_width = 10
radius = 0.005
char_u = 1120

position, velocity = grid_of_random_velocity_particles(dimension, grid_width, radius, char_u)

config = HardSphereSetup(radius, char_u, position, velocity)
gas = HardSphereSimulation(config, opengl = True, t_scale=0.5)

histogram = VelocityHistogram(gas, [1.1,0], [1,1])

instruments = [ WireBox(0,1,0,1,0,dimension == 3 if 1 else 0), histogram ]

boltzgas.visualizer.simulate(config, gas, instruments)
