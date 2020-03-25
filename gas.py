import boltzgas.visualizer

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles
from boltzgas.visual import VelocityHistogram, ColoredBox

grid_width = 40
radius = 0.002
char_u = 1120

position, velocity = grid_of_random_velocity_particles(grid_width, radius, char_u)
velocity[:,:] = 0
velocity[0,0] = 10*char_u
velocity[0,1] = 4*char_u

config = HardSphereSetup(radius, char_u, position, velocity)
gas = HardSphereSimulation(config, opengl = True, t_scale = 0.5)

histogram = VelocityHistogram(gas, [1.1,0], [1,1])

decorations = [ ColoredBox([0,0], [1,1], (0.2,0.2,0.2)) ]
instruments = [ histogram ]
windows = [ histogram ]

boltzgas.visualizer.simulate(config, gas, instruments, decorations, windows)
