import boltzgas.visualizer

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles
from boltzgas.visual import VelocityHistogram, ColoredBox, Tracer

grid_width = 50
radius = 0.002
char_u = 1120

position, velocity = grid_of_random_velocity_particles(grid_width, radius, char_u)

config = HardSphereSetup(radius, char_u, position, velocity)
gas = HardSphereSimulation(config, opengl = True)

tracer = Tracer(gas, int((grid_width**2)/2+grid_width/2))
histogram = VelocityHistogram(gas, [1.1,0], [1,1])

decorations = [ ColoredBox([0,0], [1,1], (0.2,0.2,0.2)), tracer ]
instruments = [ histogram, tracer ]
windows = [ histogram ]

boltzgas.visualizer.simulate(config, gas, instruments, decorations, windows)
