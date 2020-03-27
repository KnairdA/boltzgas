import boltzgas.visualizer

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles
from boltzgas.visual import VelocityHistogram, WireBox, Tracer

grid_width = 10
radius = 0.005
char_u = 1120

position, velocity = grid_of_random_velocity_particles(grid_width, radius, char_u)

config = HardSphereSetup(radius, char_u, position, velocity)
gas = HardSphereSimulation(config, opengl = True, t_scale=0.5)

from OpenGL.GL   import *

#tracer = Tracer(gas, int((grid_width**2)/2+grid_width/2))
histogram = VelocityHistogram(gas, [1.1,0], [1,1])

decorations = [ WireBox(0,1,0,1,0,1) ]
instruments = [ histogram ]
windows = [ histogram ]

boltzgas.visualizer.simulate(config, gas, instruments, decorations, windows)
