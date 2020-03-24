from OpenGL.GL import *
from OpenGL.GLUT import *

import numpy as np

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles
from boltzgas.visual import View, VelocityHistogram, Tracer, ColoredBox

glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowPosition(0, 0)
glutCreateWindow("BoltzGas")

grid_width = 32
radius = 0.004
char_u = 1120

position, velocity = grid_of_random_velocity_particles(grid_width, radius, char_u)
velocity[:,:] = 0
velocity[0,0] = 10*char_u
velocity[0,1] = 4*char_u

config = HardSphereSetup(radius, char_u, position, velocity)
gas = HardSphereSimulation(config, opengl = True, t_scale = 0.1)

tracer = Tracer(gas, 4)
histogram = VelocityHistogram(gas, [1.1,0], [1,1])
histogram.setup()
view = View(gas, [ ColoredBox([0,0], [1,1], (0.2,0.2,0.2)), tracer ], [ histogram ])

active = False

def on_display():
    if active:
        for i in range(0,5):
            gas.evolve()

        tracer.update()
        histogram.update()

    view.display()

def on_reshape(width, height):
    view.reshape(width, height)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

def on_keyboard(key, x, y):
    global active
    active = not active

def on_close():
    histogram.pool.shutdown()

glutDisplayFunc(on_display)
glutReshapeFunc(on_reshape)
glutTimerFunc(10, on_timer, 10)
glutKeyboardFunc(on_keyboard)
glutCloseFunc(on_close)

if __name__ == "__main__":
    glutMainLoop()
