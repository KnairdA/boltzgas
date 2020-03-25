from OpenGL.GL import *
from OpenGL.GLUT import *

import numpy as np

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles
from boltzgas.visual import View, VelocityHistogram, Tracer, ColoredBox

grid_width = 40
radius = 0.002
char_u = 1120

class SimulationController:
    def __init__(self, gas, instruments):
        self.running = False
        self.gas = gas
        self.instruments = instruments

    def isRunning(self):
        return self.running

    def run(self):
        self.running = True

    def pause(self):
        self.running = False

    def evolve(self):
        if self.running:
            for i in range(0,5):
                self.gas.evolve()

            for instrument in self.instruments:
                instrument.update()

    def shutdown(self):
        self.pause()

        for instrument in self.instruments:
            try:
                instrument.shutdown()
            except AttributeError:
                return # Doesn't matter, shutdown is optional


def make_display_handler(controller, view):
    def on_display():
        controller.evolve()
        view.display()

    return on_display

def make_reshape_handler(view):
    def on_reshape(width, height):
        view.reshape(width, height)

    return on_reshape

def make_timer():
    def on_timer(t):
        glutTimerFunc(t, on_timer, t)
        glutPostRedisplay()

    return on_timer

def make_keyboard_handler(controller):
    def on_keyboard(key, x, y):
        if controller.isRunning():
            controller.pause()
        else:
            controller.run()

    return on_keyboard

def make_close_handler(controller):
    def on_close():
        controller.shutdown()

    return on_close

if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowPosition(0, 0)
    glutCreateWindow("BoltzGas")

    position, velocity = grid_of_random_velocity_particles(grid_width, radius, char_u)
    velocity[:,:] = 0
    velocity[0,0] = 10*char_u
    velocity[0,1] = 4*char_u

    config = HardSphereSetup(radius, char_u, position, velocity)
    gas = HardSphereSimulation(config, opengl = True, t_scale = 0.5)

    tracer = Tracer(gas, 4)
    histogram = VelocityHistogram(gas, [1.1,0], [1,1])
    view = View(gas, [ ColoredBox([0,0], [1,1], (0.2,0.2,0.2)) ], [ histogram ])

    controller = SimulationController(gas, [ histogram ])

    glutDisplayFunc(make_display_handler(controller, view))
    glutReshapeFunc(make_reshape_handler(view))
    glutTimerFunc(20, make_timer(), 20)
    glutKeyboardFunc(make_keyboard_handler(controller))
    glutCloseFunc(make_close_handler(controller))

    glutMainLoop()
