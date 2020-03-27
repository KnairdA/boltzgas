from OpenGL.GL import *
from OpenGL.GLUT import *

from boltzgas.visual import View

class SimulationController:
    def __init__(self, gas, instruments, updates_per_frame):
        self.running = False
        self.gas = gas
        self.instruments = instruments
        self.updates_per_frame = updates_per_frame

    def isRunning(self):
        return self.running

    def run(self):
        self.running = True

    def pause(self):
        self.running = False

    def evolve(self):
        if self.running:
            for i in range(0,self.updates_per_frame):
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

def make_keyboard_handler(controller, view):
    def on_keyboard(key, x, y):
        if key == b' ':
            if controller.isRunning():
                controller.pause()
            else:
                controller.run()
        if key == b'h':
            view.show_histogram = not view.show_histogram

    return on_keyboard

def make_close_handler(controller):
    def on_close():
        controller.shutdown()

    return on_close

def simulate(config, gas, instruments, decorations, windows, updates_per_frame = 5):
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowPosition(0, 0)
    glutCreateWindow("BoltzGas")

    gas.setup()
    for instrument in instruments:
        instrument.setup()

    view = View(gas, decorations, windows)
    controller = SimulationController(gas, instruments, updates_per_frame)

    glutDisplayFunc(make_display_handler(controller, view))
    glutReshapeFunc(make_reshape_handler(view))
    glutTimerFunc(20, make_timer(), 20)
    glutKeyboardFunc(make_keyboard_handler(controller, view))
    glutCloseFunc(make_close_handler(controller))
    glutMouseFunc(lambda *args: list(map(lambda m: m.on_mouse(*args), view.mouse_monitors)))
    glutMotionFunc(lambda *args: list(map(lambda m: m.on_mouse_move(*args), view.mouse_monitors)))

    glutMainLoop()
