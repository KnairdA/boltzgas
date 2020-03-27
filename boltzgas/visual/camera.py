from OpenGL.GL   import *
from OpenGL.GLUT import *

from pyrr import matrix44, quaternion

import numpy as np

class Projection:
    def __init__(self, distance):
        self.distance = distance
        self.ratio    = 4./3.
        self.x = 0
        self.y = 0
        self.update()

    def update(self):
        projection = matrix44.create_perspective_projection(20.0, self.ratio, 0.1, 5000.0)
        look = matrix44.create_look_at(
            eye    = [self.x, -self.distance, self.y],
            target = [self.x, 0, self.y],
            up     = [0, 0,-1])

        self.matrix = np.matmul(look, projection)

    def update_ratio(self, width, height, update_viewport = True):
        if update_viewport:
            glViewport(0,0,width,height)

        self.ratio = width/height
        self.update()

    def update_distance(self, change):
        self.distance += change
        self.update()

    def shift(self, x, y):
        self.x -= x
        self.y -= y
        self.update()

    def get(self):
        return self.matrix

class Rotation:
    def __init__(self, shift, x = np.pi, z = np.pi):
        self.matrix = matrix44.create_from_translation(shift),
        self.rotation_x = quaternion.Quaternion()
        self.update(x,z)

    def shift(self, x, z):
        self.matrix = np.matmul(
            self.matrix,
            matrix44.create_from_translation([x,0,z])
        )
        self.inverse_matrix = np.linalg.inv(self.matrix)

    def update(self, x, z):
        rotation_x = quaternion.Quaternion(quaternion.create_from_eulers([x,0,0]))
        rotation_z = self.rotation_x.conjugate.cross(
                quaternion.Quaternion(quaternion.create_from_eulers([0,0,z])))
        self.rotation_x = self.rotation_x.cross(rotation_x)

        self.matrix = np.matmul(
            self.matrix,
            matrix44.create_from_quaternion(rotation_z.cross(self.rotation_x))
        )
        self.inverse_matrix = np.linalg.inv(self.matrix)

    def get(self):
        return self.matrix

    def get_inverse(self):
        return self.inverse_matrix

class MouseDragMonitor:
    def __init__(self, button, callback):
        self.button   = button
        self.active   = False
        self.callback = callback

    def on_mouse(self, button, state, x, y):
        if button == self.button:
            self.active = (state == GLUT_DOWN)
            self.last_x = x
            self.last_y = y

    def on_mouse_move(self, x, y):
        if self.active:
            delta_x = self.last_x - x
            delta_y = y - self.last_y
            self.last_x = x
            self.last_y = y
            self.callback(delta_x, delta_y)

class MouseScrollMonitor:
    def __init__(self, callback):
        self.callback = callback

    def on_mouse(self, button, state, x, y):
        if button == 3:
            self.callback(-1.0)
        elif button == 4:
            self.callback(1.0)

    def on_mouse_move(self, x, y):
        pass
