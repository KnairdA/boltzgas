from OpenGL.GL import *

class ColoredBox:
    def __init__(self, origin, extend, color):
        self.origin = origin
        self.extend = extend
        self.color = color

    def display(self, uniform):
        glUniform3f(uniform['face_color'], *self.color)
        glBegin(GL_TRIANGLE_STRIP)
        glVertex(self.origin[0],                  self.origin[1]                 , 0.)
        glVertex(self.origin[0] + self.extend[0], self.origin[1]                 , 0.)
        glVertex(self.origin[0]                 , self.origin[1] + self.extend[1], 0.)
        glVertex(self.origin[0] + self.extend[1], self.origin[1] + self.extend[1], 0.)
        glEnd()
