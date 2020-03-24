from OpenGL.GL import *

class Tracer:
    def __init__(self, gas, iParticle):
        self.gas = gas
        self.iParticle = iParticle
        self.trace = [ ]

    def update(self):
        position = self.gas.get_positions()[self.iParticle]
        self.trace.append((position[0], position[1]))

    def display(self, uniform):
        glUniform3f(uniform['face_color'], 1., 0., 0.)
        glLineWidth(2)
        glBegin(GL_LINE_STRIP)
        for v in self.trace:
            glVertex(*v, 0.)
        glEnd()

