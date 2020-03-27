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

class WireBox:
    def __init__(self, x0, x1, y0, y1, z0, z1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def display(self, uniform):
        glBegin(GL_LINE_STRIP)
        glVertex(self.x0, self.y0, self.z0)
        glVertex(self.x0, self.y1, self.z0)
        glVertex(self.x0, self.y1, self.z1)
        glVertex(self.x0, self.y0, self.z1)
        glVertex(self.x0, self.y0, self.z0)
        glEnd()
        glBegin(GL_LINE_STRIP)
        glVertex(self.x1, self.y0, self.z0)
        glVertex(self.x1, self.y1, self.z0)
        glVertex(self.x1, self.y1, self.z1)
        glVertex(self.x1, self.y0, self.z1)
        glVertex(self.x1, self.y0, self.z0)
        glEnd()
        glBegin(GL_LINE_STRIP)
        glVertex(self.x0, self.y0, self.z1)
        glVertex(self.x1, self.y0, self.z1)
        glVertex(self.x1, self.y1, self.z1)
        glVertex(self.x0, self.y1, self.z1)
        glVertex(self.x0, self.y0, self.z1)
        glEnd()
        glBegin(GL_LINE_STRIP)
        glVertex(self.x0, self.y0, self.z0)
        glVertex(self.x1, self.y0, self.z0)
        glVertex(self.x1, self.y1, self.z0)
        glVertex(self.x0, self.y1, self.z0)
        glVertex(self.x0, self.y0, self.z0)
        glEnd()
        glBegin(GL_LINE_STRIP)
        glVertex(self.x0, self.y0, self.z0)
        glVertex(self.x1, self.y0, self.z0)
        glVertex(self.x1, self.y0, self.z1)
        glVertex(self.x0, self.y0, self.z1)
        glVertex(self.x0, self.y0, self.z0)
        glEnd()
        glBegin(GL_LINE_STRIP)
        glVertex(self.x0,self.y1,self.z0)
        glVertex(self.x1,self.y1,self.z0)
        glVertex(self.x1,self.y1,self.z1)
        glVertex(self.x0,self.y1,self.z1)
        glVertex(self.x0,self.y1,self.z0)
        glEnd()



