from OpenGL.GL import *

import numpy as np

import scipy.stats as stats
import scipy.constants as const
from scipy.optimize import minimize

import matplotlib
import matplotlib.pyplot as plt

from concurrent.futures import Future, ProcessPoolExecutor

def get_histogram(velocities, char_u):
    maxwellian = stats.maxwell.fit(velocities)

    plt.style.use('dark_background')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1, 0.06, 0.88, 0.92])

    plt.ylim(0, 0.004)
    plt.ylabel('Probability')

    plt.xlim(0, 1.2*char_u)
    plt.xlabel('Velocity magnitude [m/s]')

    plt.hist(velocities, bins=50, density=True, alpha=0.5, label='Simulated velocities')

    xs = np.linspace(0, 1.2*char_u, 100)
    plt.plot(xs, stats.maxwell.pdf(xs, *maxwellian), label='Maxwell-Boltzmann distribution')

    plt.legend(loc='upper right')

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    texture = np.frombuffer(buf, dtype=np.uint8).reshape(width, height, 3)

    plt.close(fig)

    return (texture, height, width)


class VelocityHistogram:
    def __init__(self, gas, origin, extend):
        self.gas = gas
        self.origin = origin
        self.extend = extend
        self.steps = -1

        self.pool = ProcessPoolExecutor(max_workers=1)
        self.plotter = None

        self.tick = False
        self.mixing = 0.0

        self.vertices = np.array([
            self.origin[0]                 , self.origin[1]                 , 0., 1.,
            self.origin[0] + self.extend[0], self.origin[1]                 , 1., 1.,
            self.origin[0]                 , self.origin[1] + self.extend[1], 0., 0.,
            self.origin[0] + self.extend[0], self.origin[1] + self.extend[1], 1., 0.
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.tostring(), GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*np.dtype(np.float32).itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*np.dtype(np.float32).itemsize, ctypes.c_void_p(2*np.dtype(np.float32).itemsize))

        self.texture_id = glGenTextures(2)

        glBindTexture(GL_TEXTURE_2D, self.texture_id[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindTexture(GL_TEXTURE_2D, self.texture_id[1])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    def shutdown(self):
        self.pool.shutdown()

    def update(self):
        self.steps = self.steps + 1

        if self.steps % 50 == 0 and self.plotter == None:
            self.plotter = self.pool.submit(get_histogram, self.gas.get_velocity_norms(), self.gas.char_u)

        else:
            if self.plotter != None and self.plotter.done():
                texture, width, height = self.plotter.result()
                if self.tick:
                    glBindTexture(GL_TEXTURE_2D, self.texture_id[0])
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture)
                    self.tick = False
                    self.mixing = 1.0

                else:
                    glBindTexture(GL_TEXTURE_2D, self.texture_id[1])
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture)
                    self.tick = True
                    self.mixing = 0.0

                self.plotter = None


    def display(self, uniform):
        if self.tick:
            self.mixing = min(self.mixing+0.05, 1.0);
        else:
            self.mixing = max(self.mixing-0.05, 0.0);

        glBindTextures(self.texture_id[0], 2, self.texture_id)
        glUniform1iv(uniform['picture'], len(self.texture_id), self.texture_id)
        glUniform1f(uniform['mixing'], self.mixing)

        glBindVertexArray(self.vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
