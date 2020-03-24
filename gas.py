from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders

from pyrr import matrix44

import numpy as np

import scipy.stats as stats
import scipy.constants as const
from scipy.optimize import minimize

import matplotlib
import matplotlib.pyplot as plt

from particles import GasFlow, HardSphereSetup, grid_of_random_velocity_particles

from concurrent.futures import Future, ProcessPoolExecutor

glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowPosition(0, 0)
window = glutCreateWindow("gas")

class Shader:
    def __init__(self, vertex_src, fragment_src, uniform):
        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_src, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER))
        self.uniform = { }
        for name in uniform:
            self.uniform[name] = shaders.glGetUniformLocation(self.program, name)

    def use(self):
        shaders.glUseProgram(self.program)

particle_shader = Shader(
    fragment_src = """
        #version 430

        in vec3 color;

        void main(){
            if (length(gl_PointCoord - vec2(0.5)) > 0.5) {
                discard;
            }

            gl_FragColor = vec4(color.xyz, 0.0);
        }""",
    vertex_src = """
        #version 430

        layout (location=0) in vec2 particles;

        out vec3 color;

        uniform mat4 projection;

        void main() {
            gl_Position = projection * vec4(particles, 0., 1.);
            color = vec3(0.0);
        }""",
    uniform = ['projection']
)

decoration_shader = Shader(
    fragment_src = """
        #version 430

        in vec3 color;

        void main(){
            gl_FragColor = vec4(color.xyz, 0.0);
        }""",
    vertex_src = """
        #version 430

        in vec3 vertex;

        out vec3 color;

        uniform mat4 projection;
        uniform vec3 face_color;

        void main() {
            gl_Position = projection * vec4(vertex, 1.);
            color = face_color;
        }""",
    uniform = ['projection', 'face_color']
)

texture_shader = Shader(
    fragment_src = """
        #version 430

        in vec2 tex_coord;

        uniform sampler2D picture[2];

        uniform float mixing;

        void main() {
            gl_FragColor = mix(texture(picture[0], tex_coord), texture(picture[1], tex_coord), mixing);
        }""",
    vertex_src = """
        #version 430

        layout (location=0) in vec2 screen_vertex;
        layout (location=1) in vec2 texture_vertex;

        out vec2 tex_coord;

        uniform mat4 projection;

        void main() {
            gl_Position = projection * vec4(screen_vertex, 0.0, 1.0);
            tex_coord = texture_vertex;
        }""",
    uniform = ['picture','projection','mixing']
)

class View:
    def __init__(self, gas, decorations, windows):
        self.gas = gas
        self.decorations = decorations
        self.windows = windows

    def reshape(self, width, height):
        glViewport(0,0,width,height)

        world_height = 1.4
        world_width = world_height / height * width

        projection  = matrix44.create_orthogonal_projection(-world_width/2, world_width/2, -world_height/2, world_height/2, -1, 1)
        translation = matrix44.create_from_translation([-1.05, -1.0/2, 0])

        self.pixels_per_unit = height / world_height
        self.projection = np.matmul(translation, projection)

    def display(self):
        glClearColor(0.4,0.4,0.4,1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        decoration_shader.use()
        glUniformMatrix4fv(decoration_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))
        for decoration in self.decorations:
            decoration.display(decoration_shader.uniform)

        texture_shader.use()
        glUniformMatrix4fv(texture_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))
        for window in self.windows:
            window.display(texture_shader.uniform)

        particle_shader.use()
        glUniformMatrix4fv(particle_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))
        glEnable(GL_POINT_SPRITE)
        glPointSize(2*radius*self.pixels_per_unit)
        self.gas.gl_draw_particles()

        glutSwapBuffers()

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

def get_histogram(velocities):
    maxwellian = stats.maxwell.fit(velocities)

    fig = plt.figure(figsize=(10,10))

    plt.ylim(0, 0.003)
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

    def setup(self):
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
        print(self.texture_id)

        glBindTexture(GL_TEXTURE_2D, self.texture_id[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindTexture(GL_TEXTURE_2D, self.texture_id[1])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    def update(self):
        self.steps = self.steps + 1

        if self.steps % 100 == 0 and self.plotter == None:
            self.plotter = self.pool.submit(get_histogram, self.gas.get_velocity_norms())

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
            self.mixing = min(self.mixing+0.1, 1.0);
        else:
            self.mixing = max(self.mixing-0.1, 0.0);

        glBindTextures(self.texture_id[0], 2, self.texture_id)
        glUniform1iv(uniform['picture'], len(self.texture_id), self.texture_id)
        glUniform1f(uniform['mixing'], self.mixing)

        glBindVertexArray(self.vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

grid_width = 32
radius = 0.004
char_u = 1120

position, velocity = grid_of_random_velocity_particles(grid_width, radius, char_u)
velocity[:,:] = 0
velocity[0,0] = 10*char_u
velocity[0,1] = 4*char_u

config = HardSphereSetup(radius, char_u, position, velocity)
gas = GasFlow(config, opengl = True, t_scale = 1.0)

tracer = Tracer(gas, 4)
histogram = VelocityHistogram(gas, [1.1,0], [1,1])
histogram.setup()
view = View(gas, [ ColoredBox([0,0], [1,1], (1,1,1)), tracer ], [ histogram ])

running = False

def on_display():
    if running:
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
    global running
    running = not running

def on_close():
    histogram.pool.shutdown(wait=True)

glutDisplayFunc(on_display)
glutReshapeFunc(on_reshape)
glutTimerFunc(10, on_timer, 10)
glutKeyboardFunc(on_keyboard)
glutCloseFunc(on_close)

glutMainLoop()
