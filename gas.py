from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders

from pyrr import matrix44

import numpy as np

from particles import GasFlow, HardSphereSetup, grid_of_random_velocity_particles

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

def get_projection(width, height):
    world_height = 1.0
    world_width = world_height / height * width
    pixels_per_unit = height / world_height

    projection  = matrix44.create_orthogonal_projection(-world_width/2, world_width/2, -world_height/2, world_height/2, -1, 1)
    translation = matrix44.create_from_translation([-1.0/2, -1.0/2, 0])

    return (np.matmul(translation, projection), pixels_per_unit)

def on_reshape(width, height):
    global projection, pixels_per_unit
    glViewport(0,0,width,height)
    projection, pixels_per_unit = get_projection(width, height)

glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowPosition(0, 0)
window = glutCreateWindow("gas")
glutTimerFunc(10, on_timer, 10)

fragment_shader = shaders.compileShader("""
#version 430

in vec3 color;

void main(){
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) {
        discard;
    }

    gl_FragColor = vec4(color.xyz, 0.0);
}""", GL_FRAGMENT_SHADER)

particle_shader = shaders.compileShader("""
#version 430

layout (location=0) in vec2 particles;

out vec3 color;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(particles, 0., 1.);
    color = vec3(0.0);
}""", GL_VERTEX_SHADER)

particle_program = shaders.compileProgram(particle_shader, fragment_shader)

background_fragment_shader = shaders.compileShader("""
#version 430

in vec3 color;

void main(){
    gl_FragColor = vec4(color.xyz, 0.0);
}""", GL_FRAGMENT_SHADER)

background_vertex_shader = shaders.compileShader("""
#version 430

in vec3 vertex;

out vec3 color;

uniform mat4 projection;
uniform vec3 face_color;

void main() {
    gl_Position = projection * vec4(vertex, 1.);
    color = face_color;
}""", GL_VERTEX_SHADER)
background_program = shaders.compileProgram(background_vertex_shader, background_fragment_shader)

particle_projection_id   = shaders.glGetUniformLocation(particle_program, 'projection')
background_projection_id = shaders.glGetUniformLocation(background_program, 'projection')
background_color_id = shaders.glGetUniformLocation(background_program, 'face_color')

class View:
    def __init__(self, gas):
        self.gas = gas
        self.energy = 0
        self.tracer = [ ]

    def update_trace(self):
        positions = self.gas.get_positions()
        self.tracer.append((positions[5][0],positions[5][1]))
        if len(self.tracer) > 100:
            self.tracer.pop(0)

    def draw_trace(self):
        glUniform3f(background_color_id, 1., 0., 0.)
        glLineWidth(2)
        glBegin(GL_LINE_STRIP)
        for v in self.tracer:
            glVertex(*v, 0.)
        glEnd()

    def on_display(self):
        for i in range(0,10):
            self.gas.evolve()

        glClearColor(0.4,0.4,0.4,1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        shaders.glUseProgram(background_program)
        glUniformMatrix4fv(background_projection_id, 1, False, np.asfortranarray(projection))
        glUniform3f(background_color_id, 1., 1., 1.)
        glBegin(GL_TRIANGLE_STRIP)
        glVertex(0.,0.,0.)
        glVertex(1.,0.,0.)
        glVertex(0.,1.,0.)
        glVertex(1.,1.,0.)
        glEnd()

        if trace:
            self.draw_trace()

        shaders.glUseProgram(particle_program)
        glUniformMatrix4fv(particle_projection_id, 1, False, np.asfortranarray(projection))
        glEnable(GL_POINT_SPRITE)
        glPointSize(2*radius*pixels_per_unit)
        gas.gl_draw_particles()

        glutSwapBuffers()

        if trace:
            self.update_trace()

        velocities = gas.get_velocities()
        energy = np.sum(np.array([np.linalg.norm(v)**2 for v in velocities]))
        if abs(energy - self.energy) > 1e-4:
            print("energy = %.05f" % energy)
            self.energy = energy

grid_width = 10
radius = 0.005
char_u = 1
trace = True

config = HardSphereSetup(radius, char_u, *grid_of_random_velocity_particles(grid_width, radius, char_u))
gas = GasFlow(config, opengl = True)

view = View(gas)

glutDisplayFunc(view.on_display)
glutReshapeFunc(on_reshape)

glutMainLoop()
