from OpenGL.GL   import *
from OpenGL.GLUT import *

from pyrr import matrix44, quaternion

import numpy as np

from .shader import Shader

particle_shader = (
    """
        #version 430

        in vec3 color;

        void main(){
            if (length(gl_PointCoord - vec2(0.5)) > 0.5) {
                discard;
            }

            gl_FragColor = vec4(color.xyz, 0.0);
        }
    """,
    """
        #version 430

        layout (location=0) in vec4 particles;

        out vec3 color;

        uniform mat4 projection;
        uniform mat4 rotation;

        uniform vec3 face_color;
        uniform vec3 trace_color;
        uniform uint trace_id;

        void main() {
            gl_Position = projection * rotation * vec4(particles.xyz, 1.);

            if (particles.x < 0.0 || particles.x > 1.0 ||
                particles.y < 0.0 || particles.y > 1.0 ||
                particles.z < 0.0 || particles.z > 1.0
            ) {
                color = trace_color;
            } else {
                color = face_color;
            }
        }
    """,
    ['projection', 'rotation', 'face_color', 'trace_color', 'trace_id']
)

decoration_shader = (
    """
        #version 430

        in vec3 color;

        void main(){
            gl_FragColor = vec4(color.xyz, 0.0);
        }
    """,
    """
        #version 430

        in vec3 vertex;

        out vec3 color;

        uniform mat4 projection;
        uniform vec3 face_color;

        void main() {
            gl_Position = projection * vec4(vertex, 1.);
            color = face_color;
        }
    """,
    ['projection', 'face_color']
)

texture_shader = (
    """
        #version 430

        in vec2 tex_coord;

        uniform sampler2D picture[2];

        uniform float mixing;

        void main() {
            gl_FragColor = mix(texture(picture[0], tex_coord), texture(picture[1], tex_coord), mixing);
        }
    """,
    """
        #version 430

        layout (location=0) in vec2 screen_vertex;
        layout (location=1) in vec2 texture_vertex;

        out vec2 tex_coord;

        uniform mat4 projection;

        void main() {
            gl_Position = projection * vec4(screen_vertex, 0.0, 1.0);
            tex_coord = texture_vertex;
        }
    """,
    ['picture','projection','mixing']
)


class Projection:
    def __init__(self, distance):
        self.distance = distance
        self.ratio    = 4./3.
        self.update()

    def update(self):
        projection = matrix44.create_perspective_projection(20.0, self.ratio, 0.1, 5000.0)
        look = matrix44.create_look_at(
            eye    = [0, 0, -self.distance],
            target = [0, 0, 0],
            up     = [0,-1, 0])

        self.matrix = np.matmul(look, projection)

    def update_ratio(self, width, height, update_viewport = True):
        if update_viewport:
            glViewport(0,0,width,height)

        self.ratio = width/height
        self.update()

    def update_distance(self, change):
        self.distance += change
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



class View:
    def __init__(self, gas, decorations, windows):
        self.gas = gas
        self.decorations = decorations
        self.windows = windows

        self.texture_shader    = Shader(*texture_shader)
        self.particle_shader   = Shader(*particle_shader)
        self.decoration_shader = Shader(*decoration_shader)

        self.projection3d = Projection(distance = 7)
        self.rotation3d = Rotation([-1/2, -1/2, -1/2], 5*np.pi/4, np.pi/4)

    def reshape(self, width, height):
        glViewport(0,0,width,height)

        world_height = 1.4
        world_width = world_height / height * width

        projection = Projection(10)

        projection  = matrix44.create_orthogonal_projection(-world_width/2, world_width/2, -world_height/2, world_height/2, -1, 1)
        translation = matrix44.create_from_translation([-1.05, -1.0/2, 0])

        self.pixels_per_unit = height / world_height
        self.projection = np.matmul(translation, projection)

    def display(self):
        glClearColor(0.,0.,0.,1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.decoration_shader.use()
        glUniformMatrix4fv(self.decoration_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))
        for decoration in self.decorations:
            decoration.display(self.decoration_shader.uniform)

        self.texture_shader.use()
        glUniformMatrix4fv(self.texture_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))
        for window in self.windows:
            window.display(self.texture_shader.uniform)

        self.particle_shader.use()
        #glUniformMatrix4fv(self.particle_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))

        glUniformMatrix4fv(self.particle_shader.uniform['projection'], 1, False, np.ascontiguousarray(self.projection3d.get()))
        glUniformMatrix4fv(self.particle_shader.uniform['rotation'],   1, False, np.ascontiguousarray(self.rotation3d.get()))

        glUniform3f(self.particle_shader.uniform['face_color'],  1., 1., 1.)
        glUniform3f(self.particle_shader.uniform['trace_color'], 1., 0., 0.)
        glUniform1ui(self.particle_shader.uniform['trace_id'], -1)

        glEnable(GL_POINT_SPRITE)
        glPointSize(2*self.gas.radius*self.pixels_per_unit)
        self.gas.gl_draw_particles()

        glutSwapBuffers()
