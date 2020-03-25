from OpenGL.GL   import *
from OpenGL.GLUT import *

from pyrr import matrix44

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

        layout (location=0) in vec2 particles;

        out vec3 color;

        uniform mat4 projection;
        uniform vec3 face_color;
        uniform vec3 trace_color;
        uniform uint trace_id;

        void main() {
            gl_Position = projection * vec4(particles, 0., 1.);

            if (gl_VertexID == trace_id) {
                color = trace_color;
            } else {
                color = face_color;
            }
        }
    """,
    ['projection', 'face_color', 'trace_color', 'trace_id']
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

class View:
    def __init__(self, gas, decorations, windows):
        self.gas = gas
        self.decorations = decorations
        self.windows = windows

        self.texture_shader    = Shader(*texture_shader)
        self.particle_shader   = Shader(*particle_shader)
        self.decoration_shader = Shader(*decoration_shader)

    def reshape(self, width, height):
        glViewport(0,0,width,height)

        world_height = 1.4
        world_width = world_height / height * width

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
        glUniformMatrix4fv(self.particle_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))
        glUniform3f(self.particle_shader.uniform['face_color'],  1., 1., 1.)
        glUniform3f(self.particle_shader.uniform['trace_color'], 1., 0., 0.)
        glUniform1ui(self.particle_shader.uniform['trace_id'], -1)
        glEnable(GL_POINT_SPRITE)
        glPointSize(2*self.gas.radius*self.pixels_per_unit)
        self.gas.gl_draw_particles()

        glutSwapBuffers()
