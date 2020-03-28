from OpenGL.GL   import *
from OpenGL.GLUT import *

from pyrr import matrix44

import numpy as np

from .shader import Shader, GeometryShader
from .camera import Projection, Rotation, MouseDragMonitor, MouseScrollMonitor

particle_shader = (
    """
        #version 430

        in vec3 particle_pos;

        uniform vec4 camera_pos;

        in GS_OUT
        {
            vec3 particle_pos;
            vec2 tex_coord;
        } fs_in;

        void main(){
            if (length(fs_in.tex_coord - vec2(0.5)) > 0.5) {
                discard;
            }

            vec3 n = vec3(fs_in.tex_coord - vec2(0.5), 0.);
            n.z = -sqrt(1.0 - length(n));
            n = normalize(n);

            vec3 dir = normalize(camera_pos.xyz - fs_in.particle_pos);
            vec3 color = vec3(1.) * max(0., dot(dir, n));

            gl_FragColor = vec4(color.xyz, 1.0);
        }
    """,
    """
        #version 430

        layout (points) in;
        layout (triangle_strip, max_vertices=4) out;

        out GS_OUT
        {
            vec3 particle_pos;
            vec2 tex_coord;
        } gs_out;

        uniform mat4 projection;
        uniform mat4 rotation;

        uniform vec4 camera_pos;

        uniform float radius;

        vec4 project(vec3 v) {
            return projection * rotation * vec4(v, 1.);
        }

        void emitSquareAt(vec3 position) {
            mat4 m = projection * rotation;
            vec3 camera_right = normalize(vec3(m[0][0], m[1][0], m[2][0]));
            vec3 camera_up = normalize(vec3(m[0][1], m[1][1], m[2][1]));

            gs_out.particle_pos = gl_in[0].gl_Position.xyz;

            gl_Position = project(position + camera_right * -radius + camera_up * -radius);
            gs_out.tex_coord = vec2(0.,0.);
            EmitVertex();
            gl_Position = project(position + camera_right *  radius + camera_up * -radius);
            gs_out.tex_coord = vec2(1.,0.);
            EmitVertex();
            gl_Position = project(position + camera_right * -radius + camera_up *  radius);
            gs_out.tex_coord = vec2(0.,1.);
            EmitVertex();
            gl_Position = project(position + camera_right *  radius + camera_up *  radius);
            gs_out.tex_coord = vec2(1.,1.);
            EmitVertex();
        }

        void main() {
            emitSquareAt(gl_in[0].gl_Position.xyz);
            EndPrimitive();
        }
    """,
    """
        #version 430

        layout (location=0) in vec4 particle_pos;

        void main() {
            gl_Position = vec4(particle_pos.xyz, 1.0);
        }
    """,
    ['projection', 'rotation', 'camera_pos', 'radius']
)

decoration_shader = (
    """
        #version 430

        void main(){
            gl_FragColor = vec4(1.);
        }
    """,
    """
        #version 430

        layout (location=0) in vec3 vertex;

        uniform mat4 projection;
        uniform mat4 rotation;

        void main() {
            gl_Position = projection * rotation * vec4(vertex.xyz, 1.);
        }
    """,
    ['projection', 'rotation']
)

texture_shader = (
    """
        #version 430

        in vec2 tex_coord;

        uniform sampler2D picture[2];

        uniform float mixing;

        void main() {
            vec3 color = mix(texture(picture[0], tex_coord), texture(picture[1], tex_coord), mixing).xyz;
            if (color == vec3(0.,0.,0.)) {
                gl_FragColor = vec4(color, 0.5);
            } else {
                gl_FragColor = vec4(color, 1.0);
            }
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
    def __init__(self, gas, instruments):
        self.gas = gas
        self.instruments = instruments

        self.texture_shader    = Shader(*texture_shader)
        self.particle_shader   = GeometryShader(*particle_shader)
        self.decoration_shader = Shader(*decoration_shader)

        self.camera_projection = Projection(distance = 5)
        self.camera_rotation = Rotation([-1/2, -1/2, -1/2], np.pi/2, 0)
        self.camera_pos = np.matmul([0,-self.camera_projection.distance,0,1], self.camera_rotation.get_inverse())

        self.mouse_monitors = [
            MouseDragMonitor(GLUT_LEFT_BUTTON,  lambda dx, dy: self.camera_rotation.update(0.005*dy, 0.005*dx)),
            MouseDragMonitor(GLUT_RIGHT_BUTTON, lambda dx, dy: self.camera_projection.shift(0.005*dx, 0.005*dy)),
            MouseScrollMonitor(lambda zoom: self.camera_projection.update_distance(0.1*zoom))
        ]

        self.show_windows = False
        self.show_decorations = True

    def reshape(self, width, height):
        glViewport(0,0,width,height)

        world_height = 1.4
        world_width = world_height / height * width

        projection = Projection(10)

        projection  = matrix44.create_orthogonal_projection(-world_width/2, world_width/2, -world_height/2, world_height/2, -1, 1)
        translation = matrix44.create_from_translation([-1.05, -1.0/2, 0])

        self.camera_projection.update_ratio(width, height)

        self.pixels_per_unit = height / world_height
        self.projection = np.matmul(translation, projection)

    def display(self):
        glClearColor(0.,0.,0.,1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LEQUAL)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.particle_shader.use()
        glUniformMatrix4fv(self.particle_shader.uniform['projection'], 1, False, np.ascontiguousarray(self.camera_projection.get()))
        glUniformMatrix4fv(self.particle_shader.uniform['rotation'],   1, False, np.ascontiguousarray(self.camera_rotation.get()))
        glUniform4fv(self.particle_shader.uniform['camera_pos'], 1, self.camera_pos)
        glUniform1f(self.particle_shader.uniform['radius'], self.gas.radius)

        self.gas.gl_draw_particles()
        glBindVertexArray(0)

        if self.show_decorations:
            self.decoration_shader.use()
            glUniformMatrix4fv(self.decoration_shader.uniform['projection'], 1, False, np.ascontiguousarray(self.camera_projection.get()))
            glUniformMatrix4fv(self.decoration_shader.uniform['rotation'],   1, False, np.ascontiguousarray(self.camera_rotation.get()))
            glLineWidth(2)
            for instrument in self.instruments:
                instrument.display_decoration(self.decoration_shader.uniform)

        if self.show_windows:
            self.texture_shader.use()
            glUniformMatrix4fv(self.texture_shader.uniform['projection'], 1, False, np.asfortranarray(self.projection))
            for instrument in self.instruments:
                instrument.display_window(self.texture_shader.uniform)

        glutSwapBuffers()
