from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER
from OpenGL.GL import shaders

class Shader:
    def __init__(self, fragment_src, vertex_src, uniform):
        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_src, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER))
        self.uniform = { }
        for name in uniform:
            self.uniform[name] = shaders.glGetUniformLocation(self.program, name)

    def use(self):
        shaders.glUseProgram(self.program)

class GeometryShader:
    def __init__(self, fragment_src, geometry_src, vertex_src, uniform):
        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_src, GL_VERTEX_SHADER),
            shaders.compileShader(geometry_src, GL_GEOMETRY_SHADER),
            shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER))
        self.uniform = { }
        for name in uniform:
            self.uniform[name] = shaders.glGetUniformLocation(self.program, name)

    def use(self):
        shaders.glUseProgram(self.program)

