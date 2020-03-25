import numpy as np

import pyopencl as cl
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties

import OpenGL.GL as gl
from OpenGL.arrays import vbo

from string import Template

class HardSphereSetup:
    def __init__(self, radius, char_u, position, velocity):
        self.radius = radius
        self.char_u = char_u
        self.position = position
        self.velocity = velocity
        self.n_particles = len(position[:,0])

def build_kernel(delta_t, n_particles, radius):
    with open('boltzgas/kernel.cl', 'r') as kernel_src:
        return Template(kernel_src.read()).substitute(
            delta_t     = delta_t,
            n_particles = n_particles,
            radius      = radius)

class HardSphereSimulation:
    def setup_cl(self):
        self.platform = cl.get_platforms()[0]
        if self.opengl:
            self.context = cl.Context(
                properties=[(cl.context_properties.PLATFORM, self.platform)] + get_gl_sharing_context_properties())
        else:
            self.context = cl.Context(
                properties=[(cl.context_properties.PLATFORM, self.platform)])
        self.queue = cl.CommandQueue(self.context)

        self.program = cl.Program(self.context, self.kernel_src).build(
            '-cl-single-precision-constant -cl-opt-disable')

        self.cl_particle_velocity_a = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.np_particle_velocity)
        self.cl_particle_velocity_b = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.np_particle_velocity)

        if self.opengl:
            self.gl_particle_position_a = vbo.VBO(data=self.np_particle_position, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
            self.gl_particle_position_a.bind()
            self.gl_particle_position_b = vbo.VBO(data=self.np_particle_position, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
            self.gl_particle_position_b.bind()

            self.cl_particle_position_a = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_particle_position_a))
            self.cl_particle_position_b = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_particle_position_b))
        else:
            self.cl_particle_position_a = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.np_particle_position)
            self.cl_particle_position_b = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.np_particle_position)

        self.cl_last_collide = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.np_last_collide)
        self.cl_particle_velocity_norms = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.np_particle_velocity_norms)

    def __init__(self, setup, opengl = False, t_scale = 1.0):
        self.np_particle_position = setup.position.astype(np.float32)
        self.np_particle_velocity = setup.velocity.astype(np.float32)

        self.n_particles = setup.n_particles
        self.radius      = setup.radius
        self.char_u      = setup.char_u

        self.opengl = opengl
        self.t_scale = t_scale

        self.np_last_collide = np.ndarray((self.n_particles, 1), dtype=np.uint32)
        self.np_last_collide[:,0] = self.n_particles

        self.np_particle_velocity_norms = np.ndarray((self.n_particles, 1), dtype=np.float32)

        self.kernel_src = build_kernel(self.t_scale*self.radius/self.char_u, self.n_particles, self.radius)

        self.setup_cl()

        self.tick = True

    def evolve(self):
        if self.opengl:
            cl.enqueue_acquire_gl_objects(self.queue, [self.cl_particle_position_a, self.cl_particle_position_b])

        if self.tick:
            self.tick = False
            kernelargs = (self.cl_particle_position_a, self.cl_particle_velocity_a, self.cl_particle_position_b, self.cl_particle_velocity_b, self.cl_last_collide)
        else:
            self.tick = True
            kernelargs = (self.cl_particle_position_b, self.cl_particle_velocity_b, self.cl_particle_position_a, self.cl_particle_velocity_a, self.cl_last_collide)

        self.program.evolve(self.queue, (self.n_particles,), None, *(kernelargs)).wait()

    def gl_draw_particles(self):
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

        if self.tick:
            self.gl_particle_position_b.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, self.gl_particle_position_b)
        else:
            self.gl_particle_position_a.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, self.gl_particle_position_a)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.n_particles)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def get_velocities(self):
        if self.tick:
            cl.enqueue_copy(self.queue, self.np_particle_velocity, self.cl_particle_velocity_b).wait()
            return self.np_particle_velocity
        else:
            cl.enqueue_copy(self.queue, self.np_particle_velocity, self.cl_particle_velocity_a).wait()
            return self.np_particle_velocity

    def get_velocity_norms(self):
        if self.tick:
            self.program.get_velocity_norms(self.queue, (self.n_particles,), None, self.cl_particle_velocity_b, self.cl_particle_velocity_norms)
            cl.enqueue_copy(self.queue, self.np_particle_velocity_norms, self.cl_particle_velocity_norms).wait()
            return self.np_particle_velocity_norms
        else:
            self.program.get_velocity_norms(self.queue, (self.n_particles,), None, self.cl_particle_velocity_a, self.cl_particle_velocity_norms)
            cl.enqueue_copy(self.queue, self.np_particle_velocity_norms, self.cl_particle_velocity_norms).wait()
            return self.np_particle_velocity_norms

    def get_positions(self):
        if self.tick:
            cl.enqueue_copy(self.queue, self.np_particle_position, self.cl_particle_position_b).wait()
            return self.np_particle_position
        else:
            cl.enqueue_copy(self.queue, self.np_particle_position, self.cl_particle_position_a).wait()
            return self.np_particle_position
