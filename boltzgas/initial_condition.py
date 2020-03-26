import numpy as np

def grid_of_random_velocity_particles(width, radius, u_scale):
    np_position = np.ndarray((width**3, 4))
    np_velocity = np.ndarray((width**3, 4))

    grid = np.meshgrid(np.linspace(2*radius, 1-2*radius, width),
                       np.linspace(2*radius, 1-2*radius, width),
                       np.linspace(2*radius, 1-2*radius, width))
    np_position[:,0] = grid[0].flatten()
    np_position[:,1] = grid[1].flatten()
    np_position[:,2] = grid[2].flatten()
    np_position[:,3] = 0

    np_velocity[:,0] = u_scale*(-0.5 + np.random.random_sample((width**3,)))
    np_velocity[:,1] = u_scale*(-0.5 + np.random.random_sample((width**3,)))
    np_velocity[:,2] = u_scale*(-0.5 + np.random.random_sample((width**3,)))
    np_velocity[:,3] = 0

    return (np_position, np_velocity)
