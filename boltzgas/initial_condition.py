import numpy as np

def grid_of_random_velocity_particles(width, radius, u_scale):
    np_position = np.ndarray((width**2, 2))
    np_velocity = np.ndarray((width**2, 2))

    grid = np.meshgrid(np.linspace(2*radius, 1-2*radius, width),
                       np.linspace(2*radius, 1-2*radius, width))
    np_position[:,0] = grid[0].flatten()
    np_position[:,1] = grid[1].flatten()

    np_velocity[:,0] = u_scale*(-0.5 + np.random.random_sample((width**2,)))
    np_velocity[:,1] = u_scale*(-0.5 + np.random.random_sample((width**2,)))

    return (np_position, np_velocity)
