import numpy as np
import scipy.stats as stats
import scipy.constants as const
from scipy.optimize import minimize

import matplotlib
import matplotlib.pyplot as plt

from boltzgas import HardSphereSetup, HardSphereSimulation
from boltzgas.initial_condition import grid_of_random_velocity_particles

grid_width = 30
radius = 0.002
char_u = 1120

position, velocity = grid_of_random_velocity_particles(grid_width, radius, char_u)
velocity[:,:] = 0
velocity[0,0] = 10.75*char_u
velocity[0,1] = -.25*char_u
config = HardSphereSetup(radius, char_u, position, velocity)
gas = HardSphereSimulation(config)

m_nitrogen = 0.028 / const.N_A

def plot(step, velocities):
    velocities = np.array([np.linalg.norm(v) for v in velocities])
    maxwellian = stats.maxwell.fit(velocities)

    print("T = %.0f K; u_mean = %.0f [m/s]; energy = %.05f" % ((maxwellian[1]**2 / const.k * m_nitrogen, stats.maxwell.mean(*maxwellian), np.sum([x**2 for x in velocities]))))

    plt.figure()

    plt.ylim(0, 0.003)
    plt.ylabel('Probability')

    plt.xlim(0, 1.2*char_u)
    plt.xlabel('Velocity magnitude [m/s]')

    plt.hist(velocities, bins=50, density=True, alpha=0.5, label='Simulated velocities')

    xs = np.linspace(0, 1.2*char_u, 100)
    plt.plot(xs, stats.maxwell.pdf(xs, *maxwellian), label='Maxwell-Boltzmann distribution')

    plt.legend(loc='upper right')

    plt.savefig("result/%04d.png" % step)
    plt.close()

def simulate(n_steps, section):
    for i in range(0, int(n_steps / section)):
        print("Plot step %d." % (i * section))

        velocities = gas.get_velocities()

        for j in range(0,section):
            gas.evolve()

        plot(i, velocities)

simulate(100000, 1000)
