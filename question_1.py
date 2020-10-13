import numpy as np
import matplotlib.pyplot as plt

from simulation import *

# simulation parameters
dt = 0.1
n = 1000

# run simulation
sim = Simulation()
sim.init_integrator('verlet', dt)
sim.init_SHO(1.0, 0.0)
sim.init_output(x=True, v=True)
sim.run(n)

# copy simulation history
t = np.linspace(0.0, n*dt, n+1)
x = sim.x_hist.flatten()
v = sim.v_hist.flatten()

# plot results
fig, (ax1, ax2) = plt.subplots(2)
ax1.set(xlabel='Time', ylabel='Position')
ax1.plot(t, x)
ax2.set(xlabel='Position', ylabel='Velocity')
ax2.plot(v, x)
plt.show()
