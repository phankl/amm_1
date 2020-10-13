import numpy as np
import matplotlib.pyplot as plt

from simulation import *

# simulation parameters
N = 100
L = 1.0
alpha = 0.5
dt = 0.1
n = 10000

# run simulation
sim = Simulation()
sim.init_integrator('verlet', dt)
sim.init_chain(N, L, alpha, 'gaussian', 0.2)
sim.init_output(x=True, e_pot=True, e_kin=True)
sim.run(n)

# copy simulation history
t = np.linspace(0.0, n*dt, n+1)
x = sim.x_hist
e_pot = sim.e_pot_hist
e_kin = sim.e_kin_hist
e_tot = e_pot + e_kin

# plot results
fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0})

# plot positions as a function of time
ax1.set(xlabel='Time', ylabel='Position')
for i in range(N):
  ax1.plot(t, x[:, i], 'k', linewidth=0.5)

# plot energies as a function of time
ax2.set(xlabel='Time', ylabel='Energy')
ax2.plot(t, e_pot, 'b', label='PE')
ax2.plot(t, e_kin, 'r', label='KE')
ax2.plot(t, e_tot, 'g', label='Total')
ax2.legend()
plt.show()
