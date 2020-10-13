import numpy as np
import matplotlib.pyplot as plt

from simulation import *

# simulation parameters
N = 100
L = 1.0
alpha = 0.5
dt = 0.05
gamma = 0.05
T = 0.05
n = 100000

# histogram parameters
average_start = int(0.5*n)
bins = 1000

# run simulation
sim = Simulation()
sim.init_integrator('baoab', dt)
sim.init_thermostat(gamma, T)
sim.init_chain(N, L, alpha, 'gaussian', 0.1)
sim.init_output(x=True, v=True, e_pot=True, e_kin=True)
sim.run(n)

# copy simulation history
t = np.linspace(0.0, n*dt, n+1)
x = sim.x_hist
v = sim.v_hist
e_pot = sim.e_pot_hist
e_kin = sim.e_kin_hist
e_tot = e_pot + e_kin

# compute speed histogram
# large number of steps necessary for good convergence (order 1e6)
s = np.abs(v[average_start:]).flatten()
s_min = np.amin(s)
s_max = np.amax(s)

# theoretical speed histogram (Maxwell-Boltzmann distribution)
s_theory = np.linspace(s_min, s_max, bins)
p_theory = 2.0*np.sqrt(0.5/np.pi/T) * np.exp(-0.5*s_theory**2/T)

# plot results
fig1, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0})

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

# plot speed histogram
fig2, ax1 = plt.subplots()
ax1.set(xlabel='Speed', ylabel='Probability')
ax1.hist(s, bins, density=True, label='Speed Distribution')
ax1.plot(s_theory, p_theory, label='Maxwell-Boltzmann Distribution')
ax1.legend()
plt.show()
