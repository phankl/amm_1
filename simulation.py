import numpy as np

class Simulation:
  
  def __init__(self):

    ''' initialise constants, parameters and arrays '''

    # Boltzmann constant
    self.k = 1.38064852e-23
    self.k = 1.0

    # default values for physical parameters
    self.alpha = 0.5
    self.T = 0.05
    self.N = 100
    self.L = 1.0
    
    # default values for numerical parameters
    self.dt = 1.0
    self.gamma = 1.0
    self.MC_std = 0.0

    # time step
    self.n = 0
    self.accepted = 0

    # positions, velocities and forces
    self.x = []
    self.x_prev = []
    self.v = []
    self.f = []

    # potential energy for MC
    self.e_pot = 0.0

    # output booleans
    self.x_bool = False
    self.v_bool = False
    self.l_bool = False
    self.e_pot_bool = False
    self.e_kin_bool = False

    # containers for simulation history
    self.x_hist = []
    self.v_hist = []
    self.l_hist = []
    self.e_pot_hist = []
    self.e_kin_hist = []
    

  def init_integrator(self, integrator, dt):
    
    ''' initialise integrator for simulations '''
    
    '''
    integrator: name of integrator (string)
    possible values:
      'euler', 'verlet', 'velocity_verlet', 'baoab'
    dt: time step for integrator (float)
    '''

    self.integrator = integrator
    self.dt = dt
    
    if integrator != 'euler' and integrator != 'verlet' \
      and integrator != 'velocity_verlet' and integrator != 'baoab':
      print("Error! Input #3 should be either 'euler', 'verlet', \
        'velocity_verlet' or 'baoab'")


  def init_thermostat(self, gamma, T):
    
    ''' initialise thermostat for simulations '''

    '''
    gamma: Langevin thermostat inverse relaxation time (float)
    T: temperature (float)
    '''

    self.gamma = gamma
    self.T = T


  def init_MC(self, T, MC_std):
   
    ''' initialise temperature and standard deviation for random moves '''

    '''
    T: temperature (float)
    '''

    self.integrator = 'mc'
    self.T = T
    self.MC_std = MC_std


  def init_output(self, x=False, v=False, l=False, e_pot=False, e_kin=False):

    ''' initialise boolean output variables '''

    '''
    x: output positions (bool)
    v: output velocities (bool)
    l: output chain lengths (bool)
    e_pot: output potential energy (bool)
    e_kin: output kinetic energy (bool)
    '''

    self.x_bool = x
    self.v_bool = v
    self.l_bool = l
    self.e_pot_bool = e_pot
    self.e_kin_bool = e_kin


  def init_SHO(self, x_0, v_0):
    
    ''' initialise position and velocity of simple harmonic oscillator '''

    '''
    x_0: initial position (float)
    v_0: initial velocity (float)
    '''

    self.system = 'sho'

    self.x = np.array([x_0])
    self.x_prev = np.array([x_0])
    self.v = np.array([v_0])
    self.compute_forces()


  def init_chain(self, N, L, alpha, distr, sigma):
    
    ''' initialise positions and velocities of chain '''
    
    '''
    N: number of particles (int)
    L: equilibrium spring length (float)
    alpha: potential parameter (float)
    distr: 'uniform' or 'gaussian', statistical distribution for velocities (string)
    sigma: parameter of distribution, max possible velocity (uniform) or 
           standard deviation (gaussian) (float)
    '''

    self.system = 'chain'

    # assign chain parameters

    self.N = N
    self.L = L

    # initialise positions

    self.x = np.linspace(0.0, (N-1)*L, N)
    self.x_prev = np.linspace(0.0, (N-1)*L, N)

    # initialise velocities according to chosen distribution

    if distr == 'uniform':
      self.v = np.random.uniform(-sigma, sigma, size=N)
    elif distr == 'gaussian':
      self.v = np.random.normal(0.0, sigma, size=N)
    else:
      print("Error! Input #3 should be either 'uniform' or 'gaussian'")

    # shift velocities such that COM speed is zero
    
    drift = np.sum(self.v) / N
    self.v -= drift
    
    self.compute_forces()


  def potential(self, x):
    
    ''' compute potential energy of a spring '''
   
    '''
    x: spring extension (float)
    '''

    # use Horner scheme to minimise number of operations

    if self.system == 'sho':
      return 0.5 * x * x
    elif self.system == 'chain':
      return x*x*(1.0 - x*(self.alpha - 0.1*x))
  
  
  def force(self, x):

    ''' compute force of a spring '''

    '''
    x: spring extension (float)
    '''

    # use Horner scheme to minimise number of operations

    if self.system == 'sho':
      return -x
    elif self.system == 'chain':
      return -x*(2.0 - x*(3.0*self.alpha - 0.4*x))


  def compute_forces(self):
    
    ''' update forces acting on all particles '''
    
    if self.system == 'sho':
      self.f = np.array([self.force(self.x[0])])
    elif self.system == 'chain':
      delta = self.x[1:] - self.x[:-1] - self.L
      forces = self.force(delta)
      zero = np.array([0.0])
      f_right = np.concatenate((forces, zero))
      f_left = np.concatenate((zero, forces))
      self.f = f_left - f_right


  def compute_potential(self):
    
    ''' compute total potential energy of the system '''

    if self.system == 'sho':
      return self.potential(self.x[0])
    elif self.system == 'chain':
      delta = self.x[1:] - self.x[:-1] - self.L
      energies = self.potential(delta)
      return np.sum(energies)


  def compute_kinetic(self):
    
    ''' compute total kinetic energy of the system '''

    return 0.5 * np.dot(self.v, self.v)


  def integrate(self):
    
    ''' integrate a single time step, update positions and velocities '''

    # copy for brevity

    k = self.k
    T = self.T
    N = self.N
    dt = self.dt
    gamma = self.gamma

    # compute positions and velocities based on given integrator
    
    if self.integrator == 'euler':
      # forward Euler method
      self.x += dt * self.v
      self.v += dt * self.f
      self.compute_forces()
    elif self.integrator == 'verlet':
      # Verlet method
      if self.n == 0:
        # first step with Euler
        self.x += dt * self.v
        self.v += dt * self.f
        self.compute_forces()
      else: 
        # all other steps with Verlet
        new_x = 2.0*self.x - self.x_prev + dt*dt*self.f
        self.x_prev = self.x
        self.x = new_x
        self.v = 1.0/dt * (self.x - self.x_prev)
        self.compute_forces()
    elif self.integrator == 'velocity_verlet':
      # Velocity Verlet method
      self.v += 0.5*dt * self.f
      self.x += dt * self.v
      self.compute_forces()
      self.v += 0.5*dt * self.f
    elif self.integrator == 'baoab':
      # Langevin thermostat with BAOAB algorithm
      self.v += 0.5*dt * self.f
      self.x += 0.5*dt * self.v
      r = np.random.normal(0.0, 1.0, size=self.N)
      r *= np.sqrt(k*T*(1.0 - np.exp(-2.0*gamma*dt)))
      self.v = np.exp(-gamma*dt)*self.v + r
      self.x += 0.5*dt * self.v
      self.compute_forces()
      self.v += 0.5*dt * self.f
    elif self.integrator == 'mc':
      # initialise potential energy at first step
      if self.n == 0:
        self.e_pot = self.compute_potential()
      # Monte Carlo integrator
      # normal distribution for displacements
      disp = np.random.uniform(-self.MC_std, self.MC_std, size=self.N)
      old_x = self.x[:]
      old_pot = self.e_pot
      new_x = old_x + disp
      self.accepted += 1
      # assume move was accepted
      self.x = new_x
      new_pot = self.compute_potential()
      self.e_pot = new_pot
      # if move discarded return to old configuration
      if old_pot < new_pot:
        p_accept = np.exp((old_pot-new_pot)/(self.k*self.T))
        if np.random.uniform(0.0, 1.0) > p_accept:
          self.x = old_x
          self.e_pot = old_pot
          self.accepted -= 1


  def run(self, steps):
    
    ''' run MD simulation '''

    '''
    steps: number of time steps or MC configurations (int)
    '''

    # initialise containers for history
    if self.x_bool:
      x_hist = (steps+1) * [0]
      x_hist[0] = np.copy(self.x)
    if self.v_bool:
      v_hist = (steps+1) * [0]
      v_hist[0] = np.copy(self.v)
    if self.l_bool:
      l_hist = (steps+1) * [0]
      l_hist[0] = self.x[-1] - self.x[0]
    if self.e_pot_bool:
      e_pot_hist = (steps+1) * [0]
      e_pot_hist[0] = np.copy(self.compute_potential())
    if self.e_kin_bool:
      e_kin_hist = (steps+1) * [0]
      e_kin_hist[0] = np.copy(self.compute_kinetic())

    # run simulation and update history
    for i in range(steps):
      self.integrate()
      self.n += 1
      if self.x_bool:
        x_hist[i+1] = np.copy(self.x)
      if self.v_bool:
        v_hist[i+1] = np.copy(self.v)
      if self.l_bool:
        l_hist[i+1] = self.x[-1] - self.x[0]
      if self.e_pot_bool:
        e_pot_hist[i+1] = np.copy(self.compute_potential())
      if self.e_kin_bool:
        e_kin_hist[i+1] = np.copy(self.compute_kinetic())

    # copy to object
    if self.x_bool:
      self.x_hist = np.array(x_hist)
    if self.v_bool:
      self.v_hist = np.array(v_hist)
    if self.l_bool:
      self.l_hist = np.array(l_hist)
    if self.e_pot_bool:
      self.e_pot_hist = np.array(e_pot_hist)
    if self.e_kin_bool:
      self.e_kin_hist = np.array(e_kin_hist)
