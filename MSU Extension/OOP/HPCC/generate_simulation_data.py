import numpy as np

class Field:
    def __init__(self, num_points, initial_conditions, dx, dt):
        self.num_points = num_points
        self.values = np.array(initial_conditions)
        self.dx = dx
        self.dt = dt
        self.prev_values = np.copy(self.values)

    def store_prev_values(self):
        self.prev_values = np.copy(self.values)
        
    def operate(self, other, operation):
        # Handling operations with another Field instance
        if isinstance(other, Field):
            # Ensure the fields are compatible for operation
            if self.dx == other.dx and self.dt == other.dt and self.num_points == other.num_points:
                result_values = operation(self.values, other.values)
            else:
                raise ValueError("Fields have different resolutions and cannot be operated directly.")
        # Handling operations with scalars or numpy arrays
        else:
            result_values = operation(self.values, other)
        # Always return a new Field instance with the result
        return Field(self.num_points, result_values, self.dx, self.dt)
        
    def __mul__(self, other):
        return self.operate(other, np.multiply)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        return self.operate(other, np.add)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.operate(other, np.subtract)
    
    def __rsub__(self, other):
        # For reverse subtraction, define a custom operation to invert operands
        return self.operate(other, lambda a, b: b - a)
    
    def __truediv__(self, other):
        return self.operate(other, np.divide)
    
    def __rtruediv__(self, other):
        # For reverse division, define a custom operation to invert operands
        return self.operate(other, lambda a, b: b / a)
    
    def __pow__(self, other):
        return self.operate(other, np.power)

    # Comparison operations return numpy boolean arrays, not Field instances
    def __ge__(self, other):
        return self.values >= (other.values if isinstance(other, Field) else other)

    def __le__(self, other):
        return self.values <= (other.values if isinstance(other, Field) else other)

    def __gt__(self, other):
        return self.values > (other.values if isinstance(other, Field) else other)

    def __lt__(self, other):
        return self.values < (other.values if isinstance(other, Field) else other)

    # Specialized, shortened roll functions to prevenet index errors
    def r(self, shift=1):
        return np.roll(self.values, -shift)

    def l(self, shift=1):
        return np.roll(self.values, shift)

    # Derivative Calculators
    def ddx(self, direction='r'):
        """
        Calculate the first derivative using upwinding or central differencing.
        """
        if direction == 'l':
            return (3*self.values - 4*self.l() + self.l(2)) / (2*self.dx)
        elif direction == 'r':
            return (-self.r(2) + 4*self.r() - 3*self.values) / (2*self.dx)
        elif direction == 'c':
            return (self.r() - self.l()) / (2 * self.dx)
        
    def d2dx2(self):
        """
        Calculate the second derivative using central differencing.
        """
        return (self.r() - 2 * self.values + self.l()) / self.dx**2

class Species:
    def __init__(self, density_initial, vel_initial, temp_initial, dx, dt, mass=1.0, gamma=1.0, thermal_diffusivity=0.01):
        self.density = Field(len(density_initial), density_initial, dx, dt)
        self.vel = Field(len(vel_initial), vel_initial, dx, dt)
        self.temp = Field(len(temp_initial), temp_initial, dx, dt)
        self.potential = Field(len(density_initial), np.zeros_like(density_initial), dx, dt)
        self.correlations = Field(len(density_initial), np.zeros_like(density_initial), dx, dt) 
        self.mass = mass
        self.gamma = gamma
        self.thermal_diffusivity = thermal_diffusivity

    def solve_poisson_periodic(self):
        """
        Implement the periodic Poisson solver here using the Modified Thomas Algorithm
        """
        a, b, c = -1.0, 2.0, -1.0
        d = -4 * np.pi * self.density.dx**2 * (self.density.values - np.mean(self.density.values))
        beta_modified = np.full(self.density.num_points, b)
        r_modified = d.copy()
        beta_modified[0] -= a
        beta_modified[-1] -= c
        r_modified[0] -= a * d[-1]
        r_modified[-1] -= c * d[0]

        for i in range(1, self.density.num_points):
            m = a / beta_modified[i-1]
            beta_modified[i] -= m * c
            r_modified[i] -= m * r_modified[i-1]

        # Initialize an array to store the solution of the Poisson equation
        potential_solution = np.zeros(self.density.num_points)
        potential_solution[-1] = r_modified[-1] / beta_modified[-1]
        for i in range(self.density.num_points - 2, -1, -1):
            potential_solution[i] = (r_modified[i] - c * potential_solution[i+1]) / beta_modified[i]
        potential_solution[0] = potential_solution[-1]  # Enforce periodicity

        # Update the potential field values with the solution
        self.potential.values = potential_solution

    ###########
    ## TODO: ##
    ###########
    def calculate_correlations(self):
        """
        Placeholder method for calculating correlations.
        Currently returns a zero array of the same shape as the density.
        This should be replaced with the actual correlation calculation logic.
        """
        # Return zeros for now, as we don't have the actual correlation logic
        self.correlations.values = np.zeros_like(self.density.values)

    def update_density(self):
        """
        Density update using upwind scheme for advection term, with velocity inside the derivative.
        """
        flux_n = np.where(self.vel >= 0,
                          (self.density * self.vel).ddx('l'),
                          (self.density * self.vel).ddx('r'))
        
        self.density.values -= self.density.dt * flux_n

    def update_velocity(self):
        """
        Velocity update using upwind scheme for the advective (nonlinear) term and central differencing for potential gradient.
        """
        # Advective term u * du/dx
        advective_term = np.where(self.vel >= 0,
                                  (self.vel * self.vel.ddx('l')).values,
                                  (self.vel * self.vel.ddx('r')).values)
        
        # Pressure gradient term (1/n) * d(nT)/dx
        pressure_term = (self.density * self.temp).ddx('c') / self.density.values

        # Potential gradient term Gamma * T * dPhi/dx
        dpotential_dx = self.gamma * self.temp.values * self.potential.ddx('c')
        
        # Correlation term - C/n
        correlation_term = self.correlations.ddx('c') / self.density.values
                
        # Update velocity
        self.vel.values -= self.vel.dt * (advective_term - pressure_term + dpotential_dx - correlation_term)

    def update_temperature(self):
        """
        Temperature update using upwind scheme for advection and central differencing for diffusion.
        """
        # Advective term: u * dT/dx
        advective_term = np.where(self.vel >= 0,
                                  self.vel.values * self.temp.ddx('l'),
                                  self.vel.values * self.temp.ddx('r'))

        # Calculate gradients
        dT_dx = self.temp.ddx('c')  # Temperature gradient
        dn_dx = self.density.ddx('c')  # Density gradient

        # Modified diffusive term: (1/n) * (dn/dx) * lambda * dT/dx
        modified_diffusive_term = (1 / self.density.values) * dn_dx * self.thermal_diffusivity * dT_dx

        # Pure diffusion term: lambda * d^2T/dx^2
        diffusion_term = self.thermal_diffusivity * self.temp.d2dx2()

        # Update temperature values
        self.temp.values -= self.temp.dt * (advective_term + modified_diffusive_term + diffusion_term)
        
    def update(self):
        self.density.store_prev_values()
        self.vel.store_prev_values()
        self.temp.store_prev_values()
        
        self.solve_poisson_periodic()
        self.calculate_correlations()
        self.update_density()
        self.update_velocity()
        self.update_temperature()
        
class Orchestrator:
    def __init__(self, species, num_snapshots):
        self.species = species
        self.num_snapshots = num_snapshots
        self.density_snapshots = np.zeros((num_snapshots, species.density.num_points))
        self.vel_snapshots = np.zeros((num_snapshots, species.vel.num_points))
        self.temp_snapshots = np.zeros((num_snapshots, species.temp.num_points))
        self.potential_snapshots = np.zeros((num_snapshots, species.potential.num_points))
        self.snapshot_counter = 0
        self.invalid_run = False

    def take_snapshot(self):
        if self.snapshot_counter < self.num_snapshots:
            if not self.check_invalid_data():
                self.density_snapshots[self.snapshot_counter] = self.species.density.values
                self.vel_snapshots[self.snapshot_counter] = self.species.vel.values
                self.temp_snapshots[self.snapshot_counter] = self.species.temp.values
                self.potential_snapshots[self.snapshot_counter] = self.species.potential.values
                self.snapshot_counter += 1
            else:
                self.invalid_run = True  # Mark the run as invalid and stop further processing

    def check_invalid_data(self):
        if (np.isnan(self.species.density.values).any() or np.isinf(self.species.density.values).any() or
            np.isnan(self.species.vel.values).any() or np.isinf(self.species.vel.values).any() or
            np.isnan(self.species.temp.values).any() or np.isinf(self.species.temp.values).any()):
            return True
        if not (np.all(self.species.density.values > 0) and np.all(self.species.density.values <= 1)):
            return True
        return False
    
    def run_sim(self, num_steps):
        snapshot_interval = max(1, num_steps // self.num_snapshots)
        
        for step in range(num_steps):
            self.species.update()
            
            if step % snapshot_interval == 0 or step == num_steps - 1:
                self.take_snapshot()
                if self.invalid_run:
                    break  # Exit early if invalid data was found            

def sinusoidal(x, frequency, phase):
    return np.sin(frequency * x + phase)

def gaussian(x, mean, std_dev):
    return np.exp(-((x - mean) ** 2 / (2 * std_dev ** 2)))

def constant(x, value):
    return np.full_like(x, value)

def random_noise(x, low, high):
    return np.random.uniform(low, high, size=x.shape)

def zero(x):
    return np.zeros_like(x)

def run_simulation_with_dynamic_conditions(num_samples, grid_size, num_snapshots, num_steps, L, dx, dt):
    density_mean = 3 / (4 * np.pi)
    x = np.linspace(0, L, grid_size, endpoint=False)

    data = []
    initial_condition_types_array = []
    initial_condition_params_array = []
    initial_condition_equilibria_array = []
    initial_condition_perturbation_amplitudes_array = []

    while len(data) < num_samples:
        initial_condition_types = np.random.choice(['sinusoidal', 'gaussian', 'constant', 'random_noise', 'zero'], 3, replace=True)
        initial_condition_params = {
            'sinusoidal': {'frequency': 2*np.pi / L, 'phase': np.random.uniform(0, 2*np.pi)},
            'gaussian': {'mean': np.random.uniform(L*0.25, L*0.75), 'std_dev': np.random.uniform(0.5, 2)},
            'constant': {'value': np.random.uniform(0.5, 1.5)},
            'random_noise': {'low': -0.1, 'high': 0.1},
            'zero': {}
        }

        initial_density_type = globals()[initial_condition_types[0]](x, **initial_condition_params[initial_condition_types[0]])
        initial_velocity_type = globals()[initial_condition_types[1]](x, **initial_condition_params[initial_condition_types[1]])
        initial_temperature_type = globals()[initial_condition_types[2]](x, **initial_condition_params[initial_condition_types[2]])

        # Generate random initial equilibrium conditions
        density_equilibrium = density_mean
        velocity_equilibrium = np.random.uniform(-5, 5)
        temperature_equilibrium = np.random.uniform(0, 1)

        # Generate random perturbation amplitude
        initial_perturbation_amplitudes = np.random.choice([0.0001, 0.001, 0.01, 0.1], 3, replace=True)
        initial_density_perturbation_amplitude = initial_perturbation_amplitudes[0]
        initial_velocity_perturbation_amplitude = initial_perturbation_amplitudes[1]
        initial_temperature_perturbation_amplitude = initial_perturbation_amplitudes[2]

        # Initialize initial conditions based on the selected types
        initial_density = density_equilibrium + initial_density_perturbation_amplitude * initial_density_type
        initial_vel = velocity_equilibrium + initial_velocity_perturbation_amplitude * initial_velocity_type
        initial_temp = temperature_equilibrium + initial_temperature_perturbation_amplitude * initial_temperature_type

        # Roll to introduce asymmetry
        roll_shift = np.random.randint(0, grid_size)
        initial_density = np.roll(initial_density, roll_shift)
        initial_vel = np.roll(initial_vel, roll_shift)
        initial_temp = np.roll(initial_temp, roll_shift)

        species = Species(initial_density, initial_vel, initial_temp, dx, dt)
        orchestrator = Orchestrator(species, num_snapshots)
        orchestrator.run_sim(num_steps)

        if not orchestrator.invalid_run:
            data_sample = np.stack([
                orchestrator.density_snapshots,
                orchestrator.vel_snapshots,
                orchestrator.temp_snapshots
            ], axis=0)  # Shape: (3, num_snapshots, num_points)
            data.append(data_sample)
            initial_condition_types_array.append(initial_condition_types)
            initial_condition_params_array.append(initial_condition_params)
            initial_condition_equilibria_array.append(np.array([density_equilibrium, velocity_equilibrium, temperature_equilibrium]))
            initial_condition_perturbation_amplitudes_array.append(initial_perturbation_amplitudes)

    data = np.array(data)
    initial_condition_types_array = np.array(initial_condition_types_array)
    initial_condition_params_array = np.array(initial_condition_params_array)
    initial_condition_equilibria_array = np.array(initial_condition_equilibria_array)
    initial_condition_perturbation_amplitudes_array = np.array(initial_condition_perturbation_amplitudes_array)
    
    return data, initial_condition_types_array, initial_condition_params_array, initial_condition_equilibria_array, initial_condition_perturbation_amplitudes_array