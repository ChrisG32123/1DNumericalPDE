import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
# import pywt
from pydmd import DMD

# Base class for field variables
class FieldVariable:
    def __init__(self, initial_conditions, grid_size, dx):
        self.values = initial_conditions
        self.grid_size = grid_size
        self.dx = dx
        self.snapshots = [np.copy(self.values)]  # Include the initial condition
    
    # Return the array shifted right
    def r(self, array):
        return np.roll(array, -1)

    # Return the array shifted left
    def l(self, array):
        return np.roll(array, 1)

    # Placeholder method for update rule, to be implemented in subclasses
    def update(self):
        raise NotImplementedError("Update method must be implemented in subclass.")

    # Method to step through time
    def step(self):
        self.update()
        # Store the current snapshot
        self.snapshots.append(np.copy(self.values))

# Subclass for the density, which is a field variable
class Density(FieldVariable):
    def __init__(self, initial_conditions, grid_size, dx, velocity):
        super().__init__(initial_conditions, grid_size, dx)
        self.velocity = velocity
        self.potential = np.zeros(grid_size)

    def solve_poisson_periodic(self):
        """
        1D Laplacian Solver for the Electrostatic Potential. Modified Thomas Algorithm used to reduce computational complexity to O(n).
        Modified Thomas Algorithm is a combination of the Thomas Algorithm for tridiagonal matrices & Sherman-Morrison for the periodic boundary conditions
        """
        a = -1.0  # Off-diagonal term
        b = 2.0   # Diagonal term
        c = -1.0  # Off-diagonal term
        
        # Adjust the right-hand side of the Poisson equation to incorporate the constant term
        d = -4 * np.pi * self.dx**2 * (self.values - np.mean(self.values))

        # Arrays for the modified Thomas algorithm
        beta_modified = np.full(self.grid_size, b)  # Diagonal terms
        r_modified = d.copy()  # Right-hand side
        
        # Modify the first and last elements of the diagonal and right-hand side for periodic boundaries
        beta_modified[0] -= a
        beta_modified[-1] -= c
        r_modified[0] -= a * d[-1]
        r_modified[-1] -= c * d[0]
        
        # Forward sweep
        for i in range(1, self.grid_size):
            m = a / beta_modified[i-1]
            beta_modified[i] -= m * c
            r_modified[i] -= m * r_modified[i-1]
        
        # Backward substitution
        phi = np.zeros(self.grid_size)
        phi[-1] = r_modified[-1] / beta_modified[-1]
        for i in range(self.grid_size - 2, -1, -1):
            phi[i] = (r_modified[i] - c * phi[i+1]) / beta_modified[i]

        # Enforce periodicity: phi[0] should be equal to phi[-1] due to periodic boundaries
        phi[0] = phi[-1]

        # The final solution phi represents the electrostatic potential
        self.potential = phi  # Store the updated potential

    def update(self):
        # First, solve the Poisson equation to update the potential
        self.solve_poisson_periodic()
        # Implementing upwinding scheme for density using r() and l() methods
        new_values = np.empty_like(self.values)
        for i in range(self.grid_size):
            if self.velocity.values[i] >= 0:
                # Upwind using left values for positive velocity
                flux = self.values[i] * self.velocity.values[i]
                flux_left = self.l(self.values)[i] * self.velocity.values[i]
            else:
                # Upwind using right values for negative velocity
                flux = self.r(self.values)[i] * self.velocity.values[i]
                flux_left = self.values[i] * self.velocity.values[i]
            new_values[i] = self.values[i] - self.dt * (flux - flux_left) / self.dx

        # Update the field variable with the new computed values
        self.values = new_values


# Subclass for the velocity, which is a field variable
class Velocity(FieldVariable):
    def __init__(self, initial_conditions, grid_size, dx, density, temperature, mass, gamma):
        super().__init__(initial_conditions, grid_size, dx)
        self.density = density
        self.temperature = temperature
        self.mass = mass
        self.gamma = gamma

    def update(self):
        # Ensure that the density's potential has been updated before calling this method
        
        ####################
        #TODO: FIX POISSON #
        ####################
        # potential = self.density.potential  # Access the potential from the Density instance
        potential = np.zeros(len(self.density.potential))

        new_values = np.empty_like(self.values)
        for i in range(self.grid_size):
            if self.values[i] >= 0:
                # Upwind using left values for positive velocity
                du_dx = (self.values[i] - self.l(self.values)[i]) / self.dx
                dlogn_dx = (np.log(self.density.values[i]) - np.log(self.l(self.density.values)[i])) / self.dx
                dpotential_dx = (potential[i] - self.l(potential)[i]) / self.dx
            else:
                # Upwind using right values for negative velocity
                # Wrap around the index for periodic boundary conditions
                du_dx = (self.r(self.values)[i] - self.values[i]) / self.dx
                dlogn_dx = (np.log(self.r(self.density.values)[i]) - np.log(self.density.values[i])) / self.dx
                dpotential_dx = (self.r(potential)[i] - potential[i]) / self.dx
            
            nonlinear_term = self.values[i] * du_dx
            grad_logn = dlogn_dx
            grad_potential = dpotential_dx

            # Update velocity values
            # The temperature variable must have been updated before calling this method
            new_values[i] = self.values[i] - self.dt * (
                nonlinear_term +
                (self.temperature.values[i] / self.mass) * grad_logn +
                self.gamma * grad_potential
            )
        
        # Update the field variable with the new computed values
        self.values = new_values


class Temperature(FieldVariable):
    def __init__(self, initial_conditions, grid_size, dx, velocity, density, thermal_diffusivity):
        super().__init__(initial_conditions, grid_size, dx)
        self.velocity = velocity
        self.density = density
        self.thermal_diffusivity = thermal_diffusivity

    def update(self):
        new_values = np.empty_like(self.values)
        for i in range(self.grid_size):
            # Advection terms
            advective_flux_T = 0
            advective_flux_logn = 0
            # if self.velocity.values[i] >= 0:
            #     advective_flux_T = self.velocity.values[i] * (self.values[i] - self.l(self.values)[i]) / self.dx
            #     advective_flux_logn = self.velocity.values[i] * (np.log(self.density.values[i]) - np.log(self.l(self.density.values)[i])) / self.dx
            # else:
            #     advective_flux_T = self.velocity.values[i] * (self.r(self.values)[i] - self.values[i]) / self.dx
            #     advective_flux_logn = self.velocity.values[i] * (np.log(self.r(self.density.values)[i]) - np.log(self.density.values[i])) / self.dx

            # Central differencing for diffusion term
            diffusive_flux = (self.r(self.values)[i] - 2*self.values[i] + self.l(self.values)[i]) / self.dx**2

            # Updating the temperature values using the explicit scheme
            new_values[i] = self.values[i] - self.dt * (advective_flux_T * advective_flux_logn + 2 * advective_flux_T - self.thermal_diffusivity * diffusive_flux)

        # Periodic boundary conditions are inherently handled by the r() and l() methods
        self.values = new_values


# Class for plotting
class Plotter:
    @staticmethod
    def plot_heatmap(data, x_domain, y_domain, title, cmap_type='hot'):
        fig, ax = plt.subplots()
        cax = ax.imshow(data, cmap=cmap_type, interpolation='nearest', aspect='auto', 
                        extent=[x_domain[0], x_domain[-1], y_domain[0], y_domain[-1]], origin='lower')
        fig.colorbar(cax)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Space')
        plt.show(block=False)
        return fig, ax

    @staticmethod
    def plot_line(data_array, x_domain, title):
        fig, ax = plt.subplots()
        for data in data_array:
            ax.plot(x_domain, data)
        ax.set_title(title)
        ax.set_xlabel('Domain')
        ax.set_ylabel('Value')
        ax.grid(True)
        plt.show(block=False)
        return fig, ax
    
    @staticmethod
    def plot_fourier_transform(signal, sampling_rate, title='Fourier Transform'):
        N = len(signal)
        T = 1.0 / sampling_rate
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(N, T)[:N//2]
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax.set_title(title)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        plt.show(block=False)
        return fig, ax

    @staticmethod
    def plot_2d_fourier_transform(data, title='2D Fourier Transform'):
        """
        Plots the 2D Fourier Transform of the provided data.

        :param data: The 2D data array to transform and plot.
        :param title: The title of the plot.
        """
        # Compute the 2D Fourier Transform
        fourier_transform = np.fft.fft2(data)
        # Shift the zero frequency component to the center of the spectrum
        fshift = np.fft.fftshift(fourier_transform)
        # Calculate the magnitude spectrum
        magnitude_spectrum = np.abs(fshift)
        
        fig, ax = plt.subplots()
        # Use logarithmic scaling to better visualize the spectrum
        ax.imshow(magnitude_spectrum, norm=LogNorm(vmin=1), cmap='hot', aspect='equal')
        ax.set_title(title)
        plt.colorbar(ax.imshow(magnitude_spectrum, norm=LogNorm(vmin=1), cmap='hot'), ax=ax)
        plt.show(block=False)
        return fig, ax
    
    @staticmethod
    def plot_wavelet_transform(signal, scales, waveletname='cmor', title='Wavelet Transform'):
        # coefficients, frequencies = pywt.cwt(signal, scales, waveletname)
        # fig, ax = plt.subplots()
        # cax = ax.contourf(coefficients, cmap='coolwarm')
        # fig.colorbar(cax)
        # ax.set_title(title)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Frequency')
        # plt.show(block=False)
        # return fig, ax
        pass

    @staticmethod
    def animate_solution(data, x_domain, y_label='Value', title='Solution Evolution', interval=200, cmap_type='hot'):
        """
        Creates an animation of the solution's evolution over time.

        :param data: The data to animate, expected shape is (time_steps, spatial_domain).
        :param x_domain: The spatial domain or x-axis values for the plot.
        :param y_label: Label for the y-axis.
        :param title: The title of the plot.
        :param interval: Time interval between frames in milliseconds.
        :param cmap_type: Colormap for the heatmap.
        """
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel('Space')
        ax.set_ylabel(y_label)
        
        # Setting the limits for x and y axes
        ax.set_xlim(x_domain[0], x_domain[-1])
        ax.set_ylim(np.min(data), np.max(data))
        
        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            y = data[i]
            line.set_data(x_domain, y)
            return line,

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(data), interval=interval, blit=True)

        plt.show(block=False)
        return anim


########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################

# Define simulation parameters
grid_size = 100  # Grid size
dx = 1.0  # Spatial step size
dt = 0.1  # Time step size
m = 1.0  # Mass
lambda_thermal = 1  # Thermal diffusivity, adjust as needed
gamma = 100  # Coulomb Coupling Parameter, adjust as needed
num_steps = 250  # Number of time steps
x = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)  # Spatial domain

# Initial conditions
initial_density = 3/(4*np.pi) + 0.1 * np.sin(3/4 * np.pi * x)  # Initial density
initial_velocity = np.sin(x)  # Initial velocity
initial_temperature = 1 + 0.1 * np.cos(x)  # Initial Temperature

# Create field variable instances for density and velocity
density = Density(initial_density, grid_size, dx, None)  # Temporarily no velocity
velocity = Velocity(initial_velocity, grid_size, dx, density, None, m, gamma)
temperature = Temperature(initial_temperature, grid_size, dx, velocity, density, lambda_thermal)
density.velocity = velocity # Now update density with the actual velocity instance
velocity.temperature = temperature # Now update density with the actual temperature instance

density.dt = dt
velocity.dt = dt
temperature.dt = dt

# Simulation function
def run_simulation(density, velocity, temperature, num_steps, plot_frequency, specific_grid_point):
    time_domain = np.linspace(0, num_steps * dt, num_steps + 1)
    space_domain = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)

    #TODO: Use snapshots variable from FieldVariable Class
    density_snapshots = [density.values.copy()]
    velocity_snapshots = [velocity.values.copy()]
    temperature_snapshots = [temperature.values.copy()]
    snapshot_times = [0]  # Time of initial condition

    for i in range(1, num_steps + 1):
        density.step()
        velocity.step()
        temperature.step()
        if i % plot_frequency == 0 or i == num_steps:
            density_snapshots.append(density.values.copy())
            velocity_snapshots.append(velocity.values.copy())
            temperature_snapshots.append(temperature.values.copy())
            snapshot_times.append(i * dt)
    
    return density_snapshots, velocity_snapshots, temperature_snapshots, time_domain, space_domain

# Ensure to provide the correct parameters when calling run_simulation
specific_grid_point = 50
plot_frequency = num_steps // 29  # for 10 snapshots
density_snapshots, velocity_snapshots, temperature_snapshots, time_domain, space_domain = run_simulation(density, velocity, temperature, num_steps, plot_frequency, specific_grid_point)

density_snapshots = np.array(density_snapshots)
velocity_snapshots = np.array(velocity_snapshots)
temperature_snapshots = np.array(temperature_snapshots)

# Output the total duration of the simulation
simulation_duration = num_steps * dt
simulation_duration

# Plotting the heatmaps
Plotter.plot_heatmap(density_snapshots.T, time_domain, space_domain, 'Heatmap of Density Over Time', 'hot')
Plotter.plot_heatmap(velocity_snapshots.T, time_domain, space_domain, 'Heatmap of Velocity Over Time', 'cool')
Plotter.plot_heatmap(temperature_snapshots.T, time_domain, space_domain, 'Heatmap of Temperature Over Time', 'cool')

# Plotting Fourier Transform of the last density snapshot
sampling_rate = 1 / dt
Plotter.plot_fourier_transform(density_snapshots[-1] - np.mean(density_snapshots[-1]), sampling_rate, 'Fourier Transform of Density')

# Plot 2D Fourier Transform of the density
Plotter.plot_2d_fourier_transform(density_snapshots - np.mean(density_snapshots), '2D Fourier Transform of Density')

# Plotting Wavelet Transform of the last density snapshot
# Extract the time series for the specific grid point
specific_point_series = [snapshot[specific_grid_point] for snapshot in density_snapshots]
# Define scales for the wavelet transform
scales = np.arange(1, 128)
# Plotting Wavelet Transform of the specific grid point over time
Plotter.plot_wavelet_transform(specific_point_series, scales, 'cmor', f'Wavelet Transform of Density at Grid Point {specific_grid_point}')

# Animating the solution for density
Plotter.animate_solution(density_snapshots, space_domain, 'Density Value', 'Evolution of Density')
# Optionally, animate the velocity field as well
Plotter.animate_solution(velocity_snapshots, space_domain, 'Velocity Value', 'Evolution of Velocity')
# Optionally, animate the velocity field as well
Plotter.animate_solution(temperature_snapshots, space_domain, 'Temperature Value', 'Evolution of Temperature')

plt.show(block=True)

#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################

# Initialize the DMD object with svd_rank -1 which includes all the singular values
dmd_solver = DMD(svd_rank=-1)
dmd_solver.fit(density_snapshots.T)

# Reconstruct the density matrix using DMD modes
density_dmd = dmd_solver.reconstructed_data.real

# Extract DMD eigenvalues and modes
dmd_modes = dmd_solver.modes.real
dmd_eigenvalues = dmd_solver.eigs

# Extract the amplitudes and frequencies
dmd_amplitudes = dmd_solver.amplitudes
dmd_frequencies = dmd_solver.frequency

# Display the results
print("DMD Eigenvalues: ", dmd_eigenvalues)
print("DMD Modes: ", dmd_modes.shape)
print("DMD Amplitudes: ", dmd_amplitudes)
print("DMD Frequencies: ", dmd_frequencies)

# Plot the DMD spectrum
plt.figure(figsize=(10, 6))
plt.scatter(dmd_frequencies.real, dmd_frequencies.imag, c='r')
plt.title('DMD Frequency Spectrum')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.grid(True)
plt.show()

# Assuming dmd_modes is your modes array from DMD analysis
num_modes = dmd_modes.shape[1]  # Number of modes

# Determine the layout of the subplots
cols = 2  # Number of columns in the subplot grid
rows = num_modes // cols + (num_modes % cols > 0)  # Calculate rows needed

plt.figure(figsize=(10, rows * 3))  # Adjust the figure size as needed
for i, mode in enumerate(dmd_modes.T):
    ax = plt.subplot(rows, cols, i+1)  # Create a subplot for each mode
    ax.plot(mode.real, label=f'Mode {i+1}')  # Plot each mode
    ax.legend()
    ax.set_title(f'DMD Mode {i+1}')

plt.tight_layout()
plt.show()

# Plot the time evolution of the modes in subplots
time_dynamics = np.abs(dmd_solver.dynamics)
rows = time_dynamics.shape[0] // cols + (time_dynamics.shape[0] % cols > 0)

plt.figure(figsize=(10, rows * 3))
for i in range(time_dynamics.shape[0]):
    ax = plt.subplot(rows, cols, i+1)
    ax.plot(time_dynamics[i, :], label=f'Dynamic {i+1}')
    ax.legend()
    ax.set_title(f'Time Evolution of Mode {i+1}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Magnitude')
    ax.grid(True)

plt.tight_layout()
plt.show(block=True)