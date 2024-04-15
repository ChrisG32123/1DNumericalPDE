import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Correcting the mistake in the constructor's parameter naming and ensuring dt is set correctly
class NonlinearAdvectionSolverAdaptiveAB:
    def __init__(self, Nx, L, dt_max):
        self.Nx = Nx
        self.L = L
        self.x = np.linspace(0, L, Nx, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.dt_max = dt_max
        self.t = 0  # Initialize time
        self.prev_du_dx = None  # To store the previous step derivative

    def update_c_and_dt(self):
        self.c = np.sin(2 * np.pi * self.x) * np.cos(2 * np.pi * self.t)
        self.dt = min(self.dt_max, 0.5 * self.dx / np.max(np.abs(self.c)))

    def compute_du_dx(self, cu):
        # Compute spatial derivative using central difference
        return (np.roll(cu, -1) - np.roll(cu, 1)) / (2 * self.dx)

    def solve(self, u0):
        self.update_c_and_dt()
        cu = self.c * u0
        current_du_dx = self.compute_du_dx(cu)

        if self.prev_du_dx is None:
            # First step, use simple forward Euler as fallback
            u_next = u0 - current_du_dx * self.dt
        else:
            # Adams-Bashforth 2-step scheme
            u_next = u0 - (1.5 * current_du_dx - 0.5 * self.prev_du_dx) * self.dt

        # Update for next step
        self.prev_du_dx = current_du_dx
        self.t += self.dt
        return u_next

# Parameters and initialization
Nx = 100
L = 1.0
dt_max = 0.01

# Re-initializing solver and parameters for animation
solver_ab = NonlinearAdvectionSolverAdaptiveAB(Nx, L, dt_max=0.01)
u0_ab = np.sin(2 * np.pi * np.linspace(0, L, Nx, endpoint=False))
total_duration_ab = 2.0  # Total simulation time
steps = int(total_duration_ab / solver_ab.dt_max)  # Estimate steps based on maximum dt

# Animation setup
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Setup for u(x, t) plot
line_u_ab, = axs[0].plot([], [], 'r-', linewidth=2)
axs[0].set_xlim(0, L)
axs[0].set_ylim(-1.5, 1.5)
axs[0].set_title('Solution u(x, t) with Adaptive dt (Adams-Bashforth)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('u')

# Setup for c(x, t) plot
line_c_ab, = axs[1].plot([], [], 'b-', linewidth=2)
axs[1].set_xlim(0, L)
axs[1].set_ylim(-2, 2)
axs[1].set_title('Advection Speed c(x, t) as an Oscillatory Standing Wave')
axs[1].set_xlabel('x')
axs[1].set_ylabel('c')

def init_ab():
    line_u_ab.set_data([], [])
    line_c_ab.set_data([], [])
    return line_u_ab, line_c_ab

def update_ab(frame):
    global u0_ab
    u0_ab = solver_ab.solve(u0_ab)  # Solve for the next step of u
    line_u_ab.set_data(solver_ab.x, u0_ab)  # Update u plot
    line_c_ab.set_data(solver_ab.x, solver_ab.c)  # Update c plot based on current time
    return line_u_ab, line_c_ab

ani_ab = FuncAnimation(fig, update_ab, frames=range(steps), init_func=init_ab, blit=True, interval=50)

plt.tight_layout()
plt.show()
