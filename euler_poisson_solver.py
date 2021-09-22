import numpy as np
import matplotlib.pyplot as plt

# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + conv(n-n_mean,c_hat)
# phi_xx = f(x,t) = 3 - 4*pi*n


def euler_poisson():
    # Define Domain
    # CFL Condition: dt / dx < .7
    N = int(1e3)  # grid points
    T = int(4e3)  # time steps
    L = 10
    x = np.linspace(0, L - L / N, N)
    dx = x[2] - x[1]
    dt = 5e-4  # step size

    # Memory Allocation
    num_snapshot = 10  # input("Number of Snapshots = ") # number of snapshots
    num_snapstep = int(T / num_snapshot)  # time steps between each snapshot

    n = np.zeros(N)
    n_snaps = np.zeros((num_snapshot, N))

    u = np.zeros(N)
    u_snaps = np.zeros((num_snapshot, N))

    phi = np.zeros(N)
    phi_snaps = np.zeros((num_snapshot, N))
    phi_coefs = np.ones(N)

    b = 1
    k = np.linspace(-np.pi/dx, np.pi * (1 - 2/N) / dx, N)
    k = np.abs(k)
    c_hat = 4 * np.pi * ((b * k * k - 1) * np.sin(k) + k * np.cos(k)) / (k * k * k + 3 * ((b * k**2 - 1) * np.sin(k) + k * np.cos(k)))

    # Initial condition
    n_mean = 3 / (4 * np.pi)
    IC_var = 5 * 2 * np.pi / L
    n_0 = n_mean + 1e-4 * np.sin(IC_var * x)
    u_0 = np.zeros(N) + 1e-4 * np.sin(IC_var * x)

    # Setting Initial Conditions
    n = n_0
    # nTot = np.zeros(N)
    u = u_0
    # uTot = np.zeros(N)
    # phiTot = np.zeros(N)
    gamma = 1  # input("Coulomb Coupling Parameter:")

    # Convolution Function
    def conv(delta_n, fft_c):
        delta_n = np.fft.fftshift(np.fft.fft(delta_n))
        nc = fft_c * delta_n
        nc = np.fft.ifft(np.fft.ifftshift(nc))
        return nc

    # Choose flux functions
    def f_n(n, u):
        return n * u

    def f_u(n, u, v):
        return .5 * u * u + np.log(n) + v + np.real(conv(n - n_mean, c_hat))

    # Solve

    counter = 0  # Counts total iterations of t
    snap_counter = 0  # Counts which array to copy snapshots to
    for tt in range(T):

        counter = counter + 1
        if counter % num_snapstep == 0:  # When to take snapshot
            n_snaps[snap_counter] = n
            u_snaps[snap_counter] = u
            phi_snaps[snap_counter] = phi
            snap_counter = snap_counter + 1

        # Phi
        # Define f(x)
        f = 3 - 4 * np.pi * dx * dx * n
        f = f - np.mean(f)

        # First sweep
        phi_coefs[0] = -0.5
        f[0] = -0.5 * f[0]

        for ii in range(1, N - 1):
            phi_coefs[ii] = -1 / (2 + phi_coefs[ii - 1])
            f[ii] = (f[ii - 1] - f[ii]) / (2 + phi_coefs[ii - 1])

        # Second sweep
        phi[0] = f[N - 1] - f[N - 2]
        for ii in range(1, N - 1):
            phi[ii] = (f[ii - 1] - phi[ii - 1]) / phi_coefs[ii - 1]

        # Lax-Friedrichs time-stepping
        # Central differencing in space
        F_0 = f_n(n, u)
        Fnplus = np.roll(F_0, -1)
        Fnminus = np.roll(F_0, 1)
        nplus = np.roll(n, -1)
        nminus = np.roll(n, 1)
        n = 0.5 * (nplus + nminus) - 0.5 * dt * (Fnplus - Fnminus) / dx

        F_1 = f_u(n, u, gamma * phi)
        Fuplus = np.roll(F_1, -1)
        Fuminus = np.roll(F_1, 1)
        uplus = np.roll(u, -1)
        uminus = np.roll(u, 1)
        u = 0.5 * (uplus + uminus) - .5 * (dt / dx) * (Fuplus - Fuminus)

        # Measure integral over n(x,t), u(x,t), phi(x,t)
        # nTot[tt] = np.sum(n) * dx
        # uTot[tt] = np.sum(u) * dx
        # phiTot[tt] = np.sum(phi) * dx

    # Plotting
    for ii in range(num_snapshot - 1):
        plt.plot(x, phi_snaps[ii], label="Snapshot " + str(ii))
    plt.plot(x, phi, label='phi')
    plt.title("Electrostatic Force")
    plt.legend()

    plt.figure()
    plt.plot(x, n_0, label="n_0")
    for ii in range(num_snapshot - 1):
        plt.plot(x, n_snaps[ii], label="Snapshot " + str(ii))
    plt.plot(x, n, label="n")
    plt.title("Density")
    plt.legend()

    plt.figure()
    plt.plot(x, u_0, label="u_0")
    for ii in range(num_snapshot - 1):
        plt.plot(x, u_snaps[ii], label="Snapshot " + str(ii))
    plt.plot(x, u, label="u")
    plt.title("Velocity")
    plt.legend()

    # Plot integral of phi(x,t), n(x,t), and u(x,t) over time
    # plt.plot(phiTot, label = "phiTot")
    # plt.plot(nTot, label = "nTot")
    # plt.plot(uTot, label = "uTot")

    plt.show(block=True)
    plt.interactive(False)
