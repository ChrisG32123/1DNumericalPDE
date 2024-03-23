# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt

def continuity_roe_solver():
# ============== #
# Define Methods #
# ============== #

    # Memory Allocation and Initialization
    def memory_allocation_pde(u_IC):
        u = np.copy(u_IC)
        flux = np.zeros((1, N))
        snap_u = np.zeros((snaps + 1, N))
        snap_u[0] = u_IC
        return u, flux, snap_u

    # Define Numerical Schemes
    def func_flux(n,v):
        # Flux Vector
        Fn = np.array(n * v)

        return Fn

    def flux_roe(n,v):

        roe_flux = np.zeros(N)

        for ii in range(N-1):
            R = np.sqrt(n[ii + 1] / n[ii])
            n_hat = R * n[ii]
            ndif = n[ii + 1] - n[ii]
            roe_flux[ii] = v[ii] * ndif

        R = np.sqrt(n[0] / n[N-1])
        n_hat = R * n[N-1]
        ndif = n[0] - n[N-1]
        roe_flux[N-1] = v[N-1] * ndif

        F = func_flux(n,v)

        # print("roe_flux", roe_flux)

        roe_flux = 0.5 * (F + np.roll(F,1)) - 0.5 * roe_flux

        dF = (roe_flux[1:N] - roe_flux[0:N-1])
        dF_BC = roe_flux[0] - roe_flux[N-1]
        dF = np.hstack([dF,dF_BC])

        return (dF)

    # Miscellaneous
    def take_snapshot(tt, T, snaps, n, snap_n):
        snap_n[int(tt / (T / snaps))] = n

    def plot(x, snap_u):
        plt.figure()
        for ii in range(len(snap_u)):
            plt.plot(x, snap_u[ii], label="T = " + str(ii * dt * T / snaps))
        plt.legend()

    def colormap(xx, yy, snap_u):
        plt.figure()
        clr = plt.contourf(xx, yy, snap_u)
        plt.colorbar()

# ========================== #
# Parameters, Initialization #
# ========================== #

    # Parameters
    N = int(1e2)  # Grid Points 5e2
    T = int(1e4)  # Time Steps 5e3
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    # TODO: x3
    dx = x[2] - x[1]  # Grid Size
    #TODO: define dt = CFL * dx / (max(abs(v)+a), where a = sqrt(gamma * p / n) is the updated speed of sound
    dt = 1e-3  # Time Step Size
    lambda_ = dt / dx
    t = dt * T

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = int(input("Number of Snapshots "))  # Number of Snapshots
    Gamma_0 = float(input("Value of Gamma "))  # Coulomb Coupling Parameter
    kappa_0 = float(input("Value of kappa "))  # screening something

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    v_IC = np.ones(N)
    n_IC[:] = n_0 * (1 + .1 * np.sin(2 * np.pi * x/L))
    # n_IC[0:int(nx / 4)] = rho_0 / 2
    # n_IC[int(nx / 4):int(3*nx / 4)] = 3 * rho_0 / 2
    # n_IC[int(3* nx / 4):nx] = rho_0 / 2
    # v_IC[:] = 2 + np.sin(2*X / Xlngth)

    n, flux_n, snap_n = memory_allocation_pde(n_IC)
    v = np.copy(v_IC)

# ===== #
# Solve #
# ===== #

    # No Correlation
    for tt in range(T + 1):
        # Compute Flux
        dF = flux_roe(n, v)
        n_old = np.copy(n)

        #TODO: Make method AND + .5 * lambda_ * (rhs[:, 1:-2] - rhs[:,1:-2])
        # n0?
        # Check Boundary Conditions

        # n[0] = n0[-1]
        # n[1:nx-1] = n0[1:nx-1] - lambda_ * dF[1:nx-1]
        # n[-1] = n[0]

        # print("dF", dF)

        n[0:N] = n_old[0:N] - lambda_ * dF[0:N]

        print("n[24:26]", n[24:26])
        print("lambda_ * dF[24:26]", lambda_ * dF[24:26])
        print("n[24:26] - lambda_ * dF[24:26]", n[24:26] - lambda_ * dF[24:26])

        if np.amin(n) < 0:
            print("negative density at ", tt)
            # exit()

        if tt % (T / snaps) == 0:
            take_snapshot(tt, T, snaps, n, snap_n)

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    colormap(xx,yy,snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plot(x, snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.show()
