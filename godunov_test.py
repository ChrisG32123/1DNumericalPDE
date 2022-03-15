#
# GodunovBurgers.py -- a Godunov solver for Burgers' inviscid equation
#
# Written by Daniel J. Bodony (bodony@illinois.edu)
# Monday, March  5, 2018 20:03:02 -0500
# Updated: Tuesday, March 26, 2019 12:57:40 -0500
#

# load Python modules
import numpy as np
import matplotlib.pyplot as plt


def godunov_test():
    # user-defined constants
    Nx = 100  # number of spatial grid cells
    Nt = 400  # number of time steps
    a = -3.0  # left-most grid point
    b = 3.0  # right-most grid point
    t0 = 0.0  # initial time
    tf = 2.0  # final time
    i_uL = 2.0  # initial left state
    i_uR = 3.0  # initial right state

    # derived constants
    L = b - a  # spatial grid length
    T = tf - t0  # final time

    # set time array
    dt = T / (Nt - 1)
    time = np.array([range(0, Nt)]) * dt

    # set x arrays
    dx = L / Nx
    x = np.zeros((Nx + 1, 1))
    xc = np.zeros((Nx, 1))
    xf = np.zeros((Nx + 1, 1))
    for i in range(0, Nx + 1):
        x[i] = i * dx + a
        xf[i] = x[i]
    for i in range(0, Nx):
        xc[i] = 0.5 * (xf[i] + xf[i + 1])

    # initialize the state
    u = np.zeros((Nx, Nt), dtype=float)
    for i in range(0, Nx):
        if xc[i] < a + L / 4:
            u[i, 0] = i_uL
        else:
            u[i, 0] = i_uR

    # define flux vector
    F = np.zeros((Nx + 1, 1))

    # define the flux function
    def flux(u):
        return 0.5 * u ** 2

    # define the Central difference numerical flux
    def CentralDifferenceFlux(uL, uR):
        # compute physical fluxes at left and right state
        FL = flux(uL)
        FR = flux(uR)
        return 0.5 * (FL + FR)

    # define the Godunov numerical flux
    def GodunovNumericalFlux(uL, uR):
        # compute physical fluxes at left and right state
        FL = flux(uL)
        FR = flux(uR)
        # compute the shock speed
        s = 0.5 * (uL + uR)
        # from Toro's book
        if (uL >= uR):
            if (s > 0.0):
                return FL
            else:
                return FR
        else:
            if (uL > 0.0):
                return FL
            elif (uR < 0.0):
                return FR
            else:
                return 0.0

    def NumericalFlux(uL, uR):
        # return CentralDifferenceFlux(uL,uR)
        return GodunovNumericalFlux(uL, uR)

    # time integrate
    print('Starting integration ...', end=' ')
    for n in range(0, Nt - 1):

        # estimate the CFL
        CFL = max(abs(u[:, n])) * dt / dx
        if CFL > 0.5:
            print("Warning: CFL > 0.5")

        # compute the interior fluxes
        for i in range(1, Nx):
            uL = u[i - 1, n]
            uR = u[i, n]
            F[i] = NumericalFlux(uL, uR)

        # compute the left boundary flux
        if u[0, n] < 0.0:
            uL = 2.0 * u[0, n] - u[1, n]
        else:
            uL = u[0, 0]
        uR = u[0, n]
        F[0] = NumericalFlux(uL, uR)

        # compute the right boundary flux
        if u[Nx - 1, n] > 0.0:
            uR = 2.0 * u[Nx - 1, n] - u[Nx - 2, n]
        else:
            uR = u[Nx - 1, 0]
        uL = u[Nx - 1, n]
        F[Nx] = NumericalFlux(uL, uR)

        # update the state
        for i in range(0, Nx):
            u[i, n + 1] = u[i, n] - dt / dx * (F[i + 1] - F[i])

    print('Done.')

    fig = plt.figure()
    plt.plot(xc, u[:, 0])
    plt.show()
