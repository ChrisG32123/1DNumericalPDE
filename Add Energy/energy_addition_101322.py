# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt
import time


def solve():
    # Parameters
    N = int(1e4)  # Grid Points
    T = int(2e1)  # Time Steps
    L = 1  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-5  # Time Step Size
    lmbd = dt / dx
    print(lmbd)
    t = dt*T

    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = 20      # int(input("Number of Snapshots "))    # Number of Snapshots
    Gamma_0 = 1     # float(input("Value of Gamma "))       # Coulomb Coupling Parameter
    kappa_0 = 1     # float(input("Value of kappa "))       # screening something
    snap = 0                                                # Current snapshot in time

    # Memory Allocation
    ntot, nfluxtot= np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    utot, ufluxtot, ucorrtot, urhstot = np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    etot, efluxtot, ecorrtot, erhstot = np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    Qtot, Qfluxtot = np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    phitot = np.zeros((snaps + 1, 2, N))

    n, nflux = np.zeros((2, N)), np.zeros((2, N))
    u, uflux, ucorr, urhs = np.zeros((2, N)), np.zeros((2, N)), np.zeros((2, N)), np.zeros((2, N))
    e, eflux, ecorr, erhs = np.zeros((2, N)), np.zeros((2, N)), np.zeros((2, N)), np.zeros((2, N))
    Q, Qflux = np.zeros((2, N)), np.zeros((2, N))
    phi, phimtx = np.zeros((2, N)), np.zeros((2, N))
    sys, sysflux = np.zeros((3, 2, N)), np.zeros((3, 2, N))

    # Spatial Derivative
    def derivative(array2D):
        return (array2D[:] - np.roll(array2D[:], 1, axis=1)) / dx

    def solvephi(den):
        phi = np.zeros((2,N))
        # phimtx * phi = b
        # Define b
        b = 3 - 4 * np.pi * dx * dx * den
        b = b - np.mean(b)
        # First sweep
        phimtx[:,0] = -0.5
        b[:,0] = -0.5 * b[:,0]
        for ii in range(1, N - 1):
            phimtx[:,ii] = -1 / (2 + phimtx[:,ii - 1])
            b[:,ii] = (b[:,ii - 1] - b[:,ii]) / (2 + phimtx[:,ii - 1])
        # Second sweep
        phi[:,0] = b[:,N - 1] - b[:,N - 2]
        for ii in range(1, N - 1):
            phi[:,ii] = (b[:,ii - 1] - phi[:,ii - 1]) / phimtx[:,ii - 1]
        return phi

    def fft_correlations(k,nc,Gamma, kappa):
        delta_n = nc - n_0
        fhat = np.fft.fftshift(np.fft.fft(delta_n))
        dcfunc = 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)

        conv = fhat * dcfunc
        conv = np.fft.ifft(np.fft.ifftshift(conv))
        conv = np.real(conv)

        return conv

    # Initial Conditions & Initialization
    ICfreq = 2* np.pi / L       # Enforce Periodicity of IC
    ntot[snap,:, ::int(N/2)] = 3/2 * n_0            # n_0 * np.ones(N)
    ntot[snap, :, int(N/2)::] = 1 / 2 * n_0
    utot[snap,:] = np.ones(N)
    etot[snap,:] = np.ones(N)
    Qtot[snap,:] = -derivative(etot[snap,:])
    phitot[snap,:] = solvephi(ntot[snap,:])

    # Define Variables at T = t0
    n, u, e = ntot[snap], utot[snap], etot[snap]
    snap += 1

    st = time.time()
    for tt in range(1, T + 1):

        # Apply Closure
        Q = -derivative(e)

        # Define Flux Functions
        nflux = derivative(n*u)
        uflux = derivative(n*e + u*u)
        eflux = derivative(Q + u + u*u)

        # if tt < 5:
        #     print(n[0]*u[0])
        #     print(nflux[0])

        # Solve Phi
        phi = solvephi(n)

        # Solve Correlation
        # ucorr = np.array([np.zeros(N), fft_correlations(k,n[1], Gamma_0, kappa_0)])          # np.zeros((2,N))           # TODO: Add Correlations
        # ecorr = np.zeros((2,N))           # TODO: Add Correlations

        # Right Hand Side
        # urhs = ucorr - derivative(phi)      # TODO: LINE CHANGE
        # erhs = ecorr - derivative(u)*e      # TODO: LINE CHANGE

        # Vectorize for Solve
        sys, sysflux = np.array([n, u, e]), np.array([nflux, uflux, eflux])
        sysL, sysR = np.roll(sys[:, :, :], 1, axis=-1), np.roll(sys[:, :, :], -1, axis=-1)
        sysfluxL, sysfluxR = np.roll(sysflux[:, :, :], 1, axis=-1), np.roll(sysflux[:, :, :], -1, axis=-1)
        # sysrhs = np.array([np.zeros((2,N)), urhs, erhs])        # TODO: LINE CHANGE

        # Lax-Friedrichs
        sys = .5 * (sysL + sysR) - .5 * lmbd * (sysfluxR - sysfluxL) # + dt * sysrhs      # TODO: LINE CHANGE

        #
        # sys = sysL - lmbd * (sysflux - sysfluxL) # + dt * sysrhs      # TODO: LINE CHANGE

        # Set Variables From tt -> tt + 1
        n, u, e = sys[0], sys[1], sys[2]

        # Store Data
        if tt % (T / snaps) == 0:
            ntot[snap] = n
            utot[snap] = u
            etot[snap] = e
            phitot[snap] = phi
            Qtot[snap] = Q
            snap += 1

        # Track Time
        if tt == int(T / 10):
            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')
            print('Approximate total time:', 10 * elapsed_time, 'seconds')

    # Visualization
    nsnap = np.swapaxes(ntot, 0, 1)
    usnap = np.swapaxes(utot, 0, 1)
    esnap = np.swapaxes(etot, 0, 1)

    syssnapname = [["Density, NC", "Density, C"], ["Velocity, NC", "Velocity, C"], ["Energy, NC", "Energy, C"]]
    syssnapnameabrv = [["n", "nc"],["u", "uc"],["e", "ec"]]
    syssnaptot = np.array([nsnap, usnap, esnap])

    nfluxsnap = np.swapaxes(nfluxtot, 0, 1)
    ufluxsnap = np.swapaxes(ufluxtot, 0, 1)
    efluxsnap = np.swapaxes(efluxtot, 0, 1)

    syssnapfluxname = [["Density Flux, NC", "Density Flux, C"], ["Velocity Flux, NC", "Velocity Flux, C"], ["Energy Flux, NC", "Energy Flux, C"]]
    syssnapfluxnameabrv = [["nflux", "ncflux"],["uflux", "ucflux"],["eflux", "ecflux"]]
    syssnapfluxtot = np.array([nfluxsnap, ufluxsnap, efluxsnap])

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx,yy = np.meshgrid(x, y, sparse=False, indexing='xy')
    for ii in range(len(syssnaptot)):
        for jj in range(len(syssnaptot[ii])):
            plt.figure()
            clr = plt.contourf(xx, yy, syssnaptot[ii,jj])
            plt.colorbar()
            plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))


    for ii in range(len(syssnapfluxtot)):
        for jj in range(len(syssnapfluxtot[ii])):
            plt.figure()
            clr = plt.contourf(xx, yy, syssnapfluxtot[ii, jj])
            plt.colorbar()
            plt.title(syssnapfluxname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))


    # # Plotting
    # for ii in range(len(syssnaptot)):
    #     for jj in range(len(syssnaptot[ii])):
    #         plt.figure()
    #         for kk in range(len(syssnaptot[ii,jj])):
    #             plt.plot(x, syssnaptot[ii,jj,kk], label= syssnapnameabrv[ii][jj] + " @ T = " + str(kk * dt * T / snaps))
    #         plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #         plt.legend()

    # Plotting
    for ii in range(len(syssnapfluxtot)):
        for jj in range(len(syssnapfluxtot[ii])):
            plt.figure()
            for kk in range(len(syssnapfluxtot[ii, jj])):
                plt.plot(x, syssnapfluxtot[ii, jj, kk],
                         label=syssnapfluxnameabrv[ii][jj] + " @ T = " + str(kk * dt * T / snaps))
            plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
            plt.legend()

    # # Difference Plots
    # plt.figure()
    # clr = plt.contourf(xx, yy, nsnap[0] - nsnap[1])
    # plt.colorbar()
    # plt.title("Difference: n-nc")

    plt.show()

