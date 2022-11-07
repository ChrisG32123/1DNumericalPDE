# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt
import time


def solve():
    # Parameters
    N = int(1e2)  # Grid Points
    T = int(5e3)  # Time Steps
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    lmbd = dt / dx
    print(lmbd)
    t = dt*T

    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = 50      # int(input("Number of Snapshots "))    # Number of Snapshots
    Gamma_0 = 1     # float(input("Value of Gamma "))       # Coulomb Coupling Parameter
    kappa_0 = 1     # float(input("Value of kappa "))       # screening something
    snap = 0                                                # Current snapshot in time

    # Memory Allocation
    n, ntot, nfluxtot= np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    u, utot, ufluxtot, ucorrtot, urhstot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    e, etot, efluxtot, ecorrtot, erhstot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    Q, Qtot, Qfluxtot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    phitot, phimtx = np.zeros((snaps + 1, 2, N)), np.zeros((2, N))
    u, utot, e, etot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((2, N)), np.zeros((snaps + 1, 2, N))

    godunov = np.zeros((3, 2, N))

    # Spatial Derivative
    def derivative(array2D):
        return (array2D[:] - np.roll(array2D[:], 1, axis=-1)) / dx

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

    ICfreq = 2 * np.pi / L       # Enforce Periodicity of Perturbed IC
    # ntot[snap,:] = 1 / 2 * n_0 * np.ones(N)           # n_0 * np.ones(N)
    # ntot[snap, :, int(N/2): int(3*N/2)] = 3 / 2 * n_0
    ntot[snap,:] = n_0 * np.ones(N) + .01 * np.sin(ICfreq * (x-L/2))
    utot[snap,:] = np.ones(N) + .1 * np.sin(2*ICfreq * (x-L/2))
    etot[snap,:] = np.ones(N) + .1 * np.sin(3*ICfreq * (x-L/2))

    Qtot[snap,:] = -derivative(etot[snap,:])
    phitot[snap,:] = solvephi(ntot[snap,:])

    # Define Variables at T = t0
    n, u, e = ntot[snap], utot[snap], etot[snap]  # Intrinsic Variables
    Q = - derivative(e)  # Apply Closure
    phi = solvephi(n)  # Solve Phi
    nflux = derivative(n * u)
    uflux = derivative(e + u * u / 2)
    eflux = derivative(Q)

    nfluxtot[snap, :] = derivative(n*u)
    ufluxtot[snap, :] = derivative(e + u*u/2)
    efluxtot[snap, :] = derivative(Q)

    sys = np.array([n, u, e])

    # print(n[0,0], n[0,-1], u[0,0], u[0,-1], e[0,0], e[0,-1])
    # print(nflux[0,0], nflux[0,-1], uflux[0,0], uflux[0,-1], eflux[0,0], eflux[0,-1])
    #
    # plt.figure()
    # plt.plot(x, ICfreq*np.cos(ICfreq*x), label="cos")
    # plt.plot(x,derivative(np.sin(ICfreq*x)), label = "sin")
    # plt.legend()
    # plt.show(block=False)

    st = time.time()
    for tt in range(1, T + 1):
        if tt == 3:
            print(tt, n[0,0])

        # Unvectorize From Solve
        n, u, e = sys[0], sys[1], sys[2]  # Intensive Variables

        if tt == 3:
            print(tt, n[0,0])

        Q = - derivative(e)  # Apply Closure
        phi = solvephi(n)  # Solve Phi

        # Define Flux Functions
        nflux = derivative(n * u)
        uflux = derivative(e + u * u / 2)
        eflux = derivative(Q)

        # Solve Correlation
        ucorr = np.zeros((2,N))  # np.array([np.zeros(N), fft_correlations(k,n[1], Gamma_0, kappa_0)])    # TODO: Add Correlations
        ecorr = np.zeros((2,N))           # TODO: Add Correlations

        # Right Hand Side
        urhs = ucorr/n - derivative(np.log(n))*(e+u*u) - derivative(phi)      # TODO: LINE CHANGE
        erhs = ecorr - derivative(u)*e      # TODO: LINE CHANGE

        if tt == 3:
            print(tt, n[0,0])

        # Vectorize for Solve
        sys, sysflux = np.array([n, u, e]), np.array([nflux, uflux, eflux])
        sysL, sysR = np.roll(sys[:, :, :], 1, axis=-1), np.roll(sys[:, :, :], -1, axis=-1)
        sysfluxL, sysfluxR = np.roll(sysflux[:, :, :], 1, axis=-1), np.roll(sysflux[:, :, :], -1, axis=-1)
        sysrhs = np.array([np.zeros((2,N)), urhs, erhs])        # TODO: LINE CHANGE

        # Godunov Method
        godunov = np.maximum(sysflux, sysfluxR)
        godunov = np.where(sysflux < sysfluxR, np.minimum(sysflux, sysfluxR), godunov)
        godunov = np.where(np.greater(0, sysflux) & np.greater(sysfluxR, 0), 0, godunov)

        # Godunov
        # sys = sys - lmbd * (godunov - np.roll(godunov, 1, axis = -1)) + dt * sysrhs

        sys = sys - lmbd * (np.roll(godunov, 1, axis = 1) - godunov) + dt * sysrhs

        if tt == 3:
            print(tt, n[0,0])

        # Check Snaps & Store Data
        if tt % (T / snaps) == 0:
            snap += 1
            nfluxtot[snap], ufluxtot[snap], efluxtot[snap] = nflux, uflux, eflux
            ntot[snap], utot[snap], etot[snap] = n, u, e

        # Track Time
        if tt == 100 :
            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')
            print('Approximate total time:', elapsed_time * T / 100, 'seconds')

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

    # Plotting
    for ii in range(len(syssnaptot)):
        for jj in range(len(syssnaptot[ii])):
            plt.figure()
            for kk in range(len(syssnaptot[ii,jj])):
                plt.plot(x, syssnaptot[ii,jj,kk], label=syssnapnameabrv[ii][jj] + " @ T = " + str(round(kk * dt * T / snaps,2)))
            plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
            plt.legend()

    for ii in range(len(syssnapfluxtot)):
        for jj in range(len(syssnapfluxtot[ii])):
            plt.figure()
            for kk in range(len(syssnapfluxtot[ii, jj])):
                plt.plot(x, syssnapfluxtot[ii, jj, kk], label=syssnapfluxnameabrv[ii][jj] + " @ T = " + str(round(kk * dt * T / snaps,2)))
            plt.title(syssnapfluxname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
            plt.legend()

    # # Difference Plots
    # plt.figure()
    # clr = plt.contourf(xx, yy, nsnap[0] - nsnap[1])
    # plt.colorbar()
    # plt.title("Difference: n-nc")

    # ImShow        # TODO: Swap Rows, Normalize Axises
    # y = np.linspace(0, t, snaps + 1)
    # xx,yy = np.meshgrid(x, y, sparse=False, indexing='xy')
    # for ii in range(len(syssnaptot)):
    #     for jj in range(len(syssnaptot[ii])):
    #         plt.figure()
    #         clr = plt.imshow(syssnaptot[ii,jj])
    #         plt.colorbar()
    #         plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    # for ii in range(len(syssnapfluxtot)):
    #     for jj in range(len(syssnapfluxtot[ii])):
    #         plt.figure()
    #         clr = plt.imshow(syssnapfluxtot[ii, jj])
    #         plt.colorbar()
    #         plt.title(syssnapfluxname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    # Color Map
    # y = np.linspace(0, t, snaps + 1)
    # xx,yy = np.meshgrid(x, y, sparse=False, indexing='xy')
    # for ii in range(len(syssnaptot)):
    #     for jj in range(len(syssnaptot[ii])):
    #         plt.figure()
    #         clr = plt.contourf(xx, yy, syssnaptot[ii,jj])
    #         plt.colorbar()
    #         plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    #
    # for ii in range(len(syssnapfluxtot)):
    #     for jj in range(len(syssnapfluxtot[ii])):
    #         plt.figure()
    #         clr = plt.contourf(xx, yy, syssnapfluxtot[ii, jj])
    #         plt.colorbar()
    #         plt.title(syssnapfluxname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.show()