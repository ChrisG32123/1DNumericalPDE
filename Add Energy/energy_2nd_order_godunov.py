# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt
import time


def solve():
    # Parameters
    N = int(1e3)  # Grid Points
    T = int(1e2)  # Time Steps
    L = 100  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-4  # Time Step Size
    lmbd = dt / dx
    print(lmbd)
    t = dt*T

    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = 10      # int(input("Number of Snapshots "))    # Number of Snapshots
    Gamma_0 = 1     # float(input("Value of Gamma "))       # Coulomb Coupling Parameter
    kappa_0 = 1     # float(input("Value of kappa "))       # screening something
    snap = 0                                                # Current snapshot in time

    # Memory Allocation
    n, ntot, nfluxtot= np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    nu, nutot, nufluxtot, nucorrtot, nurhstot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    ne, netot, nefluxtot, necorrtot, nerhstot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    nQ, nQtot, nQfluxtot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    phitot, phimtx = np.zeros((snaps + 1, 2, N)), np.zeros((2, N))
    u, utot, e, etot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((2, N)), np.zeros((snaps + 1, 2, N))

    godunov = np.zeros((3, 2, N))

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

    ICfreq = 2 * np.pi / L       # Enforce Periodicity of Perturbed IC
    # ntot[cursnap,:] = 1 / 2 * mean_n * np.ones(nx)           # mean_n * np.ones(nx)
    # ntot[cursnap, :, int(nx/2): int(3*nx/2)] = 3 / 2 * mean_n
    ntot[snap,:] = n_0 * np.ones(N) + .01 * np.sin(ICfreq * x)
    nutot[snap,:] = ntot[snap] * (.5 * np.ones(N) + .1 * np.sin(2*ICfreq * x))
    netot[snap,:] = ntot[snap] * (.5 * np.ones(N) + .1 * np.sin(3*ICfreq * x))

    nQtot[snap,:] = -derivative(netot[snap,:])
    phitot[snap,:] = solvephi(ntot[snap,:])
    utot[snap,:] = nutot[snap,:] / ntot[snap]
    etot[snap,:] = netot[snap] / ntot[snap]

    # Define Variables at T = t0
    n, nu, ne = ntot[snap], nutot[snap], netot[snap]    # Extrinsic Variables
    u, e = utot[snap], etot[snap]                       # Intrinsic Variables

    st = time.time()
    for tt in range(1, T + 1):

        # Define Flux Functions
        nflux = derivative(nu)
        nuflux = derivative(ne)
        neflux = derivative(nQ + nu*ne/n)

        nQ = -n * derivative(e)  # Apply Closure
        phi = solvephi(n)  # Solve Phi

        # if tt < 5:
        #     print(n[0]*u[0])
        #     print(nflux[0])

        # Solve Correlation
        ucorr = np.zeros((2,N))  # np.array([np.zeros(nx), fft_correlations(k,n[1], Gamma_0, kappa_0)])    # TODO: Add Correlations
        ecorr = np.zeros((2,N))           # TODO: Add Correlations

        # Right Hand Side
        urhs = ucorr - n*derivative(phi)      # TODO: LINE CHANGE
        erhs = ecorr - derivative(u)*e      # TODO: LINE CHANGE

        # Vectorize for Solve
        sys, sysflux = np.array([n, nu, ne]), np.array([nflux, nuflux, neflux])
        sysL, sysR = np.roll(sys[:, :, :], 1, axis=-1), np.roll(sys[:, :, :], -1, axis=-1)
        sysfluxL, sysfluxR = np.roll(sysflux[:, :, :], 1, axis=-1), np.roll(sysflux[:, :, :], -1, axis=-1)
        sysrhs = np.array([np.zeros((2,N)), urhs, erhs])        # TODO: LINE CHANGE

        # Godunov Method
        lin_rcnstrct = (sysR-sysL) / (2 * dx)
        sysminus = sys + lin_rcnstrct * dx / 2
        sysplus = sysR - np.roll(lin_rcnstrct,-1, axis=-1) * dx / 2

        nminus, numinus, neminus = sysminus[0], sysminus[1], sysminus[2]
        nplus, nuplus, neplus = sysplus[0], sysplus[1], sysplus[2]
        nQminus, nQplus = -nminus * derivative(neminus/nminus), -nplus * derivative(neplus/nplus)

        nfluxminus, nfluxplus = derivative(numinus), derivative(nuplus)
        nufluxminus, nufluxplus = derivative(neminus), derivative(neplus)
        nefluxminus, nefluxplus = derivative(nQminus + numinus*neminus/nminus), derivative(nQplus + nuplus*neplus/nplus)

        sysfluxminus = np.array([nfluxminus, nufluxminus, nefluxminus])
        sysfluxplus = np.array([nfluxplus, nufluxplus, nefluxplus])

        godunov = np.maximum(sysfluxminus, sysfluxplus)
        godunov = np.where(sysfluxminus < sysfluxplus, np.minimum(sysfluxminus, sysfluxplus), godunov)
        godunov = np.where(np.greater(0, sysfluxminus) & np.greater(sysfluxplus, 0), 0, godunov)

        # Lax-Friedrichs
        # sys = .5 * (sysL + sysR) - .5 * lmbd * (sysfluxR - sysfluxL) + dt * sysrhs      # TODO: LINE CHANGE

        # Godunov
        sys = sys - lmbd * (godunov - np.roll(godunov, 1, axis = -1)) + dt * sysrhs

        # Unvectorize From Solve
        n, nu, ne = sys[0], sys[1], sys[2]  # Extrinsic Variables
        u, e = nu / n, ne / n  # Intrinsic Variables

        # Check Snaps & Store Data
        if tt % (T / snaps) == 0:
            snap += 1
            nfluxtot[snap], nufluxtot[snap], nefluxtot[snap] = nflux, nuflux, neflux
            ntot[snap], nutot[snap], netot[snap] = n, nu, ne
            utot[snap], etot[snap] = u, e

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
    nufluxsnap = np.swapaxes(nufluxtot, 0, 1)
    nefluxsnap = np.swapaxes(nefluxtot, 0, 1)

    syssnapfluxname = [["Density Flux, NC", "Density Flux, C"], ["Velocity Flux, NC", "Velocity Flux, C"], ["Energy Flux, NC", "Energy Flux, C"]]
    syssnapfluxnameabrv = [["nflux", "ncflux"],["uflux", "ucflux"],["eflux", "ecflux"]]
    syssnapfluxtot = np.array([nfluxsnap, nufluxsnap, nefluxsnap])

    # Plotting
    for ii in range(len(syssnaptot)):
        for jj in range(len(syssnaptot[ii])):
            plt.figure()
            for kk in range(len(syssnaptot[ii,jj])):
                plt.plot(x, syssnaptot[ii,jj,kk], label= syssnapnameabrv[ii][jj] + " @ T = " + str(kk * dt * T / snaps))
            plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
            plt.legend()

    for ii in range(len(syssnapfluxtot)):
        for jj in range(len(syssnapfluxtot[ii])):
            plt.figure()
            for kk in range(len(syssnapfluxtot[ii, jj])):
                plt.plot(x, syssnapfluxtot[ii, jj, kk],
                         label=syssnapfluxnameabrv[ii][jj] + " @ T = " + str(kk * dt * T / snaps))
            plt.title(syssnapfluxname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
            plt.legend()

    # # Difference Plots
    # plt.figure()
    # clr = plt.contourf(xx, yy, nsnap[0] - nsnap[1])
    # plt.colorbar()
    # plt.title("Difference: n-nc")

    # ImShow        # TODO: Swap Rows, Normalize Axises
    # y = np.linspace(0, t, snaps + 1)
    # xx,yy = np.meshgrid(X, y, sparse=False, indexing='xy')
    # for ii in range(len(syssnaptot)):
    #     for jj in range(len(syssnaptot[ii])):
    #         plt.figure()
    #         clr = plt.imshow(syssnaptot[ii,jj])
    #         plt.colorbar()
    #         plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    #
    # for ii in range(len(syssnapfluxtot)):
    #     for jj in range(len(syssnapfluxtot[ii])):
    #         plt.figure()
    #         clr = plt.imshow(syssnapfluxtot[ii, jj])
    #         plt.colorbar()
    #         plt.title(syssnapfluxname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    # Color Map
    # y = np.linspace(0, t, snaps + 1)
    # xx,yy = np.meshgrid(X, y, sparse=False, indexing='xy')
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