import numpy as np
import matplotlib.pyplot as plt
import time


def solve():
    start = time.time()
    # Parameters
    N = int(5e2)  # Grid Points
    T = int(2e3)  # Time Steps
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    lmbd = dt / dx
    print(lmbd)
    t = dt * T

    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = 20  # int(input("Number of Snapshots "))     # Number of Snapshots
    Gamma_0 = 1  # float(input("Value of Gamma "))       # Coulomb Coupling Parameter
    kappa_0 = 1  # float(input("Value of kappa "))       # Screened Coulomb Parameter
    therm_cond = 1                                       # Thermal Conductivity
    snap = 0  # Current snapshot in time

    # Memory Allocation
    n, ntot, nfluxtot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    u, utot, ufluxtot, ucorrtot, urhstot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    e, etot, efluxtot, ecorrtot, erhstot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    Q, Qtot, Qfluxtot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2, N))
    phitot, phimtx = np.zeros((snaps + 1, 2, N)), np.zeros((2, N))
    u, utot, e, etot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((2, N)), np.zeros((snaps + 1, 2, N))

    nint = np.zeros((snaps + 1, 2))

    godunov = np.zeros((3, 2, N))

    # Retrieve Left Index
    def l(array):
        la = np.roll(array, 1, axis=-1)
        return la

    # Retrieve Right Index
    def r(array):
        ra = np.roll(array, -1, axis=-1)
        return ra

    # Spatial Derivative
    def derivative(array):
        deriv = (array[...,:] - l(array)) / dx
        return deriv

    def solvephi(den):
        phi = np.zeros((2, N))
        # phimtx * phi = b
        # Define b
        b = 3 - 4 * np.pi * dx * dx * den
        b = b - np.mean(b)
        # First sweep
        phimtx[:, 0] = -0.5
        b[:, 0] = -0.5 * b[:, 0]
        for ii in range(1, N - 1):
            phimtx[:, ii] = -1 / (2 + phimtx[:, ii - 1])
            b[:, ii] = (b[:, ii - 1] - b[:, ii]) / (2 + phimtx[:, ii - 1])
        # Second sweep
        phi[:, 0] = b[:, N - 1] - b[:, N - 2]
        for ii in range(1, N - 1):
            phi[:, ii] = (b[:, ii - 1] - phi[:, ii - 1]) / phimtx[:, ii - 1]
        return phi

    def godunov(flux):
        fluxL = l(flux)
        godunov = np.maximum(flux, fluxL)
        godunov = np.where(flux < fluxL, np.minimum(flux, fluxL), godunov)
        godunov = np.where(np.greater(0, flux) & np.greater(fluxL, 0), 0, godunov)
        return godunov

    def fft_correlations(k, nc, Gamma, kappa):
        delta_n = nc - n_0
        fhat = np.fft.fftshift(np.fft.fft(delta_n))
        dcfunc = 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)

        conv = fhat * dcfunc
        conv = np.fft.ifft(np.fft.ifftshift(conv))
        conv = np.real(conv)

        return conv

    # Initial Conditions & Initialization

    ICfreq = 2 * np.pi / L  # Enforce Periodicity of Perturbed IC
    # ntot[cursnap, :] = 1 / 2 * mean_n * np.ones(nx)  # mean_n * np.ones(nx)
    # ntot[cursnap, :, int(nx / 2): int(3 * nx / 2)] = 3 / 2 * mean_n

    # ntot[cursnap,:] = mean_n * np.ones(nx) + .01 * np.sin(5 * ICfreq * (X-Xlngth/2))
    # utot[cursnap, :] = .5 * np.ones(nx) + .01 * np.sin(5 * ICfreq * (X - Xlngth / 2))
    # etot[cursnap, :] = .5 * np.ones(nx) + .01 * np.sin(5 * ICfreq * (X - Xlngth / 2))

    ntot[snap, :] = n_0 * np.ones(N) + .02 * np.exp(-10*(x - L / 2) ** 2) - .02*np.sqrt(np.pi/10)
    utot[snap, :] = np.exp(-(x - L / 2) ** 2) / np.sqrt(np.pi)  # np.zeros(nx)
    etot[snap, :] = np.exp(-(x - L / 2) ** 2) / np.sqrt(np.pi)  # np.zeros(nx)


    # ntot[cursnap, :] = mean_n * np.ones(nx) + .01 * np.random.random()
    # utot[cursnap, :] = .01 * np.random.random()
    # etot[cursnap, :] = .01 * np.random.random()

    Qtot[snap, :] = -derivative(etot[snap, :])
    phitot[snap, :] = solvephi(ntot[snap, :])

    # Define Variables at T = t0
    n, u, e = ntot[snap], utot[snap], etot[snap]  # Intrinsic Variables
    Q = - derivative(e)  # Apply Closure
    phi = solvephi(n)  # Solve Phi
    nflux = derivative(n * u)
    uflux = derivative(e + u * u / 2)
    eflux = derivative(Q)

    nfluxtot[snap, :] = derivative(n * u)
    ufluxtot[snap, :] = derivative(e + u * u / 2)
    efluxtot[snap, :] = derivative(Q)

    nint[snap, 0] = np.trapz(n[0], x)

    # print(n[0,0], n[0,-1], u[0,0], u[0,-1], e[0,0], e[0,-1])
    # print(nflux[0,0], nflux[0,-1], uflux[0,0], uflux[0,-1], eflux[0,0], eflux[0,-1])

    # plt.figure()
    # plt.plot(X, ICfreq*np.cos(ICfreq*X), label="cos")
    # plt.plot(X,derivative(np.sin(ICfreq*X)), label = "sin")
    # plt.legend()
    # plt.show(block=False)

    st = time.time()
    for tt in range(1, T + 1):

        # u = np.ones((2,nx))
        # uflux = np.ones((2,nx))
        #
        # e = np.zeros((2,nx))
        # eflux = np.zeros((2,nx))
        # Q = np.zeros((2,nx))

        Q = - therm_cond * derivative(e)  # Apply Closure
        phi = solvephi(n)  # Solve Phi

        # Define Flux Functions
        nflux = derivative(n * u)
        uflux = derivative(e + u * u / 2)
        eflux = derivative(Q)

        # Solve Correlation
        ucorr = np.zeros((2, N))  # np.array([np.zeros(nx), fft_correlations(k,n[1], Gamma_0, kappa_0)])    # TODO: Add Correlations
        ecorr = np.zeros((2, N))  #       # TODO: Add Correlations

        # Right Hand Side
        nrhs = np.zeros((2, N))
        urhs = ucorr / n - derivative(np.log(n)) * (e + u * u) - derivative(phi)  # TODO: LINE CHANGE
        erhs = ecorr - derivative(u) * e  # TODO: LINE CHANGE

        # Godunov Method
        ngodunov = godunov(n)
        ugodunov = godunov(u)
        egodunov = godunov(e)

        # Solve
        n = n - lmbd * (ngodunov - l(ngodunov)) + dt * nrhs
        u = u - lmbd * (ugodunov - l(ugodunov)) + dt * urhs
        e = e - lmbd * (egodunov - l(egodunov)) + dt * erhs

        # Check Snaps & Store Data
        if tt % (T / snaps) == 0:

            snap += 1
            ntot[snap], utot[snap], etot[snap] = n, u, e
            nfluxtot[snap], ufluxtot[snap], efluxtot[snap] = nflux, uflux, eflux
            phitot[snap] = phi

            nint[snap,0] = np.trapz(n[0], x)
            print(nint[snap,0])

        # Track Time
        if tt == 100:
            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')
            print('Approximate total time:', elapsed_time * T / 100, 'seconds')

    # Visualization
    nsnap = np.swapaxes(ntot, 0, 1)
    usnap = np.swapaxes(utot, 0, 1)
    esnap = np.swapaxes(etot, 0, 1)
    phisnap = np.swapaxes(phitot, 0, 1)

    nintsnap = np.swapaxes(nint, 0, 1)

    syssnapname = [["Density, NC", "Density, C"], ["Velocity, NC", "Velocity, C"], ["Energy, NC", "Energy, C"]]
    syssnapnameabrv = [["n", "nc"], ["u", "uc"], ["e", "ec"]]
    syssnaptot = np.array([nsnap, usnap, esnap])

    nfluxsnap = np.swapaxes(nfluxtot, 0, 1)
    ufluxsnap = np.swapaxes(ufluxtot, 0, 1)
    efluxsnap = np.swapaxes(efluxtot, 0, 1)

    syssnapfluxname = [["Density Flux, NC", "Density Flux, C"], ["Velocity Flux, NC", "Velocity Flux, C"],
                       ["Energy Flux, NC", "Energy Flux, C"]]
    syssnapfluxnameabrv = [["nflux", "ncflux"], ["uflux", "ucflux"], ["eflux", "ecflux"]]
    syssnapfluxtot = np.array([nfluxsnap, ufluxsnap, efluxsnap])

    # Plotting
    for ii in range(len(syssnaptot)):
        for jj in range(len(syssnaptot[ii])):
            plt.figure()
            for kk in range(len(syssnaptot[ii, jj])):
                plt.plot(x, syssnaptot[ii, jj, kk],
                         label=syssnapnameabrv[ii][jj] + " @ T = " + str(round(kk * dt * T / snaps, 2)))
            plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
            plt.legend()

    for ii in range(len(syssnapfluxtot)):
        for jj in range(len(syssnapfluxtot[ii])):
            plt.figure()
            for kk in range(len(syssnapfluxtot[ii, jj])):
                plt.plot(x, syssnapfluxtot[ii, jj, kk],
                         label=syssnapfluxnameabrv[ii][jj] + " @ T = " + str(round(kk * dt * T / snaps, 2)))
            plt.title(syssnapfluxname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
            plt.legend()

    for jj in range(len(phisnap)):
        plt.figure()
        for kk in range(len(phisnap[jj])):
            plt.plot(x, phisnap[jj,kk],
                     label="phi" + " @ T = " + str(round(kk * dt * T / snaps, 2)))
        plt.title("phi" + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()

    plt.figure()
    plt.plot(np.arange(snaps+1), nintsnap[0],label="Integral" + " @ T = " + str(round(kk * dt * T / snaps, 2)))
    plt.title("Integral" + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
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
    #         clr = plt.imshow(np.flip(syssnaptot[ii,jj],-2), aspect='auto')
    #         plt.colorbar()
    #         plt.title(syssnapname[ii][jj] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    # for ii in range(len(syssnapfluxtot)):
    #     for jj in range(len(syssnapfluxtot[ii])):
    #         plt.figure()
    #         clr = plt.imshow(np.flip(syssnapfluxtot[ii, jj],-2), aspect='auto')
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

    for ii in range(len(nsnap)):
        fft = np.fft.ifftshift(nsnap[ii]-np.mean(nsnap[ii]))
        fft = np.fft.fft2(fft)
        fft = np.abs(np.fft.fftshift(fft)) # TODO: abs v. real

        plt.figure()
        clr = plt.imshow(np.flip(fft,-2), aspect='auto')  # TODO: Change Dimensions of Imshow
        plt.colorbar()
        plt.title("FFT: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    # def disp_rel_cmap(ux, ut, u):
    #         fft = calculate_2dft(u[c])
    #         fig = plt.figure(figsize=(15,15))
    #         color_map = plt.contourf(ux, ut, fft)
    #         color_map = plt.imshow(fft, cmap = 'viridis', origin='lower', extent=(X0,Xf,T0,Tf), aspect='auto')
    #         plt.title('Γ = ' + str(Gamma[ii]) + ', κ = ' + str(kappa[jj]))
    #         plt.title('Γ = ' + str(Gamma_0) + ', κ = ' + str(kappa_0))
    #         plt.colorbar()
    #         plt.ylabel("Time")
    #         plt.xlabel("Space")
    #         plt.show(block=False)

    end = time.time()
    print(end - start)

    plt.show()

    if __name__ == '__main__':
        pass