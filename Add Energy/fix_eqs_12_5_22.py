import numpy as np
import matplotlib.pyplot as plt
# import scipy.fftpack as spfft
import time


def solve():
    start = time.time()

    # Parameters
    N = int(5e3)  # Grid Points
    T = int(2e1)  # Time Steps
    L = 100  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    dx = x[2] - x[1]  # Grid Size
    dt = 2e-3  # Time Step Size
    lmbd = dt / dx
    t = dt * T

    n_0 = 3 / (4 * np.pi)
    Gamma_0 = 1
    kappa_0 = 1
    therm_cond = .01

    snaps = 2000
    snap = 0

    def last_axis(array, axis=0):
        shape = list(array.shape)
        shape[axis] = 1
        return np.take(array, -1, axis=axis).reshape(tuple(shape))

    def cur(array):
        return np.roll(array, 0, axis=-1)

    def l(array):
        return np.roll(array, 1, axis=-1)

    def r(array):
        return np.roll(array, 1, axis=1)

    def derivative(array):
        return (array - l(array)) / dx

    # TODO: Optimize Thomas Algorithm with Numpy
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

    def checknan(array, name):
        if np.isnan(array.any()):
            print("Nan value at " + name + " at tt = " + str(tt))
            return True
        return False

    # Memory Allocation
    n, ntot, nint = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2))
    v, vtot, vint = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2))
    e, etot, eint = np.zeros((2, N)), np.zeros((snaps + 1, 2, N)), np.zeros((snaps + 1, 2))
    phi, phimtx, phitot = np.zeros((2, N)), np.zeros((2, N)), np.zeros((snaps + 1, 2, N))

    nflux, nfluxtot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N))
    vflux, vfluxtot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N))
    eflux, efluxtot = np.zeros((2, N)), np.zeros((snaps + 1, 2, N))

    # Initial Conditions
    IC_freq = 2 * np.pi / L
    # n[:] = n_0 * np.ones(N) + .01 * np.exp(-(x - L / 2) ** 2) - .01 * np.sqrt(np.pi)
    # v[:] = np.ones(N) + .25 * np.sin(3*IC_freq * x)
    # e[:] = np.ones(N) + .25 * np.sin(3*IC_freq * x)

    n[:] = n_0 * np.ones(N) + .1 * np.random.random(N)
    v[:] = np.ones(N) + .1 * np.random.random(N)
    e[:] = np.ones(N) + .1 * np.random.random(N)

    nflux = n * v
    vflux = e + .5*v**2
    eflux = therm_cond*e

    phi = solvephi(n)

    # Store at T = T0
    ntot[snap] = n
    vtot[snap] = v
    etot[snap] = e

    nfluxtot[snap] = nflux
    vfluxtot[snap] = vflux
    efluxtot[snap] = eflux

    phitot[snap] = phi

    nint[snap] = np.array([np.trapz(n[0], x), np.trapz(n[1], x)])
    vint[snap] = np.array([np.trapz(v[0], x), np.trapz(v[1], x)])
    eint[snap] = np.array([np.trapz(e[0], x), np.trapz(e[1], x)])

    st = time.time()
    for tt in range(1, T + 1):

        # Flux
        nflux = n * v
        vflux = e + .5*v**2
        eflux = therm_cond*e

        # Solve Correlation
        vcorr = np.zeros((2, N))  # np.array([np.zeros(N), fft_correlations(k,n[1], Gamma_0, kappa_0)])    # TODO: Add Correlations
        ecorr = np.zeros((2, N))  # TODO: Add Correlations

        # Right Hand Side
        phi = solvephi(n)

        vrhs = vcorr - derivative(phi) - derivative(np.log(n))*e  # TODO: LINE CHANGE
        erhs = ecorr + 4*derivative(v)*e - derivative(np.log(n))*(therm_cond*derivative(e)) - v*derivative(e)

        # Update
        n = n - lmbd * (nflux - l(nflux))
        v = v - lmbd * (vflux - l(vflux)) + dt * vrhs
        e = e - lmbd / dx * (r(eflux) - 2 * eflux + l(eflux)) + dt * erhs

        # Check Nans
        nnan, vnan, enan, nfluxnan, vfluxnan, efluxnan = checknan(n, "n"), checknan(v, "v"), checknan(e, "e"), checknan(nflux, "nflux"), checknan(vflux, "vflux"), checknan(eflux, "eflux")
        if nnan or vnan or enan or nfluxnan or vfluxnan or efluxnan:
            exit()

        # Take Snapshot
        if tt % (T / snaps) == 0:
            snap += 1
            ntot[snap], vtot[snap], etot[snap] = n, v, e
            nfluxtot[snap], vfluxtot[snap], efluxtot[snap] = nflux, vflux, eflux

            nint[snap] = np.array([np.trapz(n[0], x), np.trapz(n[1], x)])
            vint[snap] = np.array([np.trapz(v[0], x), np.trapz(v[1], x)])
            eint[snap] = np.array([np.trapz(e[0], x), np.trapz(e[1], x)])

            phitot[snap] = phi

        # Keep Progress
        if tt % (T/10) == 0:
            print(str(np.round(100*tt/T)) + "% Done")

        # Track Time
        if tt == 100:
            et = time.time()
            elapsed_time = et - st
            elapsed_time = 1.1 * elapsed_time
            print('Approximate total time:', elapsed_time * T / 100, 'seconds')

    nsnap = np.swapaxes(ntot, 0, 1)
    vsnap = np.swapaxes(vtot, 0, 1)
    esnap = np.swapaxes(etot, 0, 1)
    syssnaptot = np.array([nsnap, vsnap, esnap])

    phisnap = np.swapaxes(phitot, 0, 1)

    nfluxsnap = np.swapaxes(nfluxtot, 0, 1)
    vfluxsnap = np.swapaxes(vfluxtot, 0, 1)
    efluxsnap = np.swapaxes(efluxtot, 0, 1)
    syssnapfluxtot = np.array([nfluxsnap, vfluxsnap, efluxsnap])

    nintsnap = np.swapaxes(nint, 0, 1)
    vintsnap = np.swapaxes(vint, 0, 1)
    eintsnap = np.swapaxes(eint, 0, 1)
    syssnapint = np.array([nintsnap, vintsnap, eintsnap])

    syssnapname = [["Density, NC", "Density, C"], ["Velocity, NC", "Velocity, C"], ["Energy, NC", "Energy, C"]]
    syssnapnameabrv = [["n", "nc"], ["u", "uc"], ["e", "ec"]]

    syssnapfluxname = [["Density Flux, NC", "Density Flux, C"], ["Velocity Flux, NC", "Velocity Flux, C"],
                       ["Energy Flux, NC", "Energy Flux, C"]]
    syssnapfluxnameabrv = [["nflux", "ncflux"], ["uflux", "ucflux"], ["eflux", "ecflux"]]

    syssnapintname = [["Density Integral, NC", "Density Integral, C"],
                      ["Velocity Integral, NC", "Velocity Integral, C"], ["Energy Integral, NC", "Energy Integral, C"]]
    syssnapintnameabrv = [["nint", "ncint"], ["uint", "ucint"], ["eint", "ecint"]]

    # Plotting
    for ii in range(len(syssnaptot)):
        plt.figure()
        for kk in range(len(syssnaptot[ii, 0])):
            plt.plot(x, syssnaptot[ii, 0, kk],
                     label=syssnapnameabrv[ii][0] + " @ T = " + str(round(kk * dt * T / snaps, 2)))
        plt.title(syssnapname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()

    for ii in range(len(syssnapfluxtot)):
        plt.figure()
        for kk in range(len(syssnapfluxtot[ii, 0])):
            plt.plot(x, syssnapfluxtot[ii, 0, kk],
                     label=syssnapfluxnameabrv[ii][0] + " @ T = " + str(round(kk * dt * T / snaps, 2)))
        plt.title(syssnapfluxname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()

    for jj in range(len(phisnap)):
        plt.figure()
        for kk in range(len(phisnap[jj])):
            plt.plot(x, phisnap[jj, kk], label="phi" + " @ T = " + str(round(kk * dt * T / snaps, 2)))
        plt.title("phi" + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()

    # Plot Integrals, Ensure Conservation
    for ii in range(len(syssnapint)):
        plt.figure()
        plt.plot(np.arange(snaps + 1), syssnapint[ii,0], label=syssnapintnameabrv[ii][0] + " Integral")
        plt.title(syssnapintname[ii][0] + " Integral: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()

    # ImShow
    # y = np.linspace(0, t, snaps + 1)
    # xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')
    #
    # for ii in range(len(syssnaptot)):
    #     plt.figure()
    #     clr = plt.imshow(syssnaptot[ii, 0], aspect='auto', origin='lower', extent=(0,L,0,t))
    #     plt.colorbar()
    #     plt.title(syssnapname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    # for ii in range(len(syssnapfluxtot)):
    #     plt.figure()
    #     clr = plt.imshow(syssnapfluxtot[ii, 0], aspect='auto', origin='lower', extent=(0,L,0,t))
    #     plt.colorbar()
    #     plt.title(syssnapfluxname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.figure()
    clr = plt.imshow(phisnap[0], aspect='auto', origin='lower', extent=(0,L,0,t))
    plt.colorbar()
    plt.title("Phi, NC: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    # FFT
    for ii in range(len(nsnap)):
        n_fft = nsnap[ii] - np.mean(nsnap[ii])
        n_fft = np.swapaxes(n_fft,0,1)

        n_fft_flip = np.flip(n_fft,axis=1)
        n_fft = np.hstack((n_fft, n_fft_flip))

        fft = np.fft.fft2(n_fft, axes=(0,1))
        fft = np.fft.fftshift(fft, axes=(0,1))
        fft = np.transpose(fft)

        # Reflect
        # TODO: If fft axis i length odd/even -> Add 1 in index -> ffti_1st_half = fft[:int(fftlength0 / 2 + 1)]
        fftlength0, fftlength1 = fft.shape
        fft1_1st_half, fft1_2nd_half = fft[:int(fftlength0 / 2)], fft[int(fftlength0 / 2):]
        reflect1 = (np.flip(fft1_1st_half, axis=0) + fft1_2nd_half) / 2             # Reflect Bottom -> Up
        reflect1_1st_half, reflect1_2nd_half = reflect1[:, :int(fftlength1 / 2)], reflect1[:, int(fftlength1 / 2):]
        fft_avg = (np.flip(reflect1_1st_half, axis=1) + reflect1_2nd_half) / 2      # Reflect Left -> Right

        # Plot FFT
        plt.figure()
        plt.imshow(np.abs(fft_avg), aspect='auto', origin='lower', extent=(-0.5, snaps-0.5, -0.5, N-0.5))
        plt.colorbar()
        plt.xlabel("Spatial Frequency (k)")
        plt.ylabel("Dispersion (omega)")
        plt.title("FFT: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    # for ii in range(len(nsnap)):
    #     fft = np.fft.ifftshift(nsnap[ii] - np.mean(nsnap[ii]))
    #     fft = np.fft.fft2(fft)
    #     print(np.abs(np.fft.fftshift(fft))-np.real(np.fft.fftshift(fft)))
    #     fft = np.abs(np.fft.fftshift(fft))  # TODO: abs v. real
    #
    #     plt.figure()
    #     clr = plt.imshow(fft, aspect='auto', origin='lower')
    #     plt.colorbar()
    #     plt.title("FFT: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

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
    print("Total Time: ", end - start, " seconds")
    # print("Off By:", np.round(100*(end - start - (elapsed_time * T / 100))/(end - start), 2), " %")

    plt.show()
