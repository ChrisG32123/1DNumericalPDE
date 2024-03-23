import numpy as np
import matplotlib.pyplot as plt
import time

def solve():
    start = time.time()

    # Parameters
    N = int(1e3)  # Grid Points
    T = int(1e5)  # Time Steps
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    lmbd = dt / dx
    t = dt * T

    n_0 = 3 / (4*np.pi)
    Gamma_0 = 1
    kappa_0 = 1

    snaps = 10
    snap = 0

    def last_axis(array, axis=0):
        shape = list(array.shape)
        shape[axis] = 1
        return np.take(array, -1, axis=axis).reshape(tuple(shape))

    def cur(array):
        return np.roll(array, 0, axis = -1)

    def l(array):
        return np.roll(array, 1, axis = -1)

    def r(array):
        return np.roll(array, 1, axis = 1)

    def derivative(array):
        return (array - l(array))/dx

    def upwind(flux):
        return flux

    def godunov(flux, u):
        # TODO: FROM MATLAB
        # fR = godunov(flux_i, flux_{i+1})
        # fL = l(fR)
        #
        # def godunov(uL,uR):
        #    fL = flux(uL)
        #    fR = flux(uR)
        #    f(uL < uR) = min(fL(uL < uR), fR(uL < uR))
        #    f((uL < 0) & ( uR > 0)) = 0

        fluxL = l(flux)
        uL = l(u)
        godunov = np.maximum(flux, fluxL)
        for ii in range(len(godunov)):
            for jj in range(len(godunov[ii])):
                if uL[ii,jj] < u[ii,jj]:
                    godunov[ii,jj] = np.minimum(flux[ii,jj], fluxL[ii,jj])
                elif uL[ii,jj] < 0 and u[ii,jj] > 0:
                    godunov[ii, jj] = 0

        # godunov = np.where(uL < u, np.minimum(flux, fluxL), godunov)
        # godunov = np.where(np.greater(0, fluxL) & np.greater(flux, 0), 0, godunov)

        return godunov

    # Memory Allocation
    n, ntot, nint = np.zeros((2,N)), np.zeros((snaps+1,2,N)), np.zeros((snaps+1,2))
    v, vtot, vint = np.zeros((2,N)), np.zeros((snaps+1,2,N)), np.zeros((snaps+1,2))

    nflux, nfluxtot = np.zeros((2,N)), np.zeros((snaps+1,2,N))
    vflux, vfluxtot = np.zeros((2,N)), np.zeros((snaps+1,2,N))

    ngodunov, ngodunovtot = np.zeros((2,N)), np.zeros((snaps+1,2,N))
    vgodunov, vgodunovtot = np.zeros((2,N)), np.zeros((snaps+1,2,N))

    # Initial Conditions
    n[:] = n_0 * np.ones(N) + .02 * np.exp(-(x - L / 2) ** 2) - .02*np.sqrt(np.pi)
    v[:] = np.ones(N)

    nflux = n * v
    vflux = np.log(n) + v**2

    ngodunov = godunov(nflux, n)
    vgodunov = godunov(vflux, v)

    # Store at T = T0
    ntot[snap] = n
    vtot[snap] = v

    nfluxtot[snap] = nflux
    vfluxtot[snap] = vflux

    ngodunovtot[snap] = ngodunov
    vgodunovtot[snap] = vgodunov

    nint[snap] = np.array([np.trapz(n[0], x), np.trapz(n[1], x)])
    vint[snap] = np.array([np.trapz(v[0], x), np.trapz(v[1], x)])

    for tt in range(1,T+1):

        nflux = n * v
        ngodunov = godunov(nflux, n)

        vflux = np.log(n) + v**2

        # n = n - lmbd * (ngodunov - l(ngodunov))
        n = n - lmbd * ((nflux)-l(nflux))
        v = v - lmbd * ((vflux) - l(vflux))

        if tt % (T / snaps) == 0:
            snap += 1
            ntot[snap], vtot[snap] = n, v
            nfluxtot[snap], vfluxtot[snap] = nflux, vflux

            nint[snap] = np.array([np.trapz(n[0], x), np.trapz(n[1], x)])
            vint[snap] = np.array([np.trapz(v[0], x), np.trapz(v[1], x)])

            ngodunovtot[snap], vgodunovtot[snap] = ngodunov, vgodunov

    # print("ntot", ntot)
    # print("cur(n)", n)
    # print("cur(ntot)", cur(ntot))
    # print("l(n)", l(ntot))
    # print("derivative(n)", derivative(ntot))
    # print(dx)

    # for ii in range(snaps):
    #     print(nfluxtot[ii, 0, -5:-1], nfluxtot[ii, 0, 0:5])
    #     print(nfluxtot[ii, 0, -5:-1] - l(nfluxtot[ii, 0, -5:-1]), nfluxtot[ii, 0, 0:5]- l(nfluxtot[ii, 0, 0:5]))
    for ii in range(snaps):
        print(ngodunovtot[ii, 0,-5:-1], ngodunovtot[ii, 0,0:5])
        print(ngodunovtot[ii, 0,-5:-1] - l(ngodunovtot[ii, 0,-5:-1]), ngodunovtot[ii, 0,0:5]- l(ngodunovtot[ii, 0,0:5]))


    nsnap = np.swapaxes(ntot, 0, 1)
    vsnap = np.swapaxes(vtot, 0, 1)
    syssnaptot = np.array([nsnap, vsnap])

    nfluxsnap = np.swapaxes(ngodunovtot, 0, 1)
    ufluxsnap = np.swapaxes(vgodunovtot, 0, 1)
    syssnapfluxtot = np.array([nfluxsnap, ufluxsnap])

    nintsnap = np.swapaxes(nint, 0, 1)
    vintsnap = np.swapaxes(nint, 0, 1)

    syssnapname = [["Density, NC", "Density, C"], ["Velocity, NC", "Velocity, C"]]
    syssnapnameabrv = [["n", "nc"], ["u", "uc"]]
    syssnapfluxname = [["Density Flux, NC", "Density Flux, C"], ["Velocity Flux, NC", "Velocity Flux, C"]]
    syssnapfluxnameabrv = [["nflux", "ncflux"], ["uflux", "ucflux"]]

    # Plotting
    for ii in range(len(syssnaptot)):
        plt.figure()
        for kk in range(len(syssnaptot[ii, 0])):
            plt.plot(x, syssnaptot[ii, 0, kk], label=syssnapnameabrv[ii][0] + " @ T = " + str(round(kk * dt * T / snaps, 2)))
        plt.title(syssnapname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()

    for ii in range(len(syssnapfluxtot)):
        plt.figure()
        for kk in range(len(syssnapfluxtot[ii, 0])):
            plt.plot(x, syssnapfluxtot[ii, 0, kk], label=syssnapfluxnameabrv[ii][0] + " @ T = " + str(round(kk * dt * T / snaps, 2)))
        plt.title(syssnapfluxname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()

    # for jj in range(len(phisnap)):
    #     plt.figure()
    #     for kk in range(len(phisnap[jj])):
    #         plt.plot(X, phisnap[jj, kk],
    #                  label="phi" + " @ T = " + str(round(kk * dt * T / snaps, 2)))
    #     plt.title("phi" + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #     plt.legend()

    # Plot Integrals, Ensure Conservation
    plt.figure()
    plt.plot(np.arange(snaps + 1), nintsnap[0], label="Integral" + " @ T = " + str(round(kk * dt * T / snaps, 2)))
    plt.title("Integral" + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    plt.legend()

    # # Difference Plots
    # plt.figure()
    # clr = plt.contourf(xx, yy, nsnap[0] - nsnap[1])
    # plt.colorbar()
    # plt.title("Difference: n-nc")

    # ImShow        # TODO: Swap Rows, Normalize Axises
    y = np.linspace(0, t, snaps + 1)
    xx,yy = np.meshgrid(x, y, sparse=False, indexing='xy')
    for ii in range(len(syssnaptot)):
        plt.figure()
        clr = plt.imshow(np.flip(syssnaptot[ii,0],-2), aspect='auto')
        plt.colorbar()
        plt.title(syssnapname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    for ii in range(len(syssnapfluxtot)):
        plt.figure()
        clr = plt.imshow(np.flip(syssnapfluxtot[ii, 0],-2), aspect='auto')
        plt.colorbar()
        plt.title(syssnapfluxname[ii][0] + " Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

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

    # for ii in range(len(nsnap)):
    #     fft = np.fft.ifftshift(nsnap[ii] - np.mean(nsnap[ii]))
    #     fft = np.fft.fft2(fft)
    #     fft = np.abs(np.fft.fftshift(fft))  # TODO: abs v. real
    #
    #     plt.figure()
    #     clr = plt.imshow(np.flip(fft, -2), aspect='auto')  # TODO: Change Dimensions of Imshow
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
    print("Total Time: ", end - start)

    plt.show()