# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# v_t + (.5v^2 + log(n))_x = - gamma_0*phi_x + f_corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt


def subplot(Gamma_0,kappa_0):
# ============== #
# Define Methods #
# ============== #

    def memory_allocation_PDE(u_IC):
        u = np.copy(u_IC)
        uL = np.roll(u, 1)
        uR = np.copy(u)
        flux = np.zeros(N)
        FuL = np.zeros(N)
        FuR = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        snap_u[0] = np.copy(u_IC)
        return u, uL, uR, flux, FuL, FuR, snap_u

    def memory_allocation_RHS():
        u = np.zeros(N)
        uL = np.roll(u, 1)
        uR = np.copy(u)
        matrix = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        rhs = np.zeros(N)
        return u, uL, uR, matrix, snap_u, rhs

    def f_n(n, v):
        flux = n * v
        return flux

    def f_v(n, v):
        flux = .5 * v * v + np.log(n)
        return flux

    def compute_phi(n, phi, A):
        # Define b
        b = 3 - 4 * np.pi * dx * dx * n
        b = b - np.mean(b)
        # First sweep
        A[0] = -0.5
        b[0] = -0.5 * b[0]
        for ii in range(1, N - 1):
            A[ii] = -1 / (2 + A[ii - 1])
            b[ii] = (b[ii - 1] - b[ii]) / (2 + A[ii - 1])
        # Second sweep
        phi[0] = b[N - 1] - b[N - 2]
        for ii in range(1, N - 1):
            phi[ii] = (b[ii - 1] - phi[ii - 1]) / A[ii - 1]
        return phi

    def nonisotropic_correlations(nc, n3, x, x3,f_corr):
        conc = nc / n_0
        Gamma = Gamma_0 * conc ** (1 / 3)
        kappa = kappa_0 * conc ** (1 / 6)

        n3[0:N] = nc[0:N]
        n3[N:2 * N] = nc[0:N]
        n3[2 * N:3 * N] = nc[0:N]
        for jj in range(N):
            # TODO: rho not used, using rho_int instead?
            rho_int = - 2 * np.pi * Gamma[jj] * np.exp(- kappa[jj] * np.abs(x3 - x[jj])) / kappa[jj]
            f_corr[jj] = dx * np.sum(n3 * rho_int)
        return f_corr

    def fft_meanfield(k,nc,Gamma, kappa):
        delta_n = nc - n_0
        def dcf(k, Gamma, kappa):
            return 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)

        dcfunc = dcf(k,Gamma,kappa)
        fhat = np.fft.fftshift(np.fft.fft(delta_n))
        conv = fhat * dcfunc
        conv = np.fft.ifft(np.fft.ifftshift(conv))
        conv = np.real(conv)
        return conv

    def fft_meyerf(k,nc,Gamma, kappa, beta):
        delta_n = nc - n_0

        # f_fft_norm = 1 / dx
        # k_fft_norm = 2 * np.pi / (nx * dx)

        # Parameters
        Nr = int(1e3)
        rmax = 100  # TODO: Change per loop
        r = np.linspace(0, rmax, Nr)

        dcf = np.exp(-beta * r ** 2)
        dcf_fft = np.fft.fftshift(np.fft.fft())
        dcf_fft_ex = (np.pi / beta) ** (3 / 2) * np.exp(- k ** 2 / (4 * beta))

        n_hat = np.fft.fftshift(np.fft.fft(delta_n))
        conv = n_hat * dcf_fft
        conv = np.fft.ifft(np.fft.ifftshift(conv))
        conv = np.real(conv)
        return conv

    def take_snapshot(tt, T, snaps, u, snap_u):
        snap_u[int(tt / (T / snaps))] = u

    def godunov_flux(fL, fR, uL, uR):
        if uL > uR:
            godunov_flux = np.maximum(fL, fR)
        elif uL < uR:
            godunov_flux = np.minimum(fL, fR)
        else:
            godunov_flux = 0.0
        return godunov_flux

    def update_Riemann_values(u):
        uL = np.roll(u, 1)
        uR = np.copy(u)
        return uL, uR

    def solve(correlations, n, nL, nR, v, vL, vR, phi, phiL, phiR, f_corr, rhs, FnL, FnR, FvL, FvR, snap_n, snap_v, snap_phi, Gamma, kappa):
        for tt in range(T + 1):
            # Snapshots
            if tt % (T / snaps) == 0:
                take_snapshot(tt, T, snaps, n, snap_n)
                take_snapshot(tt, T, snaps, v, snap_v)

            # Compute RHS
            phi = compute_phi(n, phi, A)
            if correlations:
                # f_corr = nonisotropic_correlations(n,n3,X,x3,f_corr)
                f_corr = fft_meanfield(k,n,Gamma,kappa)
                f_corrL, f_corrR = update_Riemann_values(f_corr)
                rhs = -(f_corrR - f_corrL)/dx - Gamma_0 * (phiR - phiL) / dx
            else:
                rhs = - Gamma_0 * (phiR - phiL) / dx

            # Compute Fluxes
            for ii in range(N):
                FnL[ii] = f_n(nL[ii], vL[ii])
                FnR[ii] = f_n(nR[ii], vR[ii])
                flux_n[ii] = godunov_flux(FnL[ii], FnR[ii], nL[ii], nR[ii])

                FvL[ii] = f_v(nL[ii], vL[ii])
                FvR[ii] = f_v(nR[ii], vR[ii])
                flux_v[ii] = godunov_flux(FvL[ii], FvR[ii], vL[ii], vR[ii])

            # Solve
            for ii in range(0, N - 1):
                n[ii] = n[ii] - lambda_ * (flux_n[ii + 1] - flux_n[ii])
                v[ii] = v[ii] + dt * rhs[ii] - lambda_ * (flux_v[ii + 1] - flux_v[ii])
            n[N - 1] = n[N - 1] - lambda_ * (flux_n[0] - flux_n[N - 1])
            v[N - 1] = v[N - 1] + dt * rhs[N - 1] - lambda_ * (flux_v[0] - flux_v[N - 1])

            # Update Functions
            nL, nR = update_Riemann_values(n)
            vL, vR = update_Riemann_values(v)
            phiL, phiR = update_Riemann_values(phi)

    def colormap(xx, yy, snap_u):
        plt.figure()
        clr = plt.contourf(xx, yy, snap_u)
        plt.colorbar()
        plt.ylabel("Time")
        plt.xlabel("Space")

        return clr

    def plot(x, snap_u):
        plt.figure()

        plt.plot(x,snap_u[0], label="first")
        plt.plot(x,snap_u[-1], label="last")

        for ii in range(len(snap_u)):
            plt.plot(x, snap_u[ii], label="T = " + str(ii * dt * T / snaps))
        plt.legend()

# ================= #
# Define Parameters #
# ================= #

    # Parameters
    N = int(5e2)  # Grid Points
    T = int(3e3)  # Time Steps
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)
    lambda_ = dt / dx
    t = dt * T
    disp_freq = 3*2 * np.pi / L
    correlations = False

    n_0 = 3 / (4 * np.pi)  #5.04e-16 # Mean Density
    # snaps = int(input("Number of Snapshots "))  # Number of Snapshots
    # Gamma_0 = float(input("Value of Gamma "))  # Coulomb Coupling Parameter
    # kappa_0 = float(input("Value of kappa "))  # screening something
    snaps = 50
    beta = 1

    # Dispersion Relation
    omega = np.sqrt((disp_freq ** 2) + 3 * Gamma_0)
    omega_c = np.sqrt((1+ 3* Gamma_0 / (disp_freq ** 2 + kappa_0 ** 2))*(disp_freq ** 2) + 3 * Gamma_0)
    print("omega: ", omega)
    print("omega_c: ", omega_c)

    # Initial Conditions
    # n_IC = rho_0 * np.ones(nx)
    # n_IC[0:int(nx / 4)] = rho_0 / 2
    # n_IC[int(nx / 4):int(3 * nx / 4)] = 3 * rho_0 / 2
    # n_IC[int(3 * nx / 4):nx] = rho_0 / 2

    # n_IC = rho_0 * np.exp(-(X-Xlngth/2)**2)
    # v_IC = np.zeros(nx)

    n_IC = n_0 * np.ones(N) + .1*np.sin(disp_freq*x)
    v_IC = .1*np.sin(disp_freq*x)

    n, nL, nR, flux_n, FnL, FnR, snap_n = memory_allocation_PDE(n_IC)
    v, vL, vR, flux_v, FvL, FvR, snap_v = memory_allocation_PDE(v_IC)
    phi, phiL, phiR, A, snap_phi, rhs = memory_allocation_RHS()

    nc, ncL, ncR, flux_nc, FncL, FncR, snap_nc = memory_allocation_PDE(n_IC)
    vc, vcL, vcR, flux_vc, FvcL, FvcR, snap_vc = memory_allocation_PDE(v_IC)
    phic, phicL, phicR, Ac, snap_phic, rhsc = memory_allocation_RHS()
    f_corr = np.zeros(N)
    n3 = np.zeros(3 * N)

    # nc, ncL, ncR, flux_nc, snap_nc = memory_allocation()
    # vc, vcL, vcR, flux_vc, snap_vc = memory_allocation()
    # phic, phicL, phicR, Ac, snap_phic = memory_allocation()

# ===== #
# Solve #
# ===== #
    solve(correlations, n, nL, nR, v, vL, vR, phi, phiL, phiR, f_corr, rhs, FnL, FnR, FvL, FvR, snap_n, snap_v, snap_phi, Gamma_0, kappa_0)
    correlations = True
    solve(correlations, nc, ncL, ncR, vc, vcL, vcR, phic, phicL, phicR, f_corr, rhsc, FncL, FncR, FvcL, FvcR, snap_nc, snap_vc, snap_phic, Gamma_0, kappa_0)

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    # cmap_n = colormap(xx,yy,snap_n)
    # plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    # cmap_nc = colormap(xx,yy,snap_nc)
    # plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    # colormap(xx, yy, (snap_nc - snap_n) / rho_0)
    # plt.title("Density Difference: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    # plot(X,snap_n)
    # plt.title("Density: No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    # plot(X, snap_nc)
    # plt.title("Density: Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    # plot(X,snap_v)
    # plt.title("Velocity: No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))


    numRows = len(snap_n)
    loop = int(numRows / 2)
    for kk in range(loop):
        snap_n[[kk, numRows - kk - 1], :] = snap_n[[numRows - kk - 1, kk], :]

    numRowsc = len(snap_nc)
    loopc = int(numRowsc / 2 )
    for kk in range(loopc):
        snap_nc[[kk, numRowsc - kk - 1], :] = snap_nc[[numRowsc - kk - 1, kk], :]

    return snap_n, snap_nc, L , t