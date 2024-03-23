import numpy as np
import matplotlib.pyplot as plt
import time


class domain:
    def __init__(self, Xpts, X0, Xf, dt, T0, Tf, totsnaps, Gamma_0, kappa_0, beta):
        self.Xpts = Xpts
        self.X0 = X0
        self.Xf = Xf
        self.dt = dt
        self.T0 = T0
        self.Tf = Tf
        self.totsnaps = totsnaps
        self.Gamma_0 = Gamma_0
        self.kappa_0 = kappa_0
        self.beta = beta

        # Space
        self.Xlng = Xf - X0
        self.dx = self.Xlng / Xpts  # Grid Size
        self.X = np.arange(X0, Xf, self.dx)  # Spatial Domain

        # Time
        self.Tlng = Tf - T0
        self.Tpts = int(self.Tlng / dt)  # Time Steps
        self.T = np.linspace(T0, Tf, num=self.Tpts, endpoint=False)

        # Numerical Parameters
        self.xx, self._tt = np.meshgrid(self.X, self.T, sparse=False, indexing='xy')  # Spatial-Temporal Domain
        self.lmbd = dt / self.dx  # 1e2/2.5e2 = .4
        self.Gamma_0 = Gamma_0  # input("Enter Gamma_0: ")
        self.kappa_0 = kappa_0  # input("Enter kappa_0: ")
        self.beta = beta
        self.rho_0 = 3 / (4 * np.pi)  # Mean Density

        # self.Gamma_0 = 1  # input("Enter Gamma_0: ")
        # self.kappa_0 = 1  # input("Enter kappa_0: ")
        # self.beta = 1

        # Correlation Parameters
        k_fft_norm = 2 * np.pi / (Xpts * self.dx)
        self.k = k_fft_norm * np.linspace(-Xpts / 2, Xpts / 2 - 1, Xpts)  # Fourier Domain
        self.x3 = np.linspace(-self.Xlng, 2 * self.Xlng, 3 * Xpts - 2)  # Correlation Domain


class simulate:
    def __init__(self, domain, rhoIC, mIC, eIC):
        self.domain = domain
        self.rho, self.rhotot, self.frho, self.frhotot = self.mem(rhoIC)
        self.m, self.mtot, self.fm, self.fmtot = self.mem(mIC)
        self.e, self.etot, self.fe, self.fetot = self.mem(eIC)

    def mem(self, uIC):
        utot = np.zeros((2, self.domain.totsnaps, self.domain.nx))
        utot[:, 0] = np.copy(uIC)
        u = np.copy(uIC)

        Futot = np.zeros((2, self.domain.totsnaps - 1, self.domain.nx))
        Fu = np.zeros(self.domain.nx)
        return u, utot, Fu, Futot

    def solve_pressure(self, c, rho, phi, cor):
        p = phi + np.log(rho)
        if c != 0:
            p += cor
        return p

    def solve_phi(self, rho, phi):
        A = np.zeros(self.domain.nx)
        # Define b
        b = 3 - 4 * np.pi * self.domain.dx * self.domain.dx * rho
        b = b - np.mean(b)
        # First sweep
        A[0] = -0.5
        b[0] = -0.5 * b[0]
        for ii in range(1, self.domain.nx):
            A[ii] = -1 / (2 + A[ii - 1])
            b[ii] = (b[ii - 1] - b[ii]) / (2 + A[ii - 1])
        # Second sweep
        phi[0] = b[self.domain.nx - 1] - b[self.domain.nx - 2]
        for ii in range(1, self.domain.nx - 1):
            phi[ii] = (b[ii - 1] - phi[ii - 1]) / A[ii - 1]
        return phi

    # TODO: Expand Correlations
    def meanfield(self, k, rho, Gamma, kappa):
        delta_n = rho - self.domain.rho_0

        def dcf(k, Gamma, kappa):
            return 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)

        dcfunc = dcf(k, Gamma, kappa)
        fhat = np.fft.fftshift(np.fft.fft(delta_n))
        conv = fhat * dcfunc
        conv = np.fft.ifft(np.fft.ifftshift(conv))
        conv = np.real(conv)
        return conv

    def solve(self):
        for c in range(2):  # Iterate Correlations
            # Reset Temporary Variables
            self.rho[:] = 0
            self.m[:] = 0
            self.e[:] = 0
            snap = 1

            if c == 0:
                st = time.time()
            for tt in range(1, self.domain.Tpts):  # Iterate Time

                # Solve Internal Fields: Correlations, Pressure, Electrostatic Potential
                self.cor = self.meanfield(self.domain.k, self.rho, self.domain.Gamma_0, self.domain.kappa_0)
                self.phi = self.solve_phi(self.rho, self.phi)
                self.p = self.solve_pressure(c, self.rho, self.phi, self.cor)

                # Solve Nonlinear Fluxes
                self.frho = self.m
                self.fm = self.m ** 2 + self.p
                self.fe = (self.e + self.p) * self.m / self.rho

                # Update Next Time Step
                self.rho = self.rho - self.domain.lmbd * (self.frho - np.roll(self.frho, 1))
                self.m = self.m - self.domain.lmbd * (self.fm - np.roll(self.fm, 1))
                self.e = self.e - self.domain.lmbd * (self.fe - np.roll(self.fe, 1))

                # Take Snapshot
                if (tt - 1) % (self.domain.Tpts / self.domain.totsnaps) == 0:
                    self.rhotot[c, snap], self.mtot[c, snap], self.etot[c, snap] = self.rho, self.m, self.e
                    self.frhotot[c, snap], self.fmtot[c, snap], self.fetot[c, snap] = self.frho, self.fm, self.fe
                    snap += 1

                if tt == int(self.domain.Tpts / 10):
                    et = time.time()
                    elapsed_time = et - st
                    print('Execution time:', elapsed_time, 'seconds')
                    print('Approximate total time:', 20 * elapsed_time, 'seconds')


class plotting:

    def __init__(self, domain, simulation):
        self.domain = domain(domain.nx, domain.X0, domain.Xf, domain.dt, domain.T0, domain.Tf, domain.totsnaps,
                             domain.Gamma_0, domain.kappa_0, domain.beta)
        self.simulation = simulation(domain, simulation.rhoIC, simulation.mIC, simulation.eIC)

    def plot(self, X, u):
        fig = plt.figure(figsize=(15, 15))
        plt.title('Γ = ' + str(self.domain.Gamma_0) + ', κ = ' + str(self.domain.kappa_0))
        for tt in range(self.domain.Tpts):
            if tt % (self.domain.Tpts / self.domain.totsnaps) == 0:
                plt.plot(X, u[tt], label=str(tt / self.domain.totsnaps))
        plt.legend()
        plt.show(block=False)

    def cmap(self, X, t, u):
        fig = plt.figure(figsize=(15, 15))
        #    color_map = plt.contourf(X, t, u[c,:])
        color_map = plt.imshow(u, cmap='viridis', origin='lower',
                               extent=(self.domain.X0, self.domain.Xf, self.domain.T0, self.domain.Tf), aspect='auto')
        plt.title('Γ = ' + str(self.domain.Gamma_0) + ', κ = ' + str(self.domain.kappa_0))
        plt.colorbar()
        plt.ylabel("Time")
        plt.xlabel("Space")
        plt.show(block=False)

    def calculate_2dft(self, u):
        fft = np.fft.ifftshift(u - np.mean(u[:]))
        fft = np.fft.fft2(fft)
        return np.abs(np.fft.fftshift(fft))

    def disp_rel_cmap(self, ux, ut, u):
        fft = self.calculate_2dft(u)
        fig = plt.figure(figsize=(15, 15))
        color_map = plt.contourf(ux, ut, fft)
        color_map = plt.imshow(fft, cmap='viridis', origin='lower',
                               extent=(self.domain.X0, self.domain.Xf, self.domain.T0, self.domain.Tf), aspect='auto')
        #    plt.title('Γ = ' + str(Gamma[ii]) + ', κ = ' + str(kappa[jj]))
        plt.title('Γ = ' + str(self.domain.Gamma_0) + ', κ = ' + str(self.domain.kappa_0))
        plt.colorbar()
        plt.ylabel("Time - Frequency")
        plt.xlabel("Space - Frequency")
        plt.show(block=False)

    # def plot3D():
    #     fig = plt.figure()

    def subplot(self, u):
        fig, axes = plt.subplots(nrows=3, ncols=3)
        # find minimum of minima & maximum of maxima
        minmin = np.min(uc for uc in u)
        maxmax = np.max(uc for uc in u)
        for c in range(2):
            images = []
            for ii in range(3):
                for jj in range(3):
                    im = axes[ii][jj].imshow(u[ii + jj], vmin=minmin, vmax=maxmax,
                                             extent=(self.domain.X0, self.domain.Xf, self.domain.T0, self.domain.Tf),
                                             aspect='auto', cmap='viridis')
                    axes[ii][jj].set_title('Γ = ' + str(self.domain.Gamma[ii]) + ', κ = ' + str(self.domain.kappa[jj]))
                    axes[ii][jj].set_ylabel("Time")
                    axes[ii][jj].set_xlabel("Space")
                    images.append(im)

            fig.tight_layout(pad=.01)
            fig.subplots_adjust(top=0.9)
            fig.suptitle("Density: nx Correlations")
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1)
            plt.show(block=False)


def main():
    Xpts = int(1e2)  # Grid Points
    X0, Xf = 0, 10  # Space Domain
    dt = 1e-3  # Time Step Size
    T0, Tf = 0, 1  # Time Domain
    totsnaps = 100  # Total Snapshots in Time
    rho_0 = 3 / (4 * np.pi)  # Mean Density
    Gamma_0 = 1  # input("Enter Gamma_0: ")
    kappa_0 = 1  # input("Enter kappa_0: ")
    beta = 1

    d1 = domain(Xpts, X0, Xf, dt, T0, Tf, totsnaps, Gamma_0, kappa_0, beta)
    print(d1.Xpts)
    print(d1.X)

    rhoIC = rho_0 * np.ones(Xpts)
    mIC = np.zeros(Xpts)
    eIC = np.zeros(Xpts)

    s1 = simulate(d1, rhoIC, mIC, eIC)

# # TODO: Main
# for c in range(2):
#     plot(X, n[c])
#     plot(X, v[c])
#     cmap(X, t, n[c])
#     disp_rel_cmap(X, t, n[c])
