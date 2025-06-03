import numpy as np
import matplotlib.pyplot as plt
import time


class domain:
    def __init__(self,Xpts, X0, Xf, dt, T0, Tf, totsnaps, Gamma_0, kappa_0, beta):
        self._Xpts = Xpts
        # Many helper classes refer to the number of grid points via the
        # attribute ``nx``.  Define it here to avoid ``AttributeError`` when
        # those routines access ``domain.nx``.
        self.nx = Xpts
        self._X0 = X0
        self._Xf = Xf
        self._dt = dt
        self._T0 = T0
        self._Tf = Tf
        self._totsnaps = totsnaps
        self._Gamma_0 = Gamma_0
        self._kappa_0 = kappa_0
        self._beta = beta

        # Space
        self._Xlng = Xf - X0
        self._dx = self._Xlng / Xpts  # Grid Size
        self._X = np.arange(X0, Xf, self._dx)  # Spatial Domain

        # Time
        self._Tlng = Tf - T0
        self._Tpts = int(self._Tlng / dt)  # Time Steps
        self._T = np.linspace(T0, Tf, num=self._Tpts, endpoint=False)

        # Numerical Parameters
        self._xx, self._tt = np.meshgrid(self._X, self._T, sparse=False, indexing='xy')  # Spatial-Temporal Domain
        self._lmbd = dt/self._dx  # 1e2/2.5e2 = .4
        self._Gamma_0 = Gamma_0  # input("Enter Gamma_0: ")
        self._kappa_0 = kappa_0  # input("Enter kappa_0: ")
        self._beta = beta
        self._rho_0 = 3 / (4 * np.pi)  # Mean Density

        # self.Gamma_0 = 1  # input("Enter Gamma_0: ")
        # self.kappa_0 = 1  # input("Enter kappa_0: ")
        # self.beta = 1

        # Correlation Parameters
        k_fft_norm = 2 * np.pi / (Xpts * self._dx)
        self._k = k_fft_norm * np.linspace(-Xpts / 2, Xpts / 2 - 1, Xpts)  # Fourier Domain
        self._x3 = np.linspace(-self._Xlng, 2 * self._Xlng, 3 * Xpts - 2)  # Correlation Domain

    @property
    def Xpts(self): return self._Xpts
    @property
    def X0(self): return self._X0
    @property
    def Xf(self): return self._Xf
    @property
    def dt(self): return self._dt
    @property
    def T0(self): return self._T0
    @property
    def Tf(self): return self._Tf
    @property
    def totsnaps(self): return self._totsnaps
    @property
    def Gamma_0(self): return self._Gamma_0
    @property
    def kappa_0(self): return self._kappa_0
    @property
    def beta(self): return self._beta

    @Xpts.setter
    def Xpts(self, value): self._Xpts = value
    @X0.setter
    def X0(self, value): self._X0 = value
    @Xf.setter
    def Xf(self, value): self._Xf = value
    @dt.setter
    def dt(self, value): self._dt = value
    @T0.setter
    def T0(self, value): self._T0 = value
    @Tf.setter
    def Tf(self, value): self._Tf = value
    @totsnaps.setter
    def totsnaps(self, value): self._totsnaps = value
    @Gamma_0.setter
    def Gamma_0(self, value): self._Gamma_0 = value
    @kappa_0.setter
    def kappa_0(self, value): self._kappa_0 = value
    @beta.setter
    def beta(self, value): self._beta = value

    @Xlng.setter
    def Xlng(self, value): self._Xlng = value
    @dx.setter
    def dx(self, value): self._dx = value
    @X.setter
    def X(self, value): self._X = value
    # @beta.setter
    # def beta(self, value): self._beta = value
    # @beta.setter
    # def beta(self, value): self._beta = value
    # @beta.setter
    # def beta(self, value): self._beta = value
    # @beta.setter
    # def beta(self, value): self._beta = value
    # @beta.setter
    # def beta(self, value): self._beta = value
    # @beta.setter
    # def beta(self, value): self._beta = value

class simulate:
    def __init__(self, domain, rhoIC, mIC, eIC):
        # Store reference to the spatial/temporal discretization.  The original
        # code saved this under ``_domain`` and then attempted to access
        # ``self.domain`` elsewhere, which raised ``AttributeError``.  Keep the
        # public attribute consistent with how it is used later.
        self.domain = domain
        self._rho, self._rhotot, self._frho, self._frhotot = self.mem(rhoIC)
        self._m, self._mtot, self._fm, self._fmtot = self.mem(mIC)
        self._e, self._etot, self._fe, self._fetot = self.mem(eIC)
        # Initialize electrostatic potential to avoid missing attribute errors
        # in ``solve`` when ``self.phi`` is first updated.
        self.phi = np.zeros(self.domain.nx)

    @property
    def rhoIC(self): return self._rhoIC
    @property
    def mIC(self): return self._mIC
    @property
    def eIC(self): return self._eIC

    @rhoIC.setter
    def rhoIC(self, value): self._rhoIC = value
    @mIC.setter
    def mIC(self, value): self._mIC = value
    @eIC.setter
    def eIC(self, value): self._eIC = value

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

    #TODO: Expand Correlations
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
        for c in range(2): # Iterate Correlations
            # Reset Temporary Variables
            self.rho[:] = 0
            self.m[:] = 0
            self.e[:] = 0
            snap = 1

            if c == 0:
                st = time.time()
            for tt in range(1,self.domain.Tpts): # Iterate Time

                # Solve Internal Fields: Correlations, Pressure, Electrostatic Potential
                self.cor = self.meanfield(self.domain.k,self.rho,self.domain.Gamma_0,self.domain.kappa_0)
                self.phi = self.solve_phi(self.rho,self.phi)
                self.p = self.solve_pressure(c, self.rho, self.phi, self.cor)

                # Solve Nonlinear Fluxes
                self.frho = self.m
                self.fm = self.m**2+self.p
                self.fe = (self.e+self.p)*self.m/self.rho

                # Update Next Time Step
                self.rho = self.rho - self.domain.lmbd*(self.frho - np.roll(self.frho,1))
                self.m = self.m - self.domain.lmbd*(self.fm - np.roll(self.fm,1))
                self.e = self.e - self.domain.lmbd*(self.fe - np.roll(self.fe,1))

                # Take Snapshot
                if (tt-1) % (self.domain.Tpts / self.domain.totsnaps) == 0:
                    self.rhotot[c,snap], self.mtot[c,snap], self.etot[c,snap] = self.rho, self.m, self.e
                    self.frhotot[c, snap], self.fmtot[c, snap], self.fetot[c, snap] = self.frho, self.fm, self.fe
                    snap += 1

                if tt == int(self.domain.Tpts / 10):
                    et = time.time()
                    elapsed_time = et - st
                    print('Execution time:', elapsed_time, 'seconds')
                    print('Approximate total time:', 20 * elapsed_time, 'seconds')


class plotting:

    def __init__(self, domain, simulation):
        self.domain = domain(domain.nx, domain.X0, domain.Xf, domain.dt, domain.T0, domain.Tf, domain.totsnaps, domain.Gamma_0, domain.kappa_0, domain.beta)
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
        color_map = plt.imshow(u, cmap='viridis', origin='lower', extent=(self.domain.X0, self.domain.Xf, self.domain.T0, self.domain.Tf), aspect='auto')
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
        color_map = plt.imshow(fft, cmap='viridis', origin='lower', extent=(self.domain.X0, self.domain.Xf, self.domain.T0, self.domain.Tf), aspect='auto')
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
                                             extent=(self.domain.X0, self.domain.Xf, self.domain.T0, self.domain.Tf), aspect='auto', cmap='viridis')
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
    print(d1._Xpts)
    print(d1._X)

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
