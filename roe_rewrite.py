# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt

# TODO: Changes
#   A=P|D|P^-1 not A=PDP^-1
#   (1,N) v. N
#   vector PDE instead of n and v system
#   phi and f_c on rhs


def roe_rewrite():
# ============== #
# Define Methods #
# ============== #

    # Memory Allocation and Initialization
    def memory_allocation_pde(u_IC):
        u = np.copy(u_IC)
        flux = np.zeros((1, N))
        snap_u = np.zeros((snaps + 1, N))
        snap_u[0] = u_IC
        return u, flux, snap_u

    def memory_allocation_phi():
        u = np.zeros(N)
        matrix = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        return u, matrix, snap_u

    # Define Numerical Schemes
    def func_flux(V):
        n = V[0]
        v = V[1]

        # Flux Vector
        Fn = np.array(n * v)
        Fv = np.array(((v ** 2) / 2) + np.log(n))
        flux = np.array([Fn, Fv])

        return flux

    def flux_roe(V):
        n = V[0]
        v = V[1]

        roe_flux = np.zeros((2,N))

        # R = np.sqrt(n[0] / n[N-1])  # R_{j+1/2}
        # n_hat = R * n[ii]  # {hat rho}_{j+1/2}
        # v_hat = (R * v[ii + 1] + v[ii]) / (R + 1)  # {hat U}_{j+1/2}
        #
        # # Auxiliary variables used to compute P_{j+1/2}^{-1}
        # # Compute vector (W_{j+1}-W_j)
        # Vdif = V[:, ii + 1] - V[:, ii]
        #
        # # TODO: Does the order of eigenvalues or eigenvectors matter?
        # # Compute matrix P^{-1}_{j+1/2}
        # Pinv = np.array([[1 / (2 * n_hat), 1 / 2], [-1 / (2 * n_hat), 1 / 2]])
        #
        # # Compute matrix P_{j+1/2}
        # P = np.array([[n_hat, -n_hat], [1, 1]])
        #
        # # Compute matrix Lambda_{j+1/2}
        # D = np.array([[np.abs(v_hat + 1), 0], [0, np.abs(v_hat - 1)]])
        #
        # # Compute Roe matrix |A_{j+1/2}|
        # A = np.dot(np.dot(P, D), Pinv)
        #
        # # Compute |A_{j+1/2}| (W_{j+1}-W_j)
        # roe_flux[:, ii] = np.dot(A, Vdif)

        for ii in range(N-1):
            # Compute Roe Averages
            R = np.sqrt(n[ii + 1] / n[ii]) # R_{j+1/2}
            n_hat = R * n[ii]  # {hat rho}_{j+1/2}
            v_hat = (R * v[ii + 1] + v[ii]) / (R + 1) # {hat U}_{j+1/2}

            # Auxiliary variables used to compute P_{j+1/2}^{-1}
            # Compute vector (W_{j+1}-W_j)
            Vdif = V[:, ii + 1] - V[:, ii]

            # TODO: Does the order of eigenvalues or eigenvectors matter?
            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array([[1 / (2*n_hat), 1/2], [-1 / (2*n_hat), 1/2]])

            # Compute matrix P_{j+1/2}
            P = np.array([[n_hat, -n_hat], [1, 1]])

            # Compute matrix Lambda_{j+1/2}
            D = np.array([[np.abs(v_hat+1), 0], [0, np.abs(v_hat-1)]])

            # Compute Roe matrix |A_{j+1/2}|
            A = np.dot(np.dot(P, D), Pinv)

            # Compute |A_{j+1/2}| (W_{j+1}-W_j)
            roe_flux[:, ii] = np.dot(A, Vdif)

        # Compute Roe Averages
        R = np.sqrt(n[0] / n[N-1])  # R_{j+1/2}
        n_hat = R * n[N-1]  # {hat rho}_{j+1/2}
        v_hat = (R * v[0] + v[N-1]) / (R + 1)  # {hat U}_{j+1/2}

        # Auxiliary variables used to compute P_{j+1/2}^{-1}
        # Compute vector (W_{j+1}-W_j)
        Vdif = V[:, 0] - V[:, N-1]

        # TODO: Does the order of eigenvalues or eigenvectors matter?
        # Compute matrix P^{-1}_{j+1/2}
        Pinv = np.array([[1 / (2 * n_hat), 1 / 2], [-1 / (2 * n_hat), 1 / 2]])

        # Compute matrix P_{j+1/2}
        P = np.array([[n_hat, -n_hat], [1, 1]])

        # Compute matrix Lambda_{j+1/2}
        D = np.array([[np.abs(v_hat + 1), 0], [0, np.abs(v_hat - 1)]])

        # Compute Roe matrix |A_{j+1/2}|
        A = np.dot(np.dot(P, D), Pinv)

        # Compute |A_{j+1/2}| (W_{j+1}-W_j)
        roe_flux[:, N-1] = np.dot(A, Vdif)

        # ==============================================================
        # Compute Phi=(F(W_{j+1}+F(W_j))/2-|A_{j+1/2}| (W_{j+1}-W_j)/2
        # ==============================================================

        F = func_flux(V)
        # print("roe_flux", roe_flux.shape)
        # print("F[:, 0:N - 1]", F[:, 0:N - 1].shape)
        # print("F[:, 1:N]", F[:, 1:N].shape)
        # print("F[:, 0:N]", F[:, 0:N].shape)
        # print("F", F.shape)
        # roe_flux = 0.5 * (F[:, 0:N - 1] + F[:, 1:N]) - 0.5 * roe_flux
        roe_flux = 0.5 * (F + np.roll(F,1, axis=1)) - 0.5 * roe_flux

        # dF = (roe_flux[:, 1:N-1] - roe_flux[:, 0:N-2])
        dF = roe_flux - np.roll(roe_flux,-1, axis = 1)

        # dF_BC = roe_flux[:,0] - roe_flux[:,N-1]
        # dF = np.hstack([dF, np.atleast_2d(dF_BC).T])
        # print("dF.shape", dF.shape)
        # print("V.shape", V.shape)

        return (dF)

    def compute_phi(n_, phi_):
        # Define b
        b = 3 - 4 * np.pi * dx * dx * n_
        b = b - np.mean(b)
        # First sweep
        A[0] = -0.5
        b[0] = -0.5 * b[0]
        for ii in range(1, N - 1):
            A[ii] = -1 / (2 + A[ii - 1])
            b[ii] = (b[ii - 1] - b[ii]) / (2 + A[ii - 1])
        # Second sweep
        phi_[0] = b[N - 1] - b[N - 2]
        for ii in range(1, N - 1):
            phi_[ii] = (b[ii - 1] - phi_[ii - 1]) / A[ii - 1]
        return phi_

    # Miscellaneous
    def take_snapshot(tt, T, snaps, n_, v_, phi_, snap_n_, snap_v_, snap_phi_):
        snap_n_[int(tt / (T / snaps))] = n_
        snap_v_[int(tt / (T / snaps))] = v_
        snap_phi_[int(tt / (T / snaps))] = phi_

    def plot(x_, snap_u):
        plt.figure()
        for ii in range(len(snap_u)):
            plt.plot(x_, snap_u[ii], label="T = " + str(ii * dt * T / snaps))
        # plt.legend()

    def colormap(xx_, yy_, snap_u):
        plt.figure()
        clr = plt.contourf(xx_, yy_, snap_u)
        plt.colorbar()

# ========================== #
# Parameters, Initialization #
# ========================== #

    # Parameters
    N = int(1e2)  # Grid Points 5e2
    T = int(20)  # Time Steps 5e3
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    # TODO: x3
    dx = x[2] - x[1]  # Grid Size
    #TODO: define dt = CFL * dx / (max(abs(v)+a), where a = sqrt(gamma * p / n) is the updated speed of sound
    dt = 1e-3  # Time Step Size
    lambda_ = dt / dx
    t = dt * T

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = int(input("Number of Snapshots "))  # Number of Snapshots
    Gamma_0 = float(input("Value of Gamma "))  # Coulomb Coupling Parameter
    kappa_0 = float(input("Value of kappa "))  # screening something

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    v_IC = np.zeros(N)
    n_IC[0:int(N / 4)] = n_0 / 2
    n_IC[int(N / 4):int(3 * N / 4)] = 3 * n_0 / 2
    n_IC[int(3 * N / 4):N] = n_0 / 2
    v_IC[int(N / 2):N] = 0

    n, flux_n, snap_n = memory_allocation_pde(n_IC)
    v, flux_v, snap_v = memory_allocation_pde(v_IC)
    phi, A, snap_phi = memory_allocation_phi()

    V = np.array([n,v])
    flux_V = np.array([flux_n, flux_v])

    rhs = np.zeros((2, N))

    rho = np.zeros(N)
    f_corr = np.zeros(N)

# ===== #
# Solve #
# ===== #

    # No Correlation
    for tt in range(T + 1):
        # Phi - A * phi = b
        phi = compute_phi(n, phi)

        # Compute Flux
        dF = flux_roe(V)
        V0 = np.copy(V)

        rhs[1] = f_corr - Gamma_0 * phi

        #TODO: Make method AND + .5 * lambda_ * (rhs[:, 1:-2] - rhs[:,1:-2])
        # V0?
        # Check Boundary Conditions

        if tt < 10:
            # print("V", V.shape)
            # print("V0", V0.shape)
            print("n", n.shape)
            print(n)
            print("dF", dF.shape)
            print(dF)
            # print("V[:, 0:N-1]", V[:, 0:N-1].shape)
            # print("V[:, 0:N]", V[:, 0:N].shape)
            # print("dF[:,0:N-1]", dF[:,0:N-1].shape)
        # V[:, 0] = V0[:, 0] - lambda_ * dF[:, N-1]
        # V[:, 1:-2] = V0[:, 1:-2] - lambda_ * dF

        # V[:, 0] = V0[:, 0] - lambda_ * dF[:, N-1]
        V[:, 0:N-1] = V0[:, 0:N-1] - lambda_ * dF[:, 0:N-1]

        n = V[0]
        v = V[1]

        print("n at tt = ",tt, n)

        if np.amin(n) < 0:
            print("Bad density", np.amin(n), " at tt = ", tt)
            exit()

        if tt % (T / snaps) == 0:
            take_snapshot(tt, T, snaps, n, v, phi, snap_n, snap_v, snap_phi)

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    colormap(xx,yy,snap_n) # TODO: Debug
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plot(x, snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.show()
