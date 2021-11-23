import numpy as np
import matplotlib.pyplot as plt

from euler_poisson_solver import euler_poisson_corr, euler_poisson_nocorr, plot_solutions
from ep_corr_vs_no_corr import u_corr, u_nocorr, plot_u_solutions
from phi_corr_v_nocorr import phi_corr, phi_nocorr, plot_phi_solutions
from tester import tester
from test_corr import test_corr

def main():
    N = int(1e2)  # grid points
    T = int(1e4)  # time steps
    L = 10
    x = np.linspace(0, L - L / N, N)
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]
    dt = 1e-4  # step size
    n_0 = 3 / (4 * np.pi)  # Mean Density

    n_IC = n_0 * np.ones(N)
    u_IC = np.zeros(N)
    for ii in range(N):
        n_IC[ii] = n_IC[ii] + .2 * np.sin(2 * np.pi * ii * dx / L)
        u_IC[ii] = u_IC[ii] + np.sin(2 * np.pi * ii * dx / L)

    # Gamma_0s = [1/20,1,20]
    # kappa_0s = [.5,1,2]
    # correlation = True
    # for Gamma_0 in Gamma_0s:
    #     for kappa_0 in kappa_0s:
    #         nc = euler_poisson_corr(N,T,L,x,x3,dx,dt, n_IC, u_IC, n_0,correlation, Gamma_0,kappa_0)
    #         n = euler_poisson_nocorr(N,T,Gamma_0,dx,dt, n_IC, u_IC)
    #         plot_solutions(dt,x,n,nc,Gamma_0,kappa_0)

    # Gamma_0 = 1
    # kappa_0 = 1
    # correlation = True
    # nc = euler_poisson_corr(N, T, L, x, x3, dx, dt, n_IC, u_IC, n_0, correlation, Gamma_0, kappa_0)
    # n = euler_poisson_nocorr(N, T, Gamma_0, dx, dt, n_IC, u_IC)
    # plot_solutions(dt, x, n, nc, Gamma_0, kappa_0)

    # Gamma_0s = [1 / 20, 1, 20]
    # kappa_0s = [.5,1,2]
    # correlation = True
    # for Gamma_0 in Gamma_0s:
    #     for kappa_0 in kappa_0s:
    #         uc = u_corr(N, T, L, x, x3, dx, dt, n_IC, u_IC, n_0, correlation, Gamma_0, kappa_0)
    #         u = u_nocorr(N, T, Gamma_0, dx, dt, n_IC, u_IC)
    #         plot_u_solutions(dt, x, u, uc, Gamma_0, kappa_0)

    Gamma_0s = [1 / 20, 1, 20]
    kappa_0s = [.5,1,2]
    correlation = True
    for Gamma_0 in Gamma_0s:
        for kappa_0 in kappa_0s:
            phic = phi_corr(N, T, L, x, x3, dx, dt, n_IC, u_IC, n_0, correlation, Gamma_0, kappa_0)
            phi = phi_nocorr(N, T, Gamma_0, dx, dt, n_IC, u_IC)
            plot_phi_solutions(dt, x, phi, phic, Gamma_0, kappa_0)

    # tester()
    # test_corr()

main()
# shock_n = .5
# shock_u = 1
# for jj in range(int(N)):
#     if jj < N / 2:
#         n_IC[jj] = (1 + shock_n) * n_0
#     else:
#         n_IC[jj] = (1 - shock_n) * n_0
#
# u_IC = np.zeros(N)
# for ii in range(int(N / 2)):
#     u_IC[ii] = u_IC[ii] + shock_u
