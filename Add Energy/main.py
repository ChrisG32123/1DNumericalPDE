from oop_code import domain, simulate, plotting
from DMD.DMDTest import DMDTest
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import time
import energy_addition_101322
import energy_correction_101822


# Xpts = int(1e2)  # Grid Points
# X0, Xf = 0, 10  # Space Domain
# dt = 1e-3  # Time Step Size
# T0, Tf = 0, 1  # Time Domain
# totsnaps = 100 # Total Snapshots in Time
# rho_0 = 3 / (4 * np.pi) # Mean Density
# Gamma_0 = 1  # input("Enter Gamma_0: ")
# kappa_0 = 1  # input("Enter kappa_0: ")
# beta = 1

def main():

    # energy_addition_101322.solve()
    energy_correction_101822.solve()

# Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main
# Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main
main()

    # d1 = domain(Xpts, X0, Xf, dt, T0, Tf, totsnaps, Gamma_0, kappa_0, beta)
    #
    # rhoIC = rho_0 * np.ones(Xpts)
    # mIC = np.zeros(Xpts)
    # eIC = np.zeros(Xpts)
    # s1 = simulate(d1, rhoIC, mIC, eIC)
    #
    # data = np.ndarray((100,len(s1.rho)))
    # for ii in range(len(data)):
    #     data[ii] = s1.rho*(1.01**ii)
    #
    # dmd = DMDTest(data,10,dt)
    # b = dmd.DMD()
    #
    # print('X', d1.X,
    #       'T', d1.T,
    #       'b', b)
    #
    # # plotting.cmap(d1.X, d1.T, b)
    #
    # plt.figure(1)
    # print('X', d1.X.shape, 'dmd.phi', dmd.Phi.shape)
    # plt.plot(d1.X, np.real(dmd.Phi))
    # plt.show(block=False)
    #
    # plt.figure(2)
    # print(d1.X)
    # print(b)
    # plotting.plot(d1.X, b)

# main()


# def main():
#     # colormap_trial()
#     # color()
#     # solve()
#     # organized_solve()
#     # godunov_solve()
#     # godunov_test()
#     # roe_solve()
#     # roe_solve_2()
#     # roe_rewrite()
#     # continuity_roe_solver()
#     # godunov_scalar_continuity()
#     # test_roll()
#     # test_convolution()
#     # half_interval_test_convolution()
#     # meyer_convolution()
#     # meyer_convolution_1()
#     # meyer_convolution_2()
#     # icops()
#     corr_change(1,1)
#     # disp_rel()
#
#     # rewrite()
#
#     # Gamma = [10,1,.1]
#     # kappa = [.5,1,2]
#     #
#     # corr_change(1,1)
#     # test()
#     # corr_change(.1,2)
#     # for ii in range(len(Gamma)):
#     #     for jj in range(len(kappa)):
#     #         corr_change(Gamma[ii], kappa[jj])
#     #         print("done")
#     #
#     # # densities = np.ndarray(shape=(2,9,500))
#     # densities = [] # TODO: Change 100 to 500
#     # densities_c = []
#     # for ii in range(len(Gamma)):
#     #     for jj in range(len(kappa)):
#     #         snap_n, snap_nc, L, t = subplot(Gamma[ii], kappa[jj])
#     #
#     #         densities.append(snap_n)
#     #         densities_c.append(snap_nc)
#     #         print("done")
#     #
#     # fig, axes = plt.subplots(nrows=3, ncols=3)
#     #
#     # # find minimum of minima & maximum of maxima
#     # minmin = np.min([np.min(densities), np.min(densities_c)])
#     # maxmax = np.max([np.max(densities), np.max(densities_c)])
#     #
#     # images = []
#     # for ii in range(3):
#     #     for jj in range(3):
#     #         im = axes[ii][jj].imshow(densities[ii+jj], vmin=minmin, vmax=maxmax,
#     #                                  extent=(0,L,0,t), aspect='auto', cmap='viridis')
#     #         axes[ii][jj].set_title('Γ = ' + str(Gamma[ii]) + ', κ = ' + str(kappa[jj]))
#     #         axes[ii][jj].set_ylabel("Time")
#     #         axes[ii][jj].set_xlabel("Space")
#     #         images.append(im)
#     #
#     # fig.tight_layout(pad = .01)
#     # fig.subplots_adjust(top=0.9)
#     # fig.suptitle("Density: No Correlations")
#     # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1)
#
#     plt.show()
#
# main()
# # TODO:
# #   Better Algorithm: Godunov Scheme
# #   No Approximations
# #   Multiple Species
# #   Three Dimensional
