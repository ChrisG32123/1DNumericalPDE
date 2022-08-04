from post243A import solve
from colormap_test import colormap_trial
from colormap import color
from godunov_test import godunov_test
from godunov_scheme import godunov_solve
from organized_solve import organized_solve
from roe_solver import roe_solve
from roe_solver_2 import roe_solve_2
from roe_rewrite import roe_rewrite
from continuity_roe_solver import continuity_roe_solver
from godunov_scalar_continuity import godunov_scalar_continuity
from test_convolution import test_convolution, half_interval_test_convolution, meyer_convolution, meyer_convolution_1, meyer_convolution_2
from icops_final import icops
from corr_change import corr_change
from subplots import subplot
from tester import tester2
from dispersion_relations import disp_rel
from test import test
from rewrite_jup import rewrite

import matplotlib.pyplot as plt
import numpy as np

def main():
    # colormap_trial()
    # color()
    # solve()
    # organized_solve()
    # godunov_solve()
    # godunov_test()
    # roe_solve()
    # roe_solve_2()
    # roe_rewrite()
    # continuity_roe_solver()
    # godunov_scalar_continuity()
    # test_roll()
    # test_convolution()
    # half_interval_test_convolution()
    # meyer_convolution()
    # meyer_convolution_1()
    # meyer_convolution_2()
    # icops()
    # disp_rel()

    rewrite()

    # Gamma = [10,1,.1]
    # kappa = [.5,1,2]

    # corr_change(1,1)
    # test()
    # corr_change(.1,2)
    # for ii in range(len(Gamma)):
    #     for jj in range(len(kappa)):
    #         corr_change(Gamma[ii], kappa[jj])
    #         print("done")

    # # densities = np.ndarray(shape=(2,9,500))
    # densities = [] # TODO: Change 100 to 500
    # densities_c = []
    # for ii in range(len(Gamma)):
    #     for jj in range(len(kappa)):
    #         snap_n, snap_nc, L, t = subplot(Gamma[ii], kappa[jj])
    #
    #         densities.append(snap_n)
    #         densities_c.append(snap_nc)
    #         print("done")
    #
    # fig, axes = plt.subplots(nrows=3, ncols=3)
    #
    # # find minimum of minima & maximum of maxima
    # minmin = np.min([np.min(densities), np.min(densities_c)])
    # maxmax = np.max([np.max(densities), np.max(densities_c)])
    #
    # images = []
    # for ii in range(3):
    #     for jj in range(3):
    #         im = axes[ii][jj].imshow(densities[ii+jj], vmin=minmin, vmax=maxmax,
    #                                  extent=(0,L,0,t), aspect='auto', cmap='viridis')
    #         axes[ii][jj].set_title('Γ = ' + str(Gamma[ii]) + ', κ = ' + str(kappa[jj]))
    #         axes[ii][jj].set_ylabel("Time")
    #         axes[ii][jj].set_xlabel("Space")
    #         images.append(im)
    #
    # fig.tight_layout(pad = .01)
    # fig.subplots_adjust(top=0.9)
    # fig.suptitle("Density: No Correlations")
    # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1)

    # plt.show()

main()
# TODO:
#   Better Algorithm: Godunov Scheme
#   No Approximations
#   Multiple Species
#   Three Dimensional
