from post243A import solve
import importlib
colormap_trial = importlib.import_module('Old.03_colormap_test').colormap_trial
color = importlib.import_module('Old.02_colormap').color
godunov_test = importlib.import_module('Old.12_godunov_test').godunov_test
godunov_solve = importlib.import_module('Old.11_godunov_scheme').godunov_solve
organized_solve = importlib.import_module('Old.14_organized_solve').organized_solve
roe_solve = importlib.import_module('Old.20_roe_solver').roe_solve
roe_solve_2 = importlib.import_module('Old.21_roe_solver_2').roe_solve_2
roe_rewrite = importlib.import_module('Old.19_roe_rewrite').roe_rewrite
continuity_roe_solver = importlib.import_module('Old.04_continuity_roe_solver').continuity_roe_solver
godunov_scalar_continuity = importlib.import_module('Old.10_godunov_scalar_continuity').godunov_scalar_continuity
from test_convolution import test_convolution, half_interval_test_convolution, meyer_convolution, meyer_convolution_1, meyer_convolution_2
icops = importlib.import_module('Old.13_icops_final').icops
from corr_change import corr_change
from subplots import subplot
from tester import tester2
disp_rel = importlib.import_module('Old.05_dispersion_relations').disp_rel
from test import test
rewrite = importlib.import_module('Old.18_rewrite_jup').rewrite

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
    #         snap_n, snap_nc, Xlngth, t = subplot(Gamma[ii], kappa[jj])
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
    #                                  extent=(0,Xlngth,0,t), aspect='auto', cmap='viridis')
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
