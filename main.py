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
from test import test_roll
from test_convolution import test_convolution, half_interval_test_convolution
def main():
    # colormap_trial()
    # color()
    # solve()
    # organized_solve()
    godunov_solve()
    # godunov_test()
    # roe_solve()
    # roe_solve_2()
    # roe_rewrite()
    # continuity_roe_solver()
    # godunov_scalar_continuity()
    # test_roll()
    # test_convolution()
    # half_interval_test_convolution()

main()
# TODO:
#   Better Algorithm: Godunov Scheme
#   No Approximations
#   Multiple Species
#   Three Dimensional
