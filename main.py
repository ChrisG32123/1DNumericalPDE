import numpy as np
import matplotlib.pyplot as plt

from continuity_solver import continuity
from cauchy_momentum_solver import cauchy
from poisson_solver import poisson
from euler_poisson_solver import euler_poisson
from convolution_tester import convolution_tester

def main():
    #continuity()
    #cauchy()
    #poisson()
    euler_poisson()
    #convolution_tester()

main()
