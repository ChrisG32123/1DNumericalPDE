import numpy as np

# Parameters
N = int(1e2)  # Grid Points
T = int(1e3)  # Time Steps
L = 10  # Domain Size
x = np.linspace(0, L - L / N, N)  # Domain
x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
dx = x[2] - x[1]  # Grid Size
dt = 1e-3  # Time Step Size
lambda_ = dt / dx
t = dt * T

n_0 = 3 / (4 * np.pi)  # Mean Density
snaps = int(input("Number of Snapshots "))  # Number of Snapshots
Gamma_0 = float(input("Value of Gamma "))  # Coulomb Coupling Parameter
kappa_0 = float(input("Value of kappa "))  # screening something
rho = np.zeros(N)
f_corr = np.zeros(N)

# Memory Allocation
n = np.zeros(N)
v = np.zeros(N)
phi = np.zeros(N)
A = np.zeros(N)
snap_n = np.zeros((snaps + 1, N))
snap_v = np.zeros((snaps + 1, N))
snap_phi = np.zeros((snaps + 1, N))

nc = np.zeros(N)
n3 = np.zeros(3 * N)
vc = np.zeros(N)
phic = np.zeros(N)
Ac = np.zeros(N)
snap_nc = np.zeros((snaps + 1, N))
snap_vc = np.zeros((snaps + 1, N))
snap_phic = np.zeros((snaps + 1, N))

# Initial Conditions
# n_IC = rho_0 * np.ones(nx)
# v_IC = np.zeros(nx)
# for ii in range(nx):
#     n_IC[ii] = n_IC[ii] + (rho_0 / 2) * np.sin(2 * np.pi * ii * dx / Xlngth)
#     v_IC[ii] = v_IC[ii] + np.sin(2 * np.pi * ii * dx / Xlngth)
n_IC = n_0 * np.ones(N)
v_IC = np.zeros(N)
for ii in range(N):
    if ii < N / 2:
        n_IC[ii] = 3 * n_0 / 2
        v_IC[ii] = 1
    else:
        n_IC[ii] = n_0 / 2
        v_IC[ii] = 0

# Initialization
n = n_IC
v = v_IC
nc = n_IC
vc = v_IC
snap_n[0] = n_IC
snap_v[0] = v_IC
snap_nc[0] = n_IC
snap_vc[0] = v_IC

# Define Flux Functions
def f_n(n_, v_):
    return n_ * v_

def f_v(nc_, vc_, phic_):
    return .5 * vc_ * vc_ + np.log(nc_) + Gamma_0 * phic_ + f_corr