import numpy as np
import matplotlib.pyplot as plt

# Numerical Parameters
N = int(5e2 + 1)  # Grid Points
T = int(5e2 + 1)  # Time Steps
L = 10  # Domain Size
x = np.linspace(0, L, N)  # Spatial Domain
dx = x[2] - x[1]  # Grid Size
dt = 1e-3  # Time Step Size
lambda_ = dt / dx
t_tot = dt * T
num_snaps = 10
snaps = int((T-1) / num_snaps) # TODO: Knob
t = np.linspace(0, t_tot, int(T/num_snaps)) # Temporal Domain
xx, tt = np.meshgrid(x,t, sparse=False, indexing='xy') # Spatial-Temporal Domain

# Mathematical Parameters
n_0 = 3 / (4 * np.pi)
Gamma_0 = input("Enter Gamma_0: ")
kappa_0 = input("Enter kappa_0: ")
beta = 1
correlation = [False, True]

# Correlation Parameters
k_fft_norm = 2 * np.pi / (N * dx)
k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)  # Fourier Domain
x3 = np.linspace(-L, 2 * L, 3 * N - 2)  # Correlation Domain

# Initial Conditions
disp_freq = 3 * 2 * np.pi / L
n_IC = n_0 * np.ones(N) + .1 * np.sin(disp_freq * x)
v_IC = .1 * np.sin(disp_freq * x)

# Memory Allocation
n = np.zeros((2,N))
v = np.zeros((2,N))
phi = np.zeros((2,N))

n_tot = np.zeros((2, snaps, N))
v_tot = np.zeros((2, snaps, N))
phi_tot = np.zeros((2, snaps, N))

vec_tot = [n_tot, v_tot]

# Initial Conditions
n = np.copy(n_IC)
v = np.copy(v_IC)
nc = np.copy(n_IC)
vc = np.copy(v_IC)

n_tot[0,:] = np.copy(n)
v_tot[0,:] = np.copy(v)

def rewrite(n,v,phi, correlation):

    def flux(n, v):
        vec = [n,v]
        vec_L = np.roll(vec,1)
        flux_n = n*v
        flux_v = .5*v*v+np.log(n)
        vec_flux = [flux_n, flux_v]
        vec_flux_L = np.roll(vec_flux, 1)

        for ii in range(2):
            if vec_L[ii] > vec[ii]:
                godunov_flux = np.maximum(vec_flux_L[ii], vec_flux[ii])
            elif vec_L[ii] < vec[ii]:
                godunov_flux = np.minimum(vec_flux_L[ii], vec_flux[ii])
            else:
                godunov_flux = 0.0
        return godunov_flux

    def compute_phi(n, phi):
        A = np.zeros(N)
        # Define b
        b = 3 - 4 * np.pi * dx * dx * n
        b = b - np.mean(b)
        # First sweep
        A[0] = -0.5
        b[0] = -0.5 * b[0]
        for ii in range(1, N):
            A[ii] = -1 / (2 + A[ii - 1])
            b[ii] = (b[ii - 1] - b[ii]) / (2 + A[ii - 1])
        # Second sweep
        phi[0] = b[N - 1] - b[N - 2]
        for ii in range(1, N - 1):
            phi[ii] = (b[ii - 1] - phi[ii - 1]) / A[ii - 1]
        return phi

# ===== #
# Solve #
# ===== #
    for corr in correlation:
        if corr:
            vec = vecc_tot
        else:
            vec = vec_tot
        for tt in range(1,T):
            for ii in range(1,N):
                vec[:,ii] = vec[:,ii] - lambda_ * (flux(vec[0][ii+1],vec[1][ii+1])-flux(vec[0][ii],vec[1][ii]))
            vec[:,N] = vec[:,N] - lambda_ * (flux(vec[0][0],vec[1][0])-flux(vec[0][N],vec[1][N]))
            
        #  Snapshot
        if tt % ((T-1)/snaps) == 0:
            snap = int(tt/snaps)
            n_tot[snap,:] = n
            v_tot[snap,:] = v
            phi_tot[snap,:] = phi

        phi = compute_phi(n,phi)



    correlations = False


# ==== #
# Plot #
# ==== #

rho_tot_rev = rho_tot[:,::-1,:]
v_tot_rev = v_tot[:,::-1,:]

def plot(domain, u_tot_rev):
    for cor in range(2):
        fig = plt.figure()
        plt.title("Plot")
        for ii in range(snaps):
            plt.plot(domain, u_tot_rev[cor,ii], label=str((T-1)/snaps))
        plt.legend()
        plt.show(block=False)

def cmap(x_domain, t_domain, u_tot_rev):
    for cor in range(2):
        fig = plt.figure()
        color_map = plt.contourf(x_domain, t_domain, u_tot_rev)
        plt.colorbar()
        plt.ylabel("Time")
        plt.x_label("Space")
        plt.show(block=False)    

def calculate_2dft(u_tot_rev):
    ft = np.fft.ifftshift(u_tot_rev)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def disp_rel_cmap(x_domain, t_domain, u_tot_rev):
    ft = calculate_2dft(u_tot_rev)
    cmap(x_domain, t_domain, ft)
    
def plot3D():
    fig = plt.figure()
    