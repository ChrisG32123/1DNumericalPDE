import numpy as np
import matplotlib.pyplot as plt
import time

# Spatial Domain
nx = int(1e2)
L = 10
dx = L / nx
X = np.linspace(0, L, nx, endpoint=False)

# Temporal Domain
end_time = 40
t = 0
num_snaps = 100
snap_interval = end_time / num_snaps
cur_snap = 0

mean_n = 3 / (4*np.pi)
Gamma = 10
kappa = 2
therm_cond = 1

# Memory Allocation
ar_shape = (2, nx)
tot_ar_shape = (num_snaps+1, 2, nx)
convs_ar_shape = (num_snaps+1, 2)

n, ntot, nint = np.empty(ar_shape), np.empty(tot_ar_shape), np.empty(convs_ar_shape)
u, utot, uint = np.empty(ar_shape), np.empty(tot_ar_shape), np.empty(convs_ar_shape)
T, Ttot, Tint = np.empty(ar_shape), np.empty(tot_ar_shape), np.empty(convs_ar_shape)
phi, phimtx, phitot = np.empty(ar_shape), np.empty(ar_shape), np.empty(tot_ar_shape)


# Shift to Left
def l(array):
    return np.roll(array, 1, axis=-1)


# Shift to Right
def r(array):
    return np.roll(array, -1, axis=-1)


# Derivative
def ddx(array):
    return (array - l(array)) / dx


# Second Derivative
def d2dx2(array):
    return (r(array) - 2 * array + l(array)) / (dx * dx)


# Save Snap
def snapshot():
    ntot[cur_snap] = n
    utot[cur_snap] = u
    Ttot[cur_snap] = T
    phitot[cur_snap] = -ddx(phi)

    nint[cur_snap] = np.trapz(n, X)
    uint[cur_snap] = np.trapz(u, X)
    Tint[cur_snap] = np.trapz(T, X)


# Solve Phi
def solvephi(den):
    phi = np.zeros((2, nx))
    # phimtx * phi = b
    # Define b
    b = 3 - 4 * np.pi * dx * dx * den
    b = b - np.mean(b)
    # First sweep
    phimtx[:, 0] = -0.5
    b[:, 0] = -0.5 * b[:, 0]
    for ii in range(1, nx - 1):
        phimtx[:, ii] = -1 / (2 + phimtx[:, ii - 1])
        b[:, ii] = (b[:, ii - 1] - b[:, ii]) / (2 + phimtx[:, ii - 1])
    # Second sweep
    phi[:, 0] = b[:, nx - 1] - b[:, nx - 2]
    for ii in range(1, nx - 1):  # TODO: CHECK INDICES
        phi[:, ii] = (b[:, ii - 1] - phi[:, ii - 1]) / phimtx[:, ii - 1]

    return phi


def fluxes(n, u):
    fluxn = n*u
    fluxu = .5*u*u

    return fluxn, fluxu


def upwinding(array, flux):
    upwind_flux = np.where(array > 0, flux-l(flux), r(flux)-flux)
    return upwind_flux


# def godunov(array, flux):
#     arrayL = l(array)
#     arrayR = array
#
#     fluxL = np.where(arrayL >= 0, l(flux), flux)
#     fluxR = np.where(arrayR >= 0, flux, r(flux))
#
#     godunov_flux = np.where(arrayR < arrayL, np.minimum(fluxL, fluxR), np.maximum(fluxL, fluxR))
#     return godunov_flux


def godunov(array, flux):
    arrayL = l(array)
    arrayR = array

    fluxL = l(flux)
    fluxR = flux

    max_flux = np.maximum(fluxL, fluxR)
    min_flux = np.minimum(fluxL, fluxR)

    min_flux[(arrayR > 0) & (arrayL < 0)] = 0

    godunov_flux = np.where(arrayR > arrayL, max_flux, min_flux)

    return godunov_flux


# Initial Condition
ICchoice = np.array([0, 3, 3])  # Gaussian = 0, Wave = 1, Random = 2, Constant = 3, Zero = 4
perturb_ampltd = [.01, .01, .01]
gaus_lngth = .05 * L
IC_freq = 2 * np.pi / L
ICdict = np.array([
    np.array([
        mean_n * np.ones(nx) + perturb_ampltd[0] * np.exp(-((X - L / 2) ** 2) / (2 * gaus_lngth ** 2)) - perturb_ampltd[0] * np.sqrt(2 * np.pi) * gaus_lngth,
        mean_n * np.ones(nx) + perturb_ampltd[0] * np.cos(3 * IC_freq * X),
        mean_n * np.ones(nx) + perturb_ampltd[0] * np.random.random(nx),
        mean_n * np.ones(nx),
        np.zeros(nx)
    ]),
    np.array([
        np.ones(nx) + perturb_ampltd[1] * np.exp(-((X - L / 2) ** 2) / (2 * gaus_lngth ** 2)) - perturb_ampltd[1] * np.sqrt(2 * np.pi) * gaus_lngth,
        np.zeros(nx) + perturb_ampltd[1] * np.cos(3 * IC_freq * X),
        np.zeros(nx) + perturb_ampltd[1] * np.random.random(nx),
        np.ones(nx),
        np.zeros(nx)
    ]),
    np.array([
        np.ones(nx) + perturb_ampltd[2] * np.exp(-((X - L / 2) ** 2) / (2 * gaus_lngth ** 2)) - perturb_ampltd[2] * np.sqrt(2 * np.pi) * gaus_lngth,
        np.zeros(nx) + perturb_ampltd[2] * np.cos(3 * IC_freq * X),
        np.zeros(nx) + perturb_ampltd[2] * np.random.random(nx),
        np.ones(nx),
        np.zeros(nx)
    ])
])

n[:] = ICdict[0, ICchoice[0]]
u[:] = ICdict[1, ICchoice[1]]
T[:] = ICdict[2, ICchoice[2]]

snapshot()  # Store Initial Condition

start = time.time()
while t < end_time:

    # Find time step based off of wave speed
    dt = .1 * dx * dx / np.max(np.array([n, u, T]))  # TODO: Figure Out CFL
    if dt >= snap_interval:  # Enforce that time step must be smaller than snapshot speed
        dt = snap_interval
    if t+dt > end_time:  # Last time step
        dt = end_time-t

    t += dt  # Increment time by the adjusted time step
    lmbd = dt / dx
    mu = dt / (dx*dx)
    if round(t+dt) > round(t):
        print(t)

    phi = solvephi(n)
    flux_n, flux_u = fluxes(n, u)
    urhs = -T*ddx(np.log(n)) - ddx(phi)
    Trhs = (therm_cond*ddx(T)-T*u)*ddx(np.log(n))-2*u*ddx(T)

    n = n - lmbd * (upwinding(u, flux_n))
    u = u - lmbd * (upwinding(u, flux_u)) + dt*urhs
    T = T + dt*therm_cond*d2dx2(T) + dt*Trhs

    # n = n - dt * ddx(godunov(n, flux_n))
    # u = u - dt * ddx(godunov(u, flux_u)) + dt*urhs
    # T = T + dt*d2dx2(therm_cond*T) + dt*Trhs

    if t >= cur_snap*snap_interval:
        # print("lambda = ", lmbd)
        if cur_snap < num_snaps:
            cur_snap += 1
            snapshot()

        # Track Progress
        if cur_snap % int(num_snaps / 10) == 0:
            print(str(round(100 * t / end_time)) + "% Done")

end = time.time()


def data_visualize(*args):
    sys_data_to_visualize = np.array([np.swapaxes(array, 0, 1) for array in args])
    return sys_data_to_visualize


def data_names(var_name, *args):
    cornames = ["NC", "C"]
    if var_name.strip():
        var_name = " " + var_name
    names = np.array([np.array([name + var_name + " " + corname for corname in cornames]) for name in args])
    return names


# Plotting
def plot_snaps(data, name, xaxis, yaxis):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for kk in range(len(data)):
        ax.plot(X, data[kk], label="T = " + str(round(kk * dt * end_time / num_snaps, 2)))
    ax.set_title(name + " Γ = " + str(Gamma) + " κ = " + str(kappa))
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.legend()

    return ax


# Colormaps
def imshow(data, name, xaxis, yaxis):
    plt.figure()
    clr = plt.imshow(data, aspect='auto', origin='lower', extent=(0, L, 0, end_time))
    plt.colorbar()
    plt.title(name + " Γ = " + str(Gamma) + " κ = " + str(kappa))
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)


# Conserved Plots
def plot_conserved(data, name):
    plt.figure()
    plt.plot(np.arange(num_snaps + 1), data)
    plt.title(name + " Γ = " + str(Gamma) + " κ = " + str(kappa))
    plt.xlabel("Time")
    plt.ylabel(name)


# Subplot
def subplot(data, name, xaxis, title):
    row, col, plot, _ = data.shape
    fig, axs = plt.subplots(row, col)
    for ii in range(row):
        for jj in range(col):
            for kk in range(plot):
                axs[ii, jj].plot(X, data[ii, jj, kk], label="T = " + str(round(kk * dt * end_time / num_snaps, 2)))
            axs[ii, jj].set_title(name[ii, jj] + " Γ = " + str(Gamma) + " κ = " + str(kappa))
            axs[ii, jj].set_xlabel(xaxis[ii][jj])
            axs[ii, jj].set_ylabel(name[ii][jj])
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(title)


# Reformat data for plotting
syssnap = data_visualize(ntot, utot, Ttot, phitot)  # Shape: (4, 2, Tpts, nx)
# syssnapflux = data_visualize(nfluxtot, vfluxtot, efluxtot)  # Shape: (3, 2, Tpts, nx)
syssnapint = data_visualize(nint, uint, Tint)  # Shape: (3, 2, Tpts)

syssnapname = data_names("", "Density", "Velocity", "Temperature", "Electrostatic Potential")
syssnapfluxname = data_names("Flux", "Density", "Velocity", "Temperature")
syssnapintname = data_names("Integral", "Density", "Velocity", "Temperature")

syssnap_eq, syssnap_cor, syssnap_time, syssnap_lngth = syssnap.shape
# syssnapflux_eq, syssnapflux_cor, syssnapflux_time, syssnapflux_lngth = syssnapflux.shape
syssnapint_eq, syssnapint_cor, syssnapint_time = syssnapint.shape

# # Plot Fields
# syssnapplots = [[0 for ii in range(syssnap_cor)] for jj in range(syssnap_eq)]
# for ii in range(syssnap_eq):
#     for jj in range(syssnap_cor):
#         syssnapplots[ii][jj] = plot_snaps(syssnap[ii, jj], syssnapname[ii, jj], "Space", syssnapname[ii, jj])

# Subplot
subplot(syssnap, syssnapname, [["Space" for jj in range(syssnap_cor)] for ii in range(syssnap_eq)], "Fields")

# Plots: Integral Difference of Fields Over Time
for ii in range(len(syssnapint)):
    plot_conserved(syssnapint[ii, 0] - syssnapint[ii, 1], "Integral Difference")

# Color Plots: Fields
for ii in range(len(syssnap)):
    for jj in range(len(syssnap[ii])):
        imshow(syssnap[ii, jj], syssnapname[ii, jj], "Space", "Time")

# FFT
nsnap = syssnap[0]
for ii in range(len(nsnap)):
    n_fft = nsnap[ii] - np.mean(nsnap[ii])
    n_fft = np.swapaxes(n_fft, 0, 1)

    n_fft_flip = np.flip(n_fft, axis=1)
    n_fft = np.hstack((n_fft, n_fft_flip))

    fft = np.fft.fft2(n_fft, axes=(0, 1))
    fft = np.fft.fftshift(fft, axes=(0, 1))
    fft = np.transpose(fft)

    # Reflect  # TODO: If fft axis i length odd/even -> Add 1 in index -> reflect1[:, :int(fftlength1 / 2)+1]
    fftlength0, fftlength1 = fft.shape
    fft1_1st_half, fft1_2nd_half = fft[:int(fftlength0 / 2)], fft[int(fftlength0 / 2):]
    reflect1 = (np.flip(fft1_1st_half, axis=0) + fft1_2nd_half) / 2  # Reflect Bottom -> Up
    reflect1_1st_half, reflect1_2nd_half = reflect1[:, :int(fftlength1 / 2)], reflect1[:, int(fftlength1 / 2):]
    fft_avg = (np.flip(reflect1_1st_half, axis=1) + reflect1_2nd_half) / 2  # Reflect Left -> Right

    # Plot FFT
    plt.figure()
    plt.imshow(np.abs(fft_avg), aspect='auto', origin='lower', extent=(-0.5, num_snaps - 0.5, -0.5, nx - 0.5))
    plt.colorbar()
    plt.xlabel("Spatial Frequency (k)")
    plt.ylabel("Dispersion (ω)")
    plt.title("FFT: Γ = " + str(Gamma) + " κ0 = " + str(kappa))

end = time.time()
print("Total Time: ", end - start, " seconds")
# print("Off By:", np.round(100*(end - start - (elapsed_time * T / 100))/(end - start), 2), " %")

plt.show()

if __name__ == "__main__":
    print()