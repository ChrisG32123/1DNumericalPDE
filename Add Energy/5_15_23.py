import numpy as np
import matplotlib.pyplot as plt
import time

# Space Domain
Xpts = int(1e2)
Xlngth = 10
dx = Xlngth / Xpts
X = np.linspace(0, Xlngth - dx, Xpts)

# Time Domain
Tpts = int(5e3) + 1
Tlngth = 20
T = np.linspace(0, Tlngth, Tpts)[:-1]
dt = Tlngth / (Tpts - 1)

lmbd = dt / dx

# Snapshots
snaps = 100 + 1
cursnap = 0
Y = np.linspace(0, Tlngth, snaps)
xx, yy = np.meshgrid(X, Y, sparse=False, indexing='xy')

# Spatial Frequency Domain
k_fft_norm = 2 * np.pi / (Xpts * dx)
k = k_fft_norm * np.linspace(-Xpts / 2, Xpts / 2 - 1, Xpts)

# Parameters
mean_n = 3 / (4 * np.pi)
Gamma_0 = .1
kappa_0 = 1
therm_cond = 1
q = 1

# Integration Domain/Extensions
num_dom_ext = 5
xp_min = -int(num_dom_ext/2)
xp_max = int(num_dom_ext/2 + 1)
Xp_pts = int(num_dom_ext * Xpts)
Xp = np.linspace(xp_min, xp_max, Xp_pts)

rp_min = 0
rp_max = int(np.sqrt(3) * Xp_pts)  # TODO: Check max(sqrt(y^2+z^2)) = sqrt(2)*max(y')?
Rp_pts = int(num_dom_ext * Xpts)
drp = rp_max / Rp_pts
Rp = np.linspace(rp_min+drp, rp_max, Rp_pts)

Kp = np.fft.fftfreq(Xp_pts)

xp, rp, kp = np.meshgrid(Xp, Rp, Kp)

# Define Potential u(|r'|)
u = Gamma_0 / Rp * np.exp(-kappa_0/Rp)

# Memory Allocation
n, ntot, nint, nflux, nfluxtot = np.zeros((2, Xpts)), np.zeros((snaps, 2, Xpts)), np.zeros((snaps, 2)), \
                                 np.zeros((2, Xpts)), np.zeros((snaps, 2, Xpts))
v, vtot, vint, vflux, vfluxtot = np.zeros((2, Xpts)), np.zeros((snaps, 2, Xpts)), np.zeros((snaps, 2)), \
                                 np.zeros((2, Xpts)), np.zeros((snaps, 2, Xpts))
e, etot, eint, eflux, efluxtot = np.zeros((2, Xpts)), np.zeros((snaps, 2, Xpts)), np.zeros((snaps, 2)), \
                                 np.zeros((2, Xpts)), np.zeros((snaps, 2, Xpts))
phi, phimtx, phitot = np.zeros((2, Xpts)), np.zeros((2, Xpts)), np.zeros((snaps, 2, Xpts))

# Memory Allocation - Correlations
# TODO: Dynamic to Static Memory

# Initial Condition
ICchoice = np.array([0, 0, 0])  # Gaussian = 0, Wave = 1, Random = 2, Constant = 3, Zero = 4
perturb_ampltd = [.01, .01, .01]
gaus_lngth = .05 * Xlngth
IC_freq = 2 * np.pi / Xlngth
ICdict = np.array([
    np.array([
        mean_n * np.ones(Xpts) + perturb_ampltd[0] * np.exp(-((X - Xlngth/2) ** 2) / (2 * gaus_lngth ** 2)) - perturb_ampltd[0] * np.sqrt(2 * np.pi) * gaus_lngth,
        mean_n * np.ones(Xpts) + perturb_ampltd[0] * np.cos(3 * IC_freq * X),
        mean_n * np.ones(Xpts) + perturb_ampltd[0] * np.random.random(Xpts),
        mean_n * np.ones(Xpts),
        np.zeros(Xpts)
    ]),
    np.array([
        np.ones(Xpts) + perturb_ampltd[1] * np.exp(-((X - Xlngth/2) ** 2) / (2 * gaus_lngth ** 2)) - perturb_ampltd[1] * np.sqrt(2 * np.pi) * gaus_lngth,
        np.zeros(Xpts) + perturb_ampltd[1] * np.cos(3 * IC_freq * X),
        np.zeros(Xpts) + perturb_ampltd[1] * np.random.random(Xpts),
        np.ones(Xpts),
        np.zeros(Xpts)
    ]),
    np.array([
        np.ones(Xpts) + perturb_ampltd[2] * np.exp(-((X - Xlngth/2) ** 2) / (2 * gaus_lngth ** 2)) - perturb_ampltd[2] * np.sqrt(2 * np.pi) * gaus_lngth,
        np.zeros(Xpts) + perturb_ampltd[2] * np.cos(3 * IC_freq * X),
        np.zeros(Xpts) + perturb_ampltd[2] * np.random.random(Xpts),
        np.ones(Xpts),
        np.zeros(Xpts)
    ])
])

n[:] = ICdict[0, ICchoice[0]]
v[:] = ICdict[1, ICchoice[1]]
e[:] = ICdict[2, ICchoice[2]]


# Shift to Left
def l(array):  # Has Been Checked
    return np.roll(array, 1, axis=-1)


# Shift to Right
def r(array):  # Has Been Checked
    return np.roll(array, -1, axis=-1)


# Derivative
def ddx(array):  # Has Been Checked
    return (array - l(array)) / dx


# Second Derivative
def d2dx2(array):  # Has Been Checked
    return (r(array) - 2 * array + l(array)) / (dx * dx)


# Check Nans
def checknan(array, name, time):
    if np.isnan(array.any()):
        print("Nan value at " + name + " at tt = " + str(time))
        return True
    return False


# Take Snapshot
def savesnap(cursnap):
    ntot[cursnap] = n
    vtot[cursnap] = v
    etot[cursnap] = e
    phitot[cursnap] = phi
    nfluxtot[cursnap] = nflux
    vfluxtot[cursnap] = vflux
    efluxtot[cursnap] = eflux
    nint[cursnap] = np.trapz(n, X)
    vint[cursnap] = np.trapz(v, X)
    eint[cursnap] = np.trapz(e, X)


# Solve Phi
def solvephi(den):
    phi = np.zeros((2, Xpts))
    # phimtx * phi = b
    # Define b
    b = 3 - 4 * np.pi * dx * dx * den
    b = b - np.mean(b)
    # First sweep
    phimtx[:, 0] = -0.5
    b[:, 0] = -0.5 * b[:, 0]
    for ii in range(1, Xpts - 1):
        phimtx[:, ii] = -1 / (2 + phimtx[:, ii - 1])
        b[:, ii] = (b[:, ii - 1] - b[:, ii]) / (2 + phimtx[:, ii - 1])
    # Second sweep
    phi[:, 0] = b[:, Xpts - 1] - b[:, Xpts - 2]
    for ii in range(1, Xpts - 1):  # TODO: CHECK INDICES
        phi[:, ii] = (b[:, ii - 1] - phi[:, ii - 1]) / phimtx[:, ii - 1]

    return phi


def meyer_correlations(den, temp):
    # Define c(|r'|)
    e_ext = np.tile(temp, num_dom_ext)
    c = np.exp(-u / e_ext[1]) - 1  # + u / e_ext[1]
    c_fft = np.fft.fftn(c)

    # Integrate rho'*F(c(r')) drho'
    r_integrand = Rp[np.newaxis, :] * c[np.newaxis, :] * np.sinc(Kp[:, np.newaxis] / np.pi * Rp[np.newaxis, :])
    c_int = np.trapz(r_integrand, axis=0, dx=drp)

    # n(x-x') - mean_n
    delta_n_xp = np.tile(den, num_dom_ext) - mean_n
    delta_n_fft = np.fft.fft(delta_n_xp)

    # Full FFT
    # corr_fft = c_int[1:] * delta_n_fft[1:]  # (499,)
    # corr_fft = np.insert(corr_fft, 0, 0)  # (500,)
    corr_fft = c_int * delta_n_fft

    # IFFT
    corr = 8 * np.pi ** 3 * np.fft.ifftn(corr_fft)  # (500,)
    # Take Middle Entries
    corr = corr.real[int(num_dom_ext / 2) * Xpts:Xp_pts - int(num_dom_ext / 2) * Xpts]
    return corr


start = time.time()
for tt in range(Tpts):

    phi = solvephi(n)

    # Flux
    nflux = dt * ddx(n * v)
    vflux = dt * (ddx(.5*v*v) + e*ddx(n)/n)
    eflux = dt * (- therm_cond*d2dx2(e) - therm_cond*ddx(n)/n*ddx(e) + v*e*ddx(n)/n + 2*v*ddx(e))

    # Correlations
    vcorr = np.zeros((2, Xpts))
    # vcorr[1] += dt*(Gamma_0*ddx(phi[1]) + e[1]*ddx(meyer_correlations(n[1], e[1])))
    # vcorr[1] += dt*(e[1]*ddx(meyer_correlations(n[1], e[1])))

    # Store Values
    if (tt % int(Tpts / (snaps - 1))) == 0:
        savesnap(cursnap)
        cursnap += 1

    # Solve
    n = n - nflux
    v = v - vflux - vcorr
    e = e - eflux

    # Check Nans
    ncheck, vcheck, echeck = checknan(n, "n", tt), checknan(v, "n", tt), checknan(e, "n", tt)
    nfluxcheck, vfluxcheck, efluxcheck = checknan(nflux, "n", tt), checknan(vflux, "n", tt), checknan(eflux, "n", tt)
    if ncheck or vcheck or echeck or nfluxcheck or vfluxcheck or efluxcheck:
        exit()

    # Track Progress
    if tt % int(Tpts / 10) == 0:
        print(str(round(100 * tt / Tpts)) + "% Done")
end = time.time()


def data_visualize(*args):  # Has Been Checked
    sys_data_to_visualize = np.array([np.swapaxes(array, 0, 1) for array in args])
    return sys_data_to_visualize


def data_names(var_name, *args):
    cornames = ["NC", "C"]
    if var_name.strip():
        var_name = " " + var_name
    names = np.array([np.array([name + var_name + " " + corname for corname in cornames]) for name in args])
    return names


# Reformat data for plotting
syssnap = data_visualize(ntot, vtot, etot, phitot)  # Shape: (4, 2, Tpts, nx)
syssnapflux = data_visualize(nfluxtot, vfluxtot, efluxtot)  # Shape: (3, 2, Tpts, nx)
syssnapint = data_visualize(nint, vint, eint)  # Shape: (3, 2, Tpts)

syssnapname = data_names("", "Density", "Velocity", "Temperature", "Electrostatic Potential")
syssnapfluxname = data_names("Flux", "Density", "Velocity", "Temperature")
syssnapintname = data_names("Integral", "Density", "Velocity", "Temperature")


# Plotting
def plot_snaps(data, name, xaxis, yaxis):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for kk in range(len(data)):
        ax.plot(X, data[kk], label="T = " + str(round(kk * dt * Tlngth / snaps, 2)))
    ax.set_title(name + " Γ = " + str(Gamma_0) + " κ = " + str(kappa_0))
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.legend()

    return ax


syssnap_eq, syssnap_cor, syssnap_time, syssnap_lngth = syssnap.shape
syssnapflux_eq, syssnapflux_cor, syssnapflux_time, syssnapflux_lngth = syssnapflux.shape
syssnapint_eq, syssnapint_cor, syssnapint_time = syssnapint.shape

# Plot Fields
# syssnapplots = [[0 for ii in range(syssnap_cor)] for jj in range(syssnap_eq)]
# for ii in range(syssnap_eq):
#     for jj in range(syssnap_cor):
#         syssnapplots[ii][jj] = plot_snaps(syssnap[ii, jj], syssnapname[ii, jj], "Space", syssnapname[ii, jj])

# Plot Fluxes
# syssnapfluxplots = []
# for ii in range(syssnapflux_eq):
#     for jj in range(syssnapflux_cor):
#         plot_snaps(syssnapflux[ii, jj], syssnapfluxname[ii, jj], "Space", syssnapfluxname[ii, jj])


# Subplot: Individual Fields
def subplot(data, name, xaxis, title):
    row, col, plot, _ = data.shape
    fig, axs = plt.subplots(row, col)
    for ii in range(row):
        for jj in range(col):
            for kk in range(plot):
                axs[ii, jj].plot(X, data[ii, jj, kk], label="T = " + str(round(kk * dt * Tlngth / snaps, 2)))
            axs[ii, jj].set_title(name[ii, jj] + " Γ = " + str(Gamma_0) + " κ = " + str(kappa_0))
            axs[ii, jj].set_xlabel(xaxis[ii][jj])
            axs[ii, jj].set_ylabel(name[ii][jj])
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(title)


subplot(syssnap, syssnapname, [["Space" for jj in range(syssnap_cor)] for ii in range(syssnap_eq)], "Fields")

subplot(syssnapflux, syssnapfluxname, [["Space" for jj in range(syssnapflux_cor)] for ii in range(syssnapflux_eq)], "Flux Fields")


# Subplots: Compile Plots Using Axes
def subplot_plots(plots, title):
    row = len(plots)
    col = max(len(plots[ii]) for ii in range(row))
    fig, axs = plt.subplots(row, col)
    for ii in range(row):
        for jj in range(col):
            axs[ii, jj] = plots[ii][jj]
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(title)


# subplot_plots(syssnapplots, "Fields")
#
# subplot_plots(syssnapfluxplots, "Flux Fields")


# Subplot: Color Plots
def subplot_imshow(data, name, xaxis, title):
    row, col, plot, _ = data.shape
    fig, axs = plt.subplots(row, col)
    for ii in range(row):
        for jj in range(col):
            for kk in range(plot):
                axs[ii, jj].plot(X, data[ii, jj, kk], label="T = " + str(round(kk * dt * Tlngth / snaps, 2)))
            axs[ii, jj].set_title(name[ii, jj] + " Γ = " + str(Gamma_0) + " κ = " + str(kappa_0))
            axs[ii, jj].set_xlabel(xaxis[ii][jj])
            axs[ii, jj].set_ylabel(name[ii][jj])
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(title)


# Plot Integrals, Ensure Conservation
def plot_conserved(data, name):
    plt.figure()
    plt.plot(np.arange(snaps), data)
    plt.title(name + " Γ = " + str(Gamma_0) + " κ = " + str(kappa_0))
    plt.xlabel("Time")
    plt.ylabel(name)


# Plots: Conservation of Fields
for ii in range(len(syssnapint)):
    for jj in range(len(syssnapint[ii])):
        plot_conserved(syssnapint[ii, jj], syssnapintname[ii, jj])

# Plots: Integral Difference of Fields Over Time
for ii in range(len(syssnapint)):
    plot_conserved(syssnapint[ii, 0] - syssnapint[ii, 1], "Integral Difference")

# subplot(syssnapint, syssnapintname, [["Time" for jj in range(syssnapint_cor)] for ii in range(syssnapint_eq)])


# ImShow
def imshow(data, name, xaxis, yaxis):
    plt.figure()
    clr = plt.imshow(data, aspect='auto', origin='lower', extent=(0, Xlngth, 0, Tlngth))
    plt.colorbar()
    plt.title(name + " Γ = " + str(Gamma_0) + " κ = " + str(kappa_0))
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)


# Color Plots: Fields
for ii in range(len(syssnap)):
    for jj in range(len(syssnap[ii])):
        imshow(syssnap[ii, jj], syssnapname[ii, jj], "Space", "Time")

# Color Plots: Fields
for ii in range(len(syssnapflux)):
    for jj in range(len(syssnapflux[ii])):
        imshow(syssnapflux[ii, jj], syssnapfluxname[ii, jj], "Space", "Time")

# Color Plots: Differences
for ii in range(len(syssnap)):
    print(np.mean(syssnap))
    imshow((syssnap[ii, 0] - syssnap[ii, 1])/mean_n, "Difference", "Space", "Time")

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
    plt.imshow(np.abs(fft_avg), aspect='auto', origin='lower', extent=(-0.5, snaps - 0.5, -0.5, Xpts - 0.5))
    plt.colorbar()
    plt.xlabel("Spatial Frequency (k)")
    plt.ylabel("Dispersion (ω)")
    plt.title("FFT: Γ = " + str(Gamma_0) + " κ0 = " + str(kappa_0))

end = time.time()
print("Total Time: ", end - start, " seconds")
# print("Off By:", np.round(100*(end - start - (elapsed_time * T / 100))/(end - start), 2), " %")

plt.show()

if __name__ == "__main__":
    print()