import numpy as np
import time

# Space Domain
Xpts = int(1e2)
Xlngth = 10
dx = Xlngth / Xpts
X = np.linspace(0, Xlngth - dx, Xpts)
# Time Domain
Tpts = int(1e2) + 1
Tlngth = 3.4  # .212
T = np.linspace(0, Tlngth, Tpts)[:-1]
dt = Tlngth / (Tpts - 1)
# Snapshots
snaps = 100 + 1
cursnap = 0
Y = np.linspace(0, Tlngth, snaps)
xx, yy = np.meshgrid(X, Y, sparse=False, indexing='xy')
# Spatial Frequency Domain
k_fft_norm = 2 * np.pi / (Xpts * dx)
k = k_fft_norm * np.linspace(-Xpts / 2, Xpts / 2 - 1, Xpts)
# Spatial Frequency Domain
# freq_vals = np.fft.fftfreq(nx, dx)
# print(freq_vals)
# print(np.fft.fftshift(freq_vals))
# print(k/k_fft_norm)
# Initializations
mean_n = 3/(4*np.pi)
den = mean_n*np.ones((2, Xpts)) + .1*np.random.random((2,Xpts))
den = den[1]
temp = np.ones((2, Xpts))  # Set to 1 for beta calculation in correlation
q = 1
Gamma = 1
kappa = 1

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
u = Gamma / Rp * np.exp(-kappa/Rp)  # (500,)

# Define c(|r'|)
e_ext = np.tile(temp, num_dom_ext)
c = np.exp(-u / e_ext[1]) - 1 + u / e_ext[1]
c_fft = np.fft.fftn(c)

# Integrate rho'*F(c(r')) drho'
r_integrand = Rp[np.newaxis, :] * c[np.newaxis, :] * np.sinc(Kp[:, np.newaxis] /np.pi * Rp[np.newaxis, :])
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
print(corr.shape)

############################
# O(n^3) Direct Evaluation #
############################

# st = time.time()
# # Integration Domain/Extensions
# num_dom_ext = 5
# xp_min = yp_min = zp_min = -int(num_dom_ext/2)
# xp_max = yp_max = zp_max = int(num_dom_ext/2 + 1)
# Xppts = int(num_dom_ext * nx)
# Xp_ar = np.linspace(xp_min, xp_max, Xppts)
# Yp_ar = np.linspace(yp_min, yp_max, Xppts)
# Zp_ar = np.linspace(zp_min, zp_max, Xppts)
# Xp, Yp, Zp = np.meshgrid(Xp_ar, Yp_ar, Zp_ar)
# Rp = np.sqrt(Xp**2 + Yp**2 + Zp**2)  # Define |r'|
#
# u = q**2 / Rp  # Define Potential u
#
# # Define c(|r'|)
# e_ext = np.tile(e, num_dom_ext)
# c = np.exp(-u/e_ext[1]) - 1 + u/e_ext[1]
#
# # n(x-x') - mean_n
# delta_n_X_Xp = np.tile(n[1], num_dom_ext) - mean_n   # Shape: (nx, Xppts)
# delta_n_X_Xp = np.stack([delta_n_X_Xp] * Xppts)
# delta_n_X_Xp = np.stack([delta_n_X_Xp] * Xppts)
#
# def meyer_correlation(den, corr):
#     res_fft = np.fft.fftn(den)*np.fft.fftn(corr)
#     result = np.fft.ifft(res_fft)
#     return result.real
#
# correlations = meyer_correlation(delta_n_X_Xp,c)
# print(correlations)
#
# end = time.time()
# print(end-st)


############################
# O(n^4) Direct Evaluation #
############################
# # Integration Domain/Extensions
# num_dom_ext = 5
# xp_min = yp_min = zp_min = -int(num_dom_ext/2)
# xp_max = yp_max = zp_max = int(num_dom_ext/2 + 1)
# Xppts = int(num_dom_ext * nx)
# Xp_ar = np.linspace(xp_min, xp_max, Xppts)
# Yp_ar = np.linspace(yp_min, yp_max, Xppts)
# Zp_ar = np.linspace(zp_min, zp_max, Xppts)
# Xp, Yp, Zp = np.meshgrid(Xp_ar, Yp_ar, Zp_ar)
# Rp = np.sqrt(Xp**2 + Yp**2 + Zp**2)  # Define |r'|
#
# u = q**2 / Rp  # Define Potential u
#
# # Define c(|r'|)
# e_ext = np.tile(e, num_dom_ext)
# c = np.exp(-u/e_ext[1]) - 1 + u/e_ext[1]
#
# # n(x-x') - mean_n
# delta_n_X_Xp = np.subtract.outer(np.tile(n[1], num_dom_ext), n[1]) - mean_n   # Shape: (nx, Xppts)
# delta_n_X_Xp = np.stack([delta_n_X_Xp] * Xppts)
# delta_n_X_Xp = np.stack([delta_n_X_Xp] * Xppts)
#
# def meyer_correlation(den, corr):
#     integrand = den * corr[:,:,:, np.newaxis]
#     first_int = np.trapz(integrand, axis=0, dx=dx)  # Integrate over x'
#     second_int = np.trapz(first_int, axis=0, dx=dx)  # Integrate over y'
#     third_int = np.trapz(second_int, axis=0, dx=dx)  # Integrate over z'
#     return third_int
#
# correlations = meyer_correlation(delta_n_X_Xp,c)


# from scipy.fft import fft, fftfreq
#
# def fourier_transform(x, y):
#     """Perform Fourier transform on y(x) and return frequency and amplitude arrays."""
#     # Calculate Fourier coefficients
#     y_fft = fft(y)
#     # Calculate corresponding frequencies
#     freq = fftfreq(len(y), d=(x[1]-x[0]))
#     # Shift frequencies to be centered at zero
#     y_fft_shift = np.fft.fftshift(y_fft)
#     freq_shift = np.fft.fftshift(freq)
#     # Scale Fourier coefficients by 1/N to match mathematical definition
#     y_fft_scaled = y_fft_shift / len(y)
#     return freq_shift, y_fft_scaled
#
# # Define example function to transform
# x = np.linspace(-5, 5, 1000)
# y = np.exp(-(x**2)/2)
#
# # Perform Fourier transform
# freq, y_fft = fourier_transform(x, y)
#
# # Plot results
# import matplotlib.pyplot as plt
# plt.subplot(121)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Input Function')
# plt.subplot(122)
# plt.plot(freq, np.abs(y_fft))
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.title('Fourier Transform')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the function to transform
# def my_func(a,x, y):
#     return np.exp(-a*(x**2 + y**2))
#
# # Define the range of x and y values
# x_min = y_min = -5
# x_max = y_max = 5
# num_points = 100
# x_vals = np.linspace(x_min, x_max, num_points)
# y_vals = np.linspace(y_min, y_max, num_points)
# X, Y = np.meshgrid(x_vals, y_vals)
# a = 3
#
# # Evaluate the function at each point in the grid
# f = my_func(a, X, Y)
#
# # Calculate the 2D Fourier transform of the function
# F_fourier = np.fft.fft2(f)
# F_fourier = 1/num_points*F_fourier
#
# # Calculate the frequency values
# dx = dy = x_vals[1] - x_vals[0]
# freq_vals = np.fft.fftfreq(num_points, dx)
#
# # Reorder the Fourier modes to match the mathematical convention
# F_fourier = np.fft.fftshift(F_fourier)
# freq_vals = np.fft.fftshift(freq_vals)
#
# # Evaluate the Fourier transform using the analytic expression
# KX, KY = np.meshgrid(freq_vals, freq_vals)
# F_analytic = np.sqrt(np.pi/a) * np.exp(-(KX**2 + KY**2) / a)
# F_analytic = F_analytic
#
# # Create a 3D surface plot of the magnitude of the Fourier transform
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# Xf, Yf = np.meshgrid(freq_vals, freq_vals)
# ax.plot_surface(Xf, Yf, np.abs(F_fourier), rstride=1, cstride=1, cmap='plasma', shade=False)
# ax.plot_surface(Xf, Yf, np.abs(F_analytic), rstride=1, cstride=1, cmap='viridis', shade=False)
# ax.set_xlabel('Frequency in x')
# ax.set_ylabel('Frequency in y')
# plt.show()


