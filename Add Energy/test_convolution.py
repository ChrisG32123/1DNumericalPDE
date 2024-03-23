import numpy as np
import matplotlib.pyplot as plt
import time

def test_meyer_correlations():
    print()

    # Domain
    dx = .1
    Xpts = int(2e2)
    Xlngth = int(dx * Xpts)
    X = dx * np.arange(0, Xpts)
    num_dom_ext = 5
    Xppts = int(num_dom_ext * Xpts)
    Wlngth = 10
    Wpts = 100
    dw = Wlngth / Wpts
    Xp = np.linspace(-int((Xpts - 1) / 2), int((Xpts + 1) / 2), Xppts)  # grid points in x'
    Wp = np.linspace(0, Wlngth, Wpts) + dw  # grid points in w'; right endpoint  theoretically goes to infinity

    mean_n = 3/(4*np.pi)
    q = 1

    freq = 2*np.pi/Xlngth
    den = .5*np.ones(Xpts) + .1*np.sin(freq*X)
    temp = .5*np.ones(Xpts) + .1*np.cos(freq*X)

    def l(array):
        return np.roll(array, 1, axis=-1)

    def ddx(array):
        return (array - l(array))/dx

    # Periodic Function Extensions - Extend x to x'
    n_p = np.tile(den, num_dom_ext)
    beta_p = np.tile(1 / temp, num_dom_ext)

    # Extend x and x' to (x,x')
    n_x_p = np.subtract.outer(den, n_p) - mean_n
    beta_x_p = np.stack([beta_p] * Xpts, axis=0)

    # Extend (x,x') to (x,x',w')
    n_p_wp = np.stack([n_x_p] * Wpts, axis=1)
    beta_p_wp = np.stack([beta_x_p] * Wpts, axis=1)

    # Define Potential and Correlation Integrand
    u_potential = beta_p_wp * (1 / Wp)[:, None]
    corr_integrand = (1 - np.exp(-q ** 2 * u_potential)) * (ddx(beta_p_wp) - u_potential * Xp)

    # print(ddx(beta_p_wp) - u_potential*Xp)

    # Calculate full integral
    corr_integrand = n_p_wp * corr_integrand
    first_int = np.trapz(corr_integrand, axis=-1, dx=dx)  # Integrate over x'
    res = np.trapz(first_int, axis=-1, dx=dw)  # Integrate over w'

    # Check if integrand has Nan values
    # print(np.argwhere(np.isnan(corr_integrand)))

    print(res)

    return res


# def test_thomas_algorithm():
#
#     print()
#
#     nx = int(1e2)
#     Xlngth = 10
#     dx = Xlngth / nx
#     X = np.linspace(0, Xlngth - dx, nx)
#
#     print(X)
#
#     dx = .1
#     nx = int(1e2)
#     Xlngth = int(dx * nx)
#     X = dx*np.arange(0,nx)
#
#     print(X)
#
#     d = np.sin(X)
#
#     a = np.ones(nx-1)
#     b = -2*np.ones(nx)
#     c = np.ones(nx-1)
#
#     # thom_alg_param = 1
#     # b[0] += -thom_alg_param
#     # b[-1] += -a[0] * c[-1] / thom_alg_param
#     #
#     # B = np.diag(b) + np.diag(a, k=-1) + np.diag(c, k=1)
#     #
#     # print(B)
#
#     # a = np.diag(A, k=-1)
#     # b = np.diag(A)
#     # c = np.diag(A, k=1)
#
#     c_prime = np.zeros(nx - 1)
#     d_prime = np.zeros(nx)
#
#     c_prime[0] = c[0] / b[0]
#     d_prime[0] = d[0] / b[0]
#
#     c_prime[1:nx-1] = c[1:nx-1] / (b[1:nx-1] - a[0:nx - 2] * c_prime[0:nx - 2])
#     d_prime[1:nx-1] = (d[1:nx-1] - a[0:nx - 2] * d_prime[0:nx - 2]) / (b[1:nx-1] - a[0:nx - 2] * c_prime[0:nx - 2])
#
#     sol = np.zeros(nx)
#     sol[-1] = d_prime[-1]
#     sol[nx - 1::-1] = d_prime[nx - 1::-1] - c_prime[nx - 1::-1] * sol[nx:0:-1]
#
#     for i in range(nx - 2, -1, -1):
#         sol[i] = d_prime[i] - c_prime[i] * sol[i + 1]
#
#     return sol

# def test_integration():
#     print()
#
#     st = time.time()
#     nx = 100  # number of grid points
#     Wpts = 300
#     num_dom_ext = 5
#     Xppts = int(num_dom_ext*nx)
#     q = 1
#
#     X = np.linspace(0, 10, nx)  # grid points in x
#     Xp = np.linspace(-int((nx-1)/2), int((nx+1)/2), Xppts)  # grid points in x'
#     Wp = np.linspace(1/Wpts, 30, Wpts)  # grid points in w'; right endpoint  theoretically goes to infinity
#
#     def ddx_stand_in(array):  # Stand in for ddx() function in main code
#         return array
#
#     # Memory Allocation
#     integrand = np.ones((nx, Xppts, Wpts))
#     betap = np.ones(Xppts)
#
#     outer = np.outer(betap*q**2, 1/Wp)
#     broadcast = q**2*betap[..., None]*(1/Wp[None, :])
#     broadcast = np.repeat(broadcast, nx)
#
#     # For each x, evaluate I(x',w')
#     for ii in range(nx):
#         integrand[ii] = (np.ones((Xppts, Wpts)) - np.exp(-np.outer(betap*q**2, 1/Wp))) * (np.vstack([ddx_stand_in(betap)]*Wpts).T - np.outer(betap*Xp, 1/Wp))
#         # TODO: At I[ii,Wp=0], e^-1/Wp = 0
#
#     # Shape:(nx)
#     n = np.arange(nx)  # density
#     # print(n)
#
#     # Shape:(num_dom_ext*nx)
#     n_ext = np.tile(n, num_dom_ext)  # Extend n array from x to x' domain
#     # print(n_ext)
#
#     # Create delta n(x-x')
#     # Shape:(nx, num_dom_ext*nx)
#     delta_n = np.subtract.outer(n, n_ext)  # For each correlation: delta n(x-x')[cor] = delta n(x[ii]-x');
#     # print(delta_n)
#
#     # Memory Allocation:
#     # Extend delta_n(x-x') to w' domain. Constant along w' axis, but needed to match shape of I(x',w').
#     delta_n_ext = np.zeros((nx, num_dom_ext*nx, Wpts))
#
#     # Shape: (nx, num_dom_ext*nx, Wpts)
#     for x in range(len(delta_n_ext)):
#         delta_n_ext[x] = np.vstack([delta_n[x]] * Wpts).T
#
#     full_integrand = delta_n_ext * integrand  # Hadamard Product: delta n(x-x')*I(x'w')
#
#     # Integrate x' first, then w'
#     res = np.trapz(np.trapz(full_integrand, axis=-2), axis=-1) # TODO: Add dx= according to np.trapz and Wp array
#     print(res)
#
#     et = time.time()
#
#     print(et-st)

# def test_integration():
#     print()
#
#     N = 5  # number of grid points
#
#     print(-int((N-1)/2))
#     print(int((N+1)/2))
#
#     n = np.arange(N)  # density
#     n_ext = np.tile(n, 5)  # n repeated 5 times
#     xx = np.linspace(0, 1, N)  # grid points in x
#     xx_ext = np.linspace(-int((N-1)/2), int((N+1)/2), (N * 5))  # grid points in x
#     yy_ext = np.linspace(-int((N-1)/2), int((N+1)/2), (N * 5))  # grid points in x
#     zz_ext = np.linspace(-int((N-1)/2), int((N+1)/2), (N * 5))  # grid points in x
#
#     def I(x, x_p, y_p, z_p):
#         # does something!
#
#         return np.ones((len(x), len(x_p), len(y_p), len(z_p)))
#
#     temp_I = I(xx, xx_ext, yy_ext, zz_ext)  # shape: index X xp X yp X zp
#     print("temp_I", temp_I.shape)
#
#     delta_n = np.subtract.outer(n, n_ext)  # shape: N x 5*N (index for x-x', x' domain)
#     print("delta_n", delta_n.shape)
#     print(delta_n)
#
#     print("n_ext", n_ext.shape)
#     print(n_ext)
#
#     n_temp = np.vstack([n_ext] * len(n_ext)).T
#     print("n_temp", n_temp.shape)
#
#     n_integrand = np.stack([n_temp] * len(n_ext))
#     print("n_integrand", n_integrand.shape)
#
#     full_integrand = n_integrand * temp_I
#
#     res = np.trapz(np.trapz(np.trapz((full_integrand), axis=1), axis=1), axis=1)
#     print("res", res.shape)


    # def extend_integration_domain(arr, integration_domain):
    #     arr_prime = np.tile(arr, num_domain_extend)
    #     for domain in integration_domain:
    #         arr_prime = np.stack([arr_prime for _ in range(len(domain))], axis=-1)
    #         print(n_prime.shape)
    #     return arr_prime
    #
    # n_prime = extend_integration_domain(n,integration_domain)

    # def extend_to_integration_domain(array, *domain):
    #     # Extend Original Domain By num_domain_extend
    #     array_prime = np.tile(array, num_domain_extend)
    #     print(array.shape)
    #     print(array_prime.shape)
    #
    #     # Iterate over the number of arrays in the integration domain
    #     for ii in range(len(domain)):
    #         # Extend To Each Axis
    #         array_prime = np.vstack([array_prime] * len(domain[ii]))
    #         print(array_prime.shape)
    #     return array_prime
    #
    #
    # integration_domain = np.array([Xp, Y_prime, Z_prime])
    # n_prime = extend_to_integration_domain(n, integration_domain)
    # print(n_prime.shape)
    #
    # n_prime_int = np.trapz(n_prime, x= Xp)
    # n_prime_int = np.trapz(n_prime, x= Y_prime)
    # n_prime_int = np.trapz(n_prime, x= Z_prime)
    #
    # print(n_prime_int.shape)
    #
    # # Time Domain
    # Tpts = int(1e3) + 1
    # Tlngth = 10
    # T = np.linspace(0, Tlngth, Tpts)[:-1]
    # dt = Tlngth / (Tpts - 1)
    #
    # lmbd = dt / dx
    #
    # # Snapshots
    # snaps = 100 + 1
    # cursnap = 0
    # Y = np.linspace(0, Tlngth, snaps)
    # xx, yy = np.meshgrid(X, Y, sparse=False, indexing='xy')





# def test_args():
#     print()
#     ar = np.arange(24).reshape(3, 2, 4)
#     ar2 = np.arange(24).reshape(3, 2, 4) + 24
#     ar3 = np.arange(24).reshape(3, 2, 4) + 48
#
#     def store_to_visualize(*args):
#         sys_data_to_visualize = np.array([np.swapaxes(array, 0, 1) for array in args])
#         return sys_data_to_visualize
#
#     sys = store_to_visualize(ar, ar2)
#     sys2 = store_to_visualize(ar, ar2, ar3)
#
#     print("ar", ar)
#     print("swap ar", np.swapaxes(ar, 0, 1))
#     print("sys", sys)
#     print("sys2", sys2)
#
#     print(sys.shape)
#     print(sys2.shape)
#
#     def data_names(var_name, *args):
#         cornames = ["NC", "C"]
#         if var_name.strip():
#             var_name = " " + var_name + " "
#         names = [[name + var_name + corname for corname in cornames] for name in args]
#         return names
#
#     syssnapnames = data_names("", "Density", "Velocity", "Energy", "Electrostatic Potential")
#     syssnapfluxnames = data_names("Flux", "Density", "Velocity", "Energy", "Electrostatic Potential")
#     print(syssnapnames)
#     print(syssnapfluxnames)

# def test_roll():
#     array = np.arange(24).reshape(3,2,4)
#     array2 = np.arange(24).reshape(3,2,4)
#     print()
#
#     def l(array):
#         return np.roll(array, 1, axis=-1)
#
#     def r(array):
#         return np.roll(array, -1, axis=-1)
#
#     for ii in range(3):
#         print("a", array[0])
#         print("r", -r(array[0]))
#         print("l", -l(array2[0]))
#         array = array - r(array) - l(array2)
#         print(ii, array[0])

# def test_fourier():
#     # Parameters
#     nx = 2 ** 10
#     M = 2 ** 10
#     Xlngth = 10
#     K = 10
#     a = 100
#     b = 50
#     Gamma = 1
#     kappa = 1
#
#     # Define domains
#     dx = Xlngth / nx
#     dy = K / M
#     X = np.linspace(0, Xlngth - dx, nx)
#     y = np.linspace(0, K - dy, M)
#
#     f_fft_norm = 1 / dx
#     k_fft_norm = 2 * np.pi / (nx * dx)
#     k = k_fft_norm * np.linspace(-nx / 2, nx / 2 - 1, nx)
#
#     gaus = np.exp(-(X-Xlngth/2)**2)
#     fft = np.fft.fft(gaus)
#     fft = np.fft.fftshift(fft) * 1/(np.sqrt(np.pi)*f_fft_norm)
#
#     plt.figure()
#     plt.plot(X,gaus)
#
#     plt.figure()
#     plt.plot(k,np.abs(fft))
#
#     # plt.figure()
#     # xx, yy = np.meshgrid(X,y)
#     #
#     # gaus2d = np.exp(-(xx-Xlngth/2)**2)*np.exp(-(yy-K/2)**2)
#     #
#     # plt.imshow(gaus2d, origin="lower")
#
#     plt.show()
#

# Other Test Methods
# def test_data_flip():
#
#     print()
#     data = np.arange(60).reshape(2, 5, 6)
#     print("data", data)
#
#     length0, length1, length2 = data.shape
#     data1_1st_half, data1_2nd_half = data[:,:int(length1/2+1)], data[:, int(length1/2):]
#     reflect1 = (data1_1st_half + np.flip(data1_2nd_half, axis=1))/2
#     print("reflect1_halves", reflect1)
#
#     reflect1_1st_half, reflect1_2nd_half = reflect1[:, :, :int(length2 / 2)], reflect1[:, :, int(length2 / 2):]
#     reflect0_halves = (reflect1_1st_half + np.flip(reflect1_2nd_half, axis=2)) / 2
#     print("reflect10", reflect0_halves)
#
#     data0_1st_half, data0_2nd_half = data[:, :, :int(length2 / 2)], data[:, :, int(length2 / 2):]
#     reflect0_halves = (data0_1st_half + np.flip(data0_2nd_half, axis=2)) / 2
#     print("reflect0_halves", reflect0_halves)
#
#     data_flip = np.flip(data, axis=1)
#     print("data-flip",data_flip)
#
#     newdata = np.hstack((data, data_flip))
#     print("new-data", newdata)
#
# def testfft():
#     start = time.time()
#
#     # Parameters
#     nx = int(1e3)  # Grid Points
#     T = int(4e3)  # Time Steps
#     Xlngth = 10  # Domain Size
#     X = np.linspace(0, Xlngth - Xlngth / nx, nx)  # Domain
#     dx = X[2] - X[1]  # Grid Size
#     dt = 1e-3  # Time Step Size
#     lmbd = dt / dx
#     t = dt * T
#
#     mean_n = 3 / (4 * np.pi)
#     Gamma_0 = 1
#     kappa_0 = 1
#     therm_cond = .01
#
#     snaps = 1000
#     cursnap = 0
#     y = np.linspace(0,t,snaps)
#
#     IC_freqx = 2 * np.pi / Xlngth
#     IC_freqy = 2 * np.pi / t
#
#     sin = np.sin(3*IC_freqx*X) + np.random.random(nx) - .5*np.ones(nx)
#
#     xx,yy = np.meshgrid(X,y)
#
#     def imshow(array2d, name):
#         plot = plt.imshow(array2d, origin="lower", label=name)
#         return plot
#
#     def fft2(array2d):
#         fft = np.fft.fft2(array2d)
#         fft = np.fft.fftshift(fft)
#         return fft
#
#     def ifft2(fft2d):
#         fft2d = np.fft.ifftshift(fft2d)
#         ifft = np.fft.ifft2(fft2d)
#         return ifft
#
#     names = np.array(["v", "h", "d", "c"])
#     data = np.array((4,nx,snaps))
#     fftdata = np.array((4,nx,snaps))
#     ifftdata = np.array((4,nx,snaps))
#     difdata = np.array((4,nx,snaps))
#
#     data[0] = np.sin(3 * IC_freqx * xx) + np.random.random((nx, snaps)) - .5 * np.ones((nx, snaps))
#     data[1] = np.sin(3 * IC_freqy * yy) + np.random.random((nx, snaps)) - .5 * np.ones((nx, snaps))
#     data[2] = np.sin(np.pi / 4 * (3 * IC_freqx * xx + 3 * IC_freqy * yy)) + np.random.random((nx, snaps)) - .5 * np.ones((nx, snaps))
#     data[3] = sum(sin for sin in data[0])
#
#     fftdata = np.array([fft2(sin) for sin in data])
#     ifftdata = np.array([ifft2(fft) for fft in fftdata])
#     difdata[:] = np.abs(data[:] - ifftdata[:])
#
#     fig, ax = plt.subplots(nrows=4,ncols=4)
#     ax[0,:] = imshow(data[:], names[:])
#     ax[1,:] = imshow(fftdata[:], names[:])
#     ax[2,:] = imshow(ifftdata[:], names[:])
#     ax[3,:] = imshow(difdata[:], names[:])
#
#     plt.show()
#     end = time.time()
#     print("Total Time: ", end - start, " seconds")
#
#
# def test_convolution():
#     # Parameters
#     nx = 2**10
#     Xlngth = 100
#     a = 0.1
#
#     # Define domains
#     dx = 2*Xlngth/nx
#     X = np.linspace(-Xlngth,Xlngth,nx+1)
#     k_fft_norm = 2*np.pi/(nx*dx)
#     f_fft_norm = nx/(2*Xlngth)
#     k = k_fft_norm * np.linspace(-nx/2,nx/2,nx+1)
#
#     # FFT solution
#     f = np.exp(-a*X*X)
#     fhat = np.fft.fftshift(np.abs(np.fft.fft(f)))
#
#     # Exact solution
#     f_ex = f_fft_norm * np.sqrt(np.pi/a)*np.exp(-0.25*k*k/a)
#
#     # Plotting
#     plt.plot(k,f_ex,k,fhat)
#     plt.show()
#
# def half_interval_test_convolution():
#     # Parameters
#     nx = 2 ** 10
#     Xlngth = 10
#     a = 100
#     Gamma = 1
#     kappa = 1
#
#     def dcf(k, Gamma, kappa):
#         return 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)
#
#     # Define domains
#     dx = Xlngth / nx
#     X = np.linspace(0, Xlngth, nx + 1)
#     f_fft_norm = 1 / dx
#     k_fft_norm = 2 * np.pi / (nx * dx)
#     k = k_fft_norm * np.linspace(-nx / 2, nx / 2 - 1, nx)
#
#     # FFT solution
#     f = np.sqrt(a / np.pi) * np.exp(-a * (X - (Xlngth/2)) ** 2)
#     fhat = np.fft.fftshift(np.fft.fft(f))
#
#     # Exact solution
#     #f_ex = f_fft_norm * np.sqrt(np.pi / a) * np.exp(-0.25 * k * k / a)
#
#     conv = fhat * dcf(k,Gamma, kappa)
#     conv = np.fft.ifft(np.fft.ifftshift(conv))
#
#     conv_ex = 2 * np.pi * Gamma / kappa * np.exp(-kappa * np.abs(X-(Xlngth/2)))
#
#     # Plotting
#     # plt.plot(k,f_ex, label="exact")
#     # plt.plot(k,fhat, label="fft")
#     plt.plot(X,f, label="f(X)")
#     plt.plot(X,conv, label="convolution")
#     plt.plot(X,conv_ex, label="conv_ex")
#     plt.legend()
#     plt.show()
#
# def meyer_convolution():
#     # Parameters
#     nx = 2 ** 10
#     Xlngth = 10
#     a = 100
#     b = 50
#     Gamma = 1
#     kappa = 1
#
#     def dcf(k, Gamma, kappa):
#         return 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)
#
#     # Define domains
#     dx = Xlngth / nx
#     X = np.linspace(0, Xlngth - dx, nx)
#     f_fft_norm = 1 / dx
#     k_fft_norm = 2 * np.pi / (nx * dx)
#     k = k_fft_norm * np.linspace(-nx / 2, nx / 2 - 1, nx)
#
#     # Find c_hat(k) = 4pi integral 0 to infinity c(r)r^2 sinc(kr) dr
#     # Parameters
#     Nr = int(1e2)
#     c_hat = np.zeros(nx)
#     beta = 1  # TODO: = 1 / (kB*T) (Nondimensionalized?)
#
#     for ii in range(int(nx/2 + 1)):
#         rmax = 100  # TODO: Change per loop
#         r = np.linspace(0, rmax, Nr)
#         r[0] = 1
#         u = Gamma / r * np.exp(-r)
#         # c = np.exp(-beta*u) - 1
#         # c[0] = -1
#         r[0] = 0
#         b = 1
#         c = np.exp(-b * r ** 2)
#         c_hat[ii] = 4 * np.pi * np.trapz((r**2)*c*np.sinc(k[ii]*r/np. pi), r)  # np.sinc(X) = sinc(pi*X)/(pi*X)
#
#     for ii in range(int(nx/2 - 1)):
#         c_hat[nx-1-ii] = c_hat[ii]
#
#     c_hat_ex = (np.pi / b) ** (3/2) * np.exp(- k ** 2 / (4*b))
#
#     # FFT solution 1
#     f = np.sqrt(a / np.pi) * np.exp(-a * (X - (Xlngth/2)) ** 2)
#     fhat = np.fft.fftshift(np.fft.fft(f))
#
#     conv_ex = np.pi / b * np.exp(-b * (X - Xlngth/2) ** 2)
#
#     conv = c_hat * fhat
#     conv = np.fft.ifft(np.fft.ifftshift(conv))
#
#     # Exact solution
#     #f_ex = f_fft_norm * np.sqrt(np.pi / a) * np.exp(-0.25 * k * k / a)
#
#     # conv = fhat * ghat
#     # conv = np.fft.ifft(np.fft.ifftshift(conv))
#     #
#     # conv_ex = 2 * np.pi * Gamma / kappa * np.exp(-kappa * np.abs(X-(Xlngth/2)))
#
#     # Plotting
#     # plt.plot(k,f_ex, label="exact")
#     # plt.plot(k,fhat, label="fft")
#     plt.plot(X,f, label="f(X)")
#     plt.plot(X,conv, label="convolution")
#     plt.plot(X,conv_ex,label = "conv_ex")
#     # plt.plot(X,conv_ex, label="conv_ex")
#     plt.legend()
#
#     plt.figure()
#     plt.plot(k,np.abs(c_hat), label="c_hat")
#     plt.plot(k,c_hat_ex, label="c_hat_ex")
#     # plt.loglog(k[int(nx/2 + 1):int(nx+1)], c_hat[int(nx/2 + 1):int(nx+1)])
#     plt.legend()
#
#     plt.show()
#
#     #
#     # plt.figure()
#     # plt.plot(r,c, label="c(X)")
#     # plt.legend()
#     # plt.show()
#
# def meyer_convolution_1():
#     # Parameters
#     nx = 2 ** 10
#     Xlngth = 10
#     a = 100
#     b = 50
#
#     # Define domains
#     dx = Xlngth / nx
#     X = np.linspace(0, Xlngth - dx, nx)
#     f_fft_norm = 1 / dx
#     k_fft_norm = 2 * np.pi / (nx * dx)
#     k = k_fft_norm * np.linspace(-nx / 2, nx / 2 - 1, nx)
#
#     # Parameters
#     Nr = int(1e3)
#     rmax = 100  # TODO: Change per loop
#     r = np.linspace(0, rmax, Nr)
#
#     # FFT Function 1
#     f = np.exp(-b * r ** 2)
#     fhat = np.fft.fftshift(np.fft.fft(f))
#     f_hat_ex = (np.pi / b) ** (3/2) * np.exp(- k ** 2 / (4*b))
#
#     # FFT Function 2
#     g = np.sqrt(a / np.pi) * np.exp(-a * (X - (Xlngth/2)) ** 2)
#     ghat = np.fft.fftshift(np.fft.fft(g))
#     g_hat_ex = np.exp(- k ** 2 / (4 * a))
#
#     conv = f_hat_ex * ghat
#     conv = np.fft.ifft(np.fft.ifftshift(conv))
#
#     # conv_fft = f_hat_ex * g_hat_ex
#     # conv_fft = np.fft.ifft(np.fft.ifftshift(conv_fft))
#
#     normalization = np.sqrt(2)/3
#     conv_ex = (np.pi/b) * np.sqrt((a+b)/b) * np.exp(-(a+b) * (normalization**2) * (X - Xlngth / 2) ** 2)
#     conv_ex = conv_ex * normalization
#
#     # Plotting
#     plt.plot(X,conv, label="Code Convolution = conv")
#     # plt.plot(X,conv_fft, label="FFT Convolution = conv_fft")
#     plt.plot(X,conv_ex,label = "Exact Convolution = conv_ex")
#     plt.legend()
#     plt.show()
#
# def meyer_convolution_2():
#     # Parameters
#     nx = 2 ** 10
#     Xlngth = 10
#     a = 100
#     b = 50
#
#     # Define domains
#     dx = Xlngth / nx
#     X = np.linspace(0, Xlngth - dx, nx)
#     f_fft_norm = 1 / dx
#     k_fft_norm = 2 * np.pi / (nx * dx)
#     k = k_fft_norm * np.linspace(-nx / 2, nx / 2 - 1, nx)
#
#     # Parameters
#     Nr = int(1e3)
#     rmax = 100  # TODO: Change per loop
#     r = np.linspace(0, rmax, Nr)
#
#     # FFT Function 1
#     f = np.exp(-b * r ** 2)
#     fhat = np.fft.fftshift(np.fft.fft(f))
#     f_hat_ex = (np.pi / b) ** (3/2) * np.exp(- k ** 2 / (4*b))
#
#     # FFT Function 2
#     g = np.sqrt(a / np.pi) * np.exp(-a * (X - (Xlngth/2)) ** 2)
#     ghat = np.fft.fftshift(np.fft.fft(g))
#     g_hat_ex = np.exp(- k ** 2 / (4 * a))
#
#     conv = f_hat_ex * ghat
#     conv = np.fft.ifft(np.fft.ifftshift(conv))
#
#     # conv_fft = f_hat_ex * g_hat_ex
#     # conv_fft = np.fft.ifft(np.fft.ifftshift(conv_fft))
#
#     normalization = np.sqrt(2)/3
#     conv_ex = (np.pi/b) * np.sqrt((a+b)/b) * np.exp(-(a+b) * (normalization**2) * (X - Xlngth / 2) ** 2)
#     conv_ex = conv_ex * normalization
#
#     # Plotting
#     plt.plot(X,conv, label="Code Convolution = conv")
#     # plt.plot(X,conv_fft, label="FFT Convolution = conv_fft")
#     plt.plot(X,conv_ex,label = "Exact Convolution = conv_ex")
#     plt.legend()
#     plt.show()

if __name__ == "__main__":
    test_meyer_correlations()