import numpy as np
import matplotlib.pyplot as plt

def poisson():
    # Define domain
    N = int(1e3)
    L = 10
    x = np.linspace(0,L,N)
    dx = x[2]-x[1]

    # Memory allocation
    phi = np.zeros(N)
    c = np.ones(N)

    # Charge initial density
    chooseIC = 2
    if (chooseIC == 1):
        n = np.sin(2*np.pi*(x-dx)/L)
        n = n - 3 / (4 * np.pi) #np.mean(n) to represent correctly
        phiExact = L*L*n/np.pi
    elif (chooseIC == 2):
        n = x*(L-x) - L*L/6
        n = n - 3 / (4 * np.pi) #np.mean(n) to represent correctly
        phiExact = np.pi*x*x*(x-L)**2/3

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Modified Thomas algorithm %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Define f(x)
    f = - 4 * np.pi * dx * dx * n

    # First sweep
    c[1] = -0.5
    f[1] = -0.5*f[1]
    for ii in range (2,N):
        c[ii] = -1/(2 + c[ii-1])
        f[ii] = (f[ii-1]-f[ii])/(2 + c[ii-1])

    # Second sweep
    phi[1] = f[N-1]-f[N-2]
    for ii in range (1,N):
        phi[ii] = (f[ii-1]-phi[ii-1])/c[ii-1]

    #%%%%%%%%%%%
    # Plotting %
    #%%%%%%%%%%%

    plt.plot(x, phi, label='phi')
    plt.plot(x, phiExact, label='phiExact')
    plt.title("Poisson Solver")
    plt.legend()
    plt.show(block=True)
    plt.interactive(False)