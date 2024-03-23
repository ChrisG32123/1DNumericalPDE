import matplotlib.pyplot as plt
import numpy as np

def disp_rel():
    Gamma = [10,1,.1]
    kappa = [.5,1,2]

    # Dispersion Relations

    N = int(5e2)  # Grid Points
    T = int(3e3)  # Time Steps
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    k = np.linspace(0, L - L / N, N)
    lambda_ = dt / dx
    t = dt * T
    disp_freq = 3 * 2 * np.pi / L
    correlations = False

    omegas = []
    omegas_c = []
    for elmG in Gamma:
        for elmk in kappa:
            omega = np.sqrt((k ** 2) + 3 * elmG)
            omega_c = np.sqrt((1 + 3 * elmG / (k ** 2 + elmk ** 2)) * (k ** 2) + 3 * elmG)

            omegas.append(omega)
            omegas_c.append(omega_c)

            # plt.plot(k, omega, label='No Correlations Gamma = ' + str(elmG) + ' kappa = ' + str(elmk))
            plt.plot(k, omega_c, label='Correlations, Γ = ' + str(elmG) + ', κ = ' + str(elmk))
            plt.xlabel('k')
            plt.ylabel('ω(k)')
            plt.title('Dispersion Relations')

    plt.xlim(0,9)
    plt.ylim(0,9)
    plt.legend()
    plt.show()
