import matplotlib.pyplot as plt
import definitions_domain_parameters_functions


class plot(definitions_domain_parameters_functions):
    def plot():
        # Plotting
        for ii in range(len(snap_n)):
            plt.plot(x, snap_n[ii], label="n @ T = " + str(ii * dt * T / snaps))
        plt.title("Density: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(x, n_IC, label="nc @ T = 0")
        print(n_IC)
        for ii in range(len(snap_n)):
            plt.plot(x, snap_nc[ii], label="nc @ T = " + str(ii * dt * T / snaps))
        plt.title("Density: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
        plt.legend()
        plt.show()

        #
        # # Color Map
        # y = np.linspace(0, t, snaps + 1)
        # xx,yy = np.meshgrid(x, y, sparse=False, indexing='xy')
        #
        # plt.figure()
        # clr = plt.contourf(xx, yy, snap_n)
        # plt.title("Density: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0) + " No Correlations")
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # clr = plt.contourf(xx, yy, snap_nc)
        # plt.title("Density: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0) + " Correlations")
        # plt.colorbar()
        # plt.show()

        # plt.figure()
        # for ii in range(len(snap_v)):
        #     plt.plot(x, snap_v[ii], label="v @ T = " + str(ii * dt * T / snaps))
        # plt.title("Velocity: Gamma = " + str(Gamma))
        # plt.legend()
        #
        # plt.figure()
        # for ii in range(len(snap_phi)):
        #     plt.plot(x, snap_phi[ii], label="phi @ T = " + str(ii * dt * T / snaps))
        # plt.title("Electrostatic Potential: Gamma = " + str(Gamma))
        # plt.legend()

        plt.show(block=True)
