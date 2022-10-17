import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


class DMDTest:
    def __init__(self, data, r, dt):  # data = 2D array (row = vec{x}, column = t)
        self.n, self.m = np.shape(data)
        self.data1 = np.copy(data)
        self.data2 = np.roll(data, -1, axis = 1)
        self.r = r
        self.dt = dt

        self.Phi = np.array((r, r))
        # self.lmbda = np.array(())
        # self.omega = np.array(())

    def DMD(self):
        # Calculate SVD
        U, S, VH = la.svd(self.data1)
        V = np.transpose(VH)

        Ur = U[:, 0:self.r]
        Sr = np.diag(S)[0:self.r, 0:self.r]
        Vr = V[:, 0:self.r]

        # Reduced Rank Matrix
        Atilde = np.transpose(Ur) @ self.data2 @ Vr @ la.inv(Sr)

        # Eigenvalues, Eigenvectors of Atilde
        W, D = la.eig(Atilde)

        # Expansion with DMD Modes
        self.Phi = self.data2 @ Vr @ la.inv(Sr) @ W
        lmbda = np.diag(D)
        omega = np.log(lmbda) / self.dt
        b = self.Phi @ self.data1

        # print(W)

        return b