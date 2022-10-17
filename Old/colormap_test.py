import matplotlib.pyplot as plt
import numpy as np

def colormap_trial():
    Nx = 51 # 499
    Ny = 6
    Rx = 1
    Ry = 1
    x = np.linspace(-Rx, Rx, Nx)
    y = np.linspace(0, Ry, Ny)
    X = x
    for elmn in x:
        X = np.vstack((X,x))
    Y = y
    for elmn in y:
        Y = np.vstack((Y,y))
    Y = Y.transpose()
    print(X)
    print(Y)
    xx,yy = np.meshgrid(X, Y, sparse=False, indexing='xy')
    z = (np.sin(xx) + np.sin(yy)) / (1 + xx ** 2 + yy ** 2)
    h = plt.contourf(xx, yy, z)
    plt.plot(xx, yy, color='k', linestyle='none')
    plt.colorbar()
    plt.show()