import numpy as np

def test_roll():
    # creating a sample array with arange and reshape function
    array = np.arange(12).reshape(3, 4)
    print("Original array : \n", array[:])

    # Rolling array; Shifting one place
    print("\nRolling with 1 shift : \n", np.roll(array[:], 1, axis =1))

    q = array
    q0 = np.copy(q)
    dt = 1e-3
    dx = 10/100
    dF = 0
    q[:,  1:-2] = q0[:, 1:-2] - dt / dx * dF
    q[:, 0] = q0[:, 0]
    q[:, -1] = q0[:, -1]

    print(q)
    print(q0)