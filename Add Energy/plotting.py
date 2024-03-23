import matplotlib.pyplot as plt
import numpy as np


class ArrayVisualizer:
    def __init__(self, arrays, type):
        self.arrays = arrays
        self.axes = []

        if type == "conserved":
             = self.arrays.shape()
        else:



    def color_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, array in enumerate(self.arrays):
            plt.imshow(array, cmap='viridis')
            plt.colorbar()
            plt.show()


    def line_plot(self):
        num_rows, num_cols = self.array.shape

        # Plotting each row as a separate line
        for i in range(num_rows):
            plt.plot(self.array[i, :], label=f"Row {i + 1}")

        plt.legend()
        plt.xlabel('Column')
        plt.ylabel('Value')
        plt.title('Line Plots')
        plt.show()


    def generate_line_plot(self, array):
        for i, array in enumerate(self.arrays):
            plt.plot(array, cmap='viridis')
            plt.show()


    def generate_color_plot(self):
        for i, array in enumerate(self.arrays):
            self.color_plot(array)


# Example usage
arrays = [
    np.random.rand(100, 100),
    np.random.rand(100, 100),
    np.random.rand(100, 100)
]

visualizer = ArrayVisualizer(arrays)
visualizer.color_plots(arrays)