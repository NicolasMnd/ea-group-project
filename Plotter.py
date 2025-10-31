import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Qt5Agg")

class LivePlotter:
    def __init__(self):
        plt.ion()  # interactive mode ON
        self.x_vals = []
        self.y1_vals = []
        self.y2_vals = []

        self.fig, self.ax = plt.subplots()
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, val1, val2):
        self.x_vals.append(len(self.x_vals))
        self.y1_vals.append(val1)
        self.y2_vals.append(val2)

        self.ax.clear()
        self.ax.plot(self.x_vals, self.y1_vals, label="Value 1")
        self.ax.plot(self.x_vals, self.y2_vals, label="Value 2")

        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Values")
        self.ax.legend()
        self.ax.set_title("Live Updating Plot")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #plt.pause(0.01)

