import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class Visualizer:
    def __init__(self, grid_size, path):
        self.grid_size = grid_size
        self.path = path
        self.fig, self.ax = plt.subplots(figsize=(6, 6))  # Set a square figure size
        self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)  # Adjust margins to center the grid
        self.start_button_ax = self.fig.add_axes([0.4, 0.05, 0.2, 0.075])  # Center the button
        self.start_button = Button(self.start_button_ax, "Start")
        self.start_button.on_clicked(self.animate_path)
        
        self.draw_grid()

    def draw_grid(self, draw_initial_state=True):
        self.ax.clear()
        self.ax.set_xticks(np.arange(self.grid_size[1] + 1), minor=False)
        self.ax.set_yticks(np.arange(self.grid_size[0] + 1), minor=False)
        self.ax.grid(which="major", color="black", linestyle='-', linewidth=1)
        self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        self.ax.set_aspect('equal')  # Ensure the grid cells are square
        self.ax.set_xlim(0, self.grid_size[1])  # Set x-axis limits to fit the grid
        self.ax.set_ylim(0, self.grid_size[0])  # Set y-axis limits to fit the grid
        
        if draw_initial_state and self.path:
            initial_step = self.path[0]
            for i, (x, y) in enumerate(initial_step):
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color='Grey'))
        plt.draw()

    def animate_path(self, event):
        if not self.path:
            print("No path to animate.")
            return
        
        colors = ['Grey']
        self.ax.clear()  # Clear the initial state
        self.draw_grid(draw_initial_state=False)  # Redraw the grid without the initial state
        
        for step in self.path:
            self.ax.clear()  # Clear the previous step
            self.draw_grid(draw_initial_state=False)  # Redraw the grid without the initial state
            for i, (x, y) in enumerate(step):
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color=colors[i % len(colors)]))
            plt.pause(0.4)
        plt.show()