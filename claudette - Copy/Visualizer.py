import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import matplotlib.cm as cm

class Visualizer:
    def __init__(self, grid_size, path, start_positions):
        self.grid_size = grid_size
        self.path = path
        self.start_positions = start_positions
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)
        self.start_button_ax = self.fig.add_axes([0.4, 0.05, 0.2, 0.075])
        self.start_button = Button(self.start_button_ax, "Start")
        self.start_button.on_clicked(self.animate_path)

        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, "", 
            ha="center", fontsize=12, fontweight="bold"
        )

        self.draw_grid()

    def draw_grid(self, highlight_initial=True):
        self.ax.clear()
        self.ax.set_xticks(np.arange(self.grid_size[1] + 1), minor=False)
        self.ax.set_yticks(np.arange(self.grid_size[0] + 1), minor=False)
        self.ax.grid(which="major", color="black", linestyle='-', linewidth=1)
        self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.grid_size[1])
        self.ax.set_ylim(0, self.grid_size[0])

        if highlight_initial:
            for (x, y) in self.start_positions:
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color='grey', label="Start"))

        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, 
            self.text_annotation.get_text(), ha="center", fontsize=12, fontweight="bold"
        )

        plt.draw()

    def animate_path(self, event):
        self.ax.clear()
        self.draw_grid(highlight_initial=True)

        if not self.path:
            self.text_annotation.set_text("No paths found")
            plt.draw()
            return

        self.text_annotation.set_text("Path found")
        plt.draw()

        num_steps = len(self.path)
        colormap = cm.get_cmap("tab20", num_steps)

        for i, step in enumerate(self.path):
            self.draw_grid(highlight_initial=False)

            self.text_annotation = self.ax.text(
                self.grid_size[1] / 2, self.grid_size[0] + 0.3, 
                "Path found", ha="center", fontsize=12, fontweight="bold"
            )

            color = colormap(i / num_steps)
            for (x, y) in step:
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color=color))

            plt.pause(0.4)

        plt.show()
