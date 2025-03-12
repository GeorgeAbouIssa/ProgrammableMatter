import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class Visualizer:
    def __init__(self, grid_size, path, start_positions):
        self.grid_size = grid_size
        self.path = path  # Ensure path is passed correctly
        self.start_positions = start_positions
        self.paused = False  # Track pause state
        self.animation_done = False  # Track if animation is complete
        self.current_step = 0  # Keep track of the animation step
        self.animation_started = False  # Track if animation has started

        self.fig, self.ax = plt.subplots(figsize=(6, 6))  # Set a square figure size
        self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)  # Adjust margins to center the grid

        # Create button
        self.button_ax = self.fig.add_axes([0.4, 0.05, 0.2, 0.075])  # Center the button
        self.button = Button(self.button_ax, "Start")
        self.button.on_clicked(self.handle_button_click)

        # Initialize text annotation for status messages
        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, "", 
            ha="center", fontsize=12, fontweight="bold"
        )

        self.draw_grid()  # Ensure the grid is drawn initially

    def draw_grid(self, highlight_initial=True):
        """Draws the grid and optionally highlights the initial position."""
        self.ax.clear()
        self.ax.set_xticks(np.arange(self.grid_size[1] + 1), minor=False)
        self.ax.set_yticks(np.arange(self.grid_size[0] + 1), minor=False)
        self.ax.grid(which="major", color="black", linestyle='-', linewidth=1)
        self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        self.ax.set_aspect('equal')  # Ensure the grid cells are square
        self.ax.set_xlim(0, self.grid_size[1])  # Set x-axis limits to fit the grid
        self.ax.set_ylim(0, self.grid_size[0])  # Set y-axis limits to fit the grid

        if highlight_initial:
            for (x, y) in self.start_positions:
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color='Grey', label="Start"))  

        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, 
            self.text_annotation.get_text(), ha="center", fontsize=12, fontweight="bold"
        )

        plt.draw()
        
    def update_text(self, message, color="black"):
        """Updates the status message above the grid dynamically with color."""
        self.text_annotation.set_text(message)
        self.text_annotation.set_color(color)  # Set text color
        plt.draw()  # Force update

    def handle_button_click(self, event):
        """Handles the button click event for start, pause, resume, and restart."""
        if not self.path:
            print("No path found. Make sure path is set before starting animation.")
            return

        if self.animation_done:
            # Restart the animation
            self.animation_done = False
            self.animation_started = False
            self.current_step = 0  # Reset animation step
            self.button.label.set_text("Pause")
            self.animate_path()
        elif not self.animation_started:
            # Start animation for the first time
            self.animation_started = True
            self.button.label.set_text("Pause")
            self.animate_path()
        else:
            # Toggle pause state and resume immediately
            self.paused = not self.paused
            self.button.label.set_text("Resume" if self.paused else "Pause")
            if not self.paused:
                self.animate_path()  # Resume immediately

    def animate_path(self):
        """Animates the path step by step, ensuring the initial state remains visible."""
        if not self.path:
            self.update_text("No paths found", color="red")
            return

        self.update_text("Path found", color="green")

        colors = ['grey']

        # Continue from the current step instead of restarting
        while self.current_step < len(self.path):
            if self.paused:
                return  # Stop animation if paused

            step = self.path[self.current_step]  # Get current step

            self.draw_grid(highlight_initial=False)  # Redraw grid but don't clear initial step

            for (x, y) in step:
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color=colors[0]))
            
            plt.pause(0.1)  # Slow down animation for visibility
            self.current_step += 1  # Move to the next step

        # Mark animation as complete and update button
        self.animation_done = True
        self.button.label.set_text("Restart")

        plt.draw()
