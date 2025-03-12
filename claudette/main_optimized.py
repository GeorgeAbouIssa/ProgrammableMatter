import sys
import matplotlib.pyplot as plt
import time
from AI_agent_Optimized import AI_Agent_Optimized
from Visualizer import Visualizer

class SearchController:
    def __init__(self, grid_size, start_positions, goal_positions, topology="moore", time_limit=30):
        self.grid_size = grid_size
        self.start_positions = start_positions
        self.goal_positions = goal_positions
        self.topology = topology
        self.time_limit = time_limit

        # Enable interactive mode so the grid appears first
        plt.ion()  

        # Create visualizer and show the grid first
        self.vis = Visualizer(grid_size, [], start_positions)
        self.vis.draw_grid()
        plt.show(block=False)  # Show grid before printing anything

        # Attach the window close event to stop execution
        self.vis.fig.canvas.mpl_connect('close_event', self.on_close)

        # Now print everything after the grid appears
        print(f"Initializing optimized AI Agent...")
        print(f"Grid size: {grid_size}")
        print(f"Start positions: {start_positions}")
        print(f"Goal positions: {goal_positions}")
        print(f"Topology: {topology}")
        print(f"Time limit: {time_limit} seconds")

        # Initialize the agent
        self.agent = AI_Agent_Optimized(grid_size, start_positions, goal_positions, topology)
        self.vis.button.on_clicked(self.run_search)
        
  # Attach search function to button

        # Print initial assignments
        print("\nElement assignments (start -> goal):")
        for start, goal in self.agent.assignments.items():
            print(f"{start} -> {goal}")

    def on_close(self, event):
        """Stops execution when the Matplotlib window is closed."""
        print("\nWindow closed. Exiting program.")
        sys.exit()  # Forcefully stop the script

    def run_search(self, event):
        """Runs the search when the Start button is clicked, showing 'Loading...' while searching."""
        self.vis.update_text("Searching for a path...", color = "red")  # Update UI text
        plt.pause(0.1)  # Force update to show "Loading..." before search starts

        print("\nSearching for optimal path...")
        start_time = time.time()
        path = self.agent.search(self.time_limit)
        search_time = time.time() - start_time

        if path:
            print(f"Path found with {len(path)-1} moves in {search_time:.2f} seconds")
            print(f"Starting visualization...")
            self.vis.path = path  # Update path in visualizer
            self.vis.update_text("Path found", color = "black")  # Update UI text
            self.vis.animate_path(event)  # Start animation
        else:
            print(f"No path found after {search_time:.2f} seconds")
            self.vis.update_text("No path found", color = "black")  # Update UI text
            plt.draw()

# Example usage
if __name__ == "__main__":
    grid_size = (10, 10)
    start_positions = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4),
                       (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7), (0, 8), (1, 8), (0, 9), (1, 9)]
    goal_positions = [(8, 4), (8, 5), (7, 3), (7, 4), (7, 5), (7, 6), (6, 2), (6, 3), (6, 6), (6, 7),
                      (5, 2), (5, 3), (5, 6), (5, 7), (4, 3), (4, 4), (4, 5), (4, 6), (3, 4), (3, 5)]

    controller = SearchController(grid_size, start_positions, goal_positions, "moore", 30)

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep window open until manually closed
