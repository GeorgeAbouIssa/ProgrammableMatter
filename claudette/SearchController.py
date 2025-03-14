import sys
import matplotlib.pyplot as plt
import time
from ConnectedMatterAgent import ConnectedMatterAgent
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

        # Print initialization info
        print(f"Initializing Connected Programmable Matter Agent...")
        print(f"Grid size: {grid_size}")
        print(f"Start positions: {start_positions}")
        print(f"Goal positions: {goal_positions}")
        print(f"Topology: {topology}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Constraint: All elements must remain connected during movement")

        # Initialize the agent
        self.agent = ConnectedMatterAgent(grid_size, start_positions, goal_positions, topology)
        
        # Set up button to start search
        self.vis.button.on_clicked(self.run_search)

    def on_close(self, event):
        """Stops execution when the Matplotlib window is closed."""
        print("\nWindow closed. Exiting program.")
        sys.exit()  # Forcefully stop the script

    def run_search(self, event):
        """Runs the search when the Start button is clicked."""
        self.vis.update_text("Searching for a path...", color="red")
        plt.pause(1)  # Force update to show "Searching..." before search starts
        

        print("\nSearching for optimal path with connectivity constraint...")
        start_time = time.time()
        path = self.agent.search(self.time_limit)
        search_time = time.time() - start_time

        if path:
            print(f"Path found with {len(path)-1} moves in {search_time:.2f} seconds")
            print(f"Starting visualization...")
            self.vis.path = path  # Update path in visualizer
            self.vis.button.label.set_text("Start")  # Reset button text
            self.vis.update_text(f"Path found ({len(path)-1} moves)", color="green")
            plt.draw()

        else:
            print(f"No path found after {search_time:.2f} seconds")
            self.vis.update_text("No path found", color="red")
            plt.draw()   

# Example usage
if __name__ == "__main__":
    grid_size = (10, 10)
    
    # Example for a square shape formation (2x2 block -> target square)
    '''
    start_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    goal_positions = [(5, 5), (5, 6), (6, 5), (6, 6)]
    '''
    # For a larger shape, uncomment and use this instead
    
    start_positions = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4),
                       (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7), (0, 8), (1, 8), (0, 9), (1, 9)]
    # Ring shape formation
    goal_positions = [(7, 4), (7, 5), (6, 3), (6, 4), (6, 5), (6, 6), (5, 2), (5, 3), (5, 6), (5, 7),
                      (4, 2), (4, 3), (4, 6), (4, 7), (3, 3), (3, 4), (3, 5), (3, 6), (2, 4), (2, 5)]
    
    # Rectangle shape formation
    # goal_positions =  [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    #                    (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)]
    
    # Triangle shape formation
    # goal_positions = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 2), (3, 3),
    #                   (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (5, 4), (5, 5)]   
    
    controller = SearchController(grid_size, start_positions, goal_positions, "moore", 30)

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep window open until manually closed
