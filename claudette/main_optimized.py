from AI_agent_Optimized import AI_Agent_Optimized
from Visualizer import Visualizer
import matplotlib.pyplot as plt
import time

def run_optimized_search(grid_size, start_positions, goal_positions, topology="moore", time_limit=30):
    print(f"Initializing optimized AI Agent...")
    print(f"Grid size: {grid_size}")
    print(f"Start positions: {start_positions}")
    print(f"Goal positions: {goal_positions}")
    print(f"Topology: {topology}")
    print(f"Time limit: {time_limit} seconds")
    
    start_time = time.time()
    
    # Initialize the optimized agent
    agent = AI_Agent_Optimized(grid_size, start_positions, goal_positions, topology)
    
    # Print some information about assignments
    print("\nElement assignments (start -> goal):")
    for start, goal in agent.assignments.items():
        print(f"{start} -> {goal}")
    
    # Run the search
    print("\nSearching for optimal path...")
    path = agent.search(time_limit)
    
    end_time = time.time()
    search_time = end_time - start_time
    
    # Report results
    if path:
        print(f"Path found with {len(path)-1} moves in {search_time:.2f} seconds")
        print(f"Starting visualization...")
        
        # Visualize the path
        vis = Visualizer(grid_size, path, start_positions)
        vis.draw_grid()
        plt.show()
        
        return path
    else:
        print(f"No path found after {search_time:.2f} seconds")
        return None

# Example usage
if __name__ == "__main__":
    grid_size = (10, 10)
    
    # Example: Moving a 3x2 block pattern
    start_positions = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2), (0,3), (1,3), (0,4), (1,4), (0,5), (1,5), (0,6), (1,6), (0,7), (1,7), (0,8), (1,8), (0,9), (1,9)]
    goal_positions =  [(8,4), (8,5), (7,3), (7,4), (7,5), (7,6), (6,2), (6,3), (6,6), (6,7), (5,2), (5,3), (5,6), (5,7), (4,3), (4,4), (4,5), (4,6), (3,4), (3,5)]
    # Run the search with a 30-second time limit
    run_optimized_search(grid_size, start_positions, goal_positions, "moore", 30)
