from AI_agent import AI_Agent
from Visualizer import Visualizer
import matplotlib.pyplot as plt

# Example usage
grid_size = (10, 10)
start_positions = [(5, 5), (5, 6)]
goal_positions = [(8, 7), (8, 8)]
agent = AI_Agent(grid_size, start_positions, goal_positions, topology="moore")

path = agent.search()
vis = Visualizer(grid_size, path)
vis.draw_grid()
plt.show()

if path:
    print("Path found! Visualizing...")
else:
    print("No path found.")