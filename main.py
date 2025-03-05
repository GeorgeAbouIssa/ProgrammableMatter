from AI_agent import AI_Agent
from Visualizer import Visualizer
import matplotlib.pyplot as plt

# Example usage
grid_size = (10, 10)
start_positions = [(0,0),(0,1)]
goal_positions = [(0,9),(1,9)]
agent = AI_Agent(grid_size, start_positions, goal_positions, topology="moore")

path = agent.search()
vis = Visualizer(grid_size, path, start_positions)
vis.draw_grid()
plt.show()

if path:
    print("Path found! Visualizing...")
else:
    print("No path found.")