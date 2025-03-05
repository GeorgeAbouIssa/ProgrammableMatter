from AI_agent import AI_Agent
from Visualizer import Visualizer
import matplotlib.pyplot as plt

# Example usage
grid_size = (10, 10)
start_positions = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]
goal_positions = [(0,9),(1,9),(2,9),(3,9),(4,9),(5,9)]
agent = AI_Agent(grid_size, start_positions, goal_positions, topology="moore")
plt.show()
path = agent.search()
vis = Visualizer(grid_size, path, start_positions)
vis.draw_grid()
plt.show()

if path:
    print("Path found! Visualizing...")
else:
    print("No path found.")