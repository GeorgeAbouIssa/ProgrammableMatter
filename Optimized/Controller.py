import sys
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Button, RadioButtons, TextBox
from ConnectedMatterAgent import ConnectedMatterAgent
from Visualizer import Visualizer

class SearchController:
    def __init__(self, grid_size, formations, topology="moore", time_limit=1000):
        self.grid_size = grid_size
        self.formations = formations  # Dictionary of shape names and their goal positions
        self.start_positions = formations["start"]
        self.current_shape = "Ring"  # Default shape
        self.goal_positions = formations["Ring"]
        self.topology = topology
        self.time_limit = time_limit
        self.search_completed = False

        # Enable interactive mode so the grid appears first
        plt.ion()  

        # Create visualizer and show the grid first
        self.vis = Visualizer(grid_size, [], self.start_positions)
        self.vis.draw_grid()
        plt.show(block=False)  # Show grid before printing anything

        # Add radio buttons for shape selection instead of dropdown
        self.radio_ax = self.vis.fig.add_axes([0.05, 0.05, 0.25, 0.15])
        self.radio = RadioButtons(
            self.radio_ax, 
            labels=list(formations.keys())[1:],  # Skip "start" key
            active=0  # Default to first option (ring)
        )
        self.radio.on_clicked(self.on_shape_selected)

        # Attach the window close event to stop execution
        self.vis.fig.canvas.mpl_connect('close_event', self.on_close)
        
        self.selection_mode = False  # False = normal mode, True = goal selection mode
        self.custom_goal = []  # Store the custom goal positions
        self.selection_active = False  # New flag to track if selection is currently active
        
        # Add a button to toggle selection mode
        self.select_button_ax = self.vis.fig.add_axes([0.4, 0.05, 0.2, 0.075])
        self.select_button = Button(self.select_button_ax, "Select Goal")
        self.select_button.on_clicked(self.toggle_selection_mode)
        
        # Add label for grid size
        label_ax = self.vis.fig.add_axes([0.82, 0.75, 0.15, 0.05])
        label_ax.text(0.5, 0.5, 'Grid Size', ha='center', va='center')
        label_ax.axis('off')
        
        # Add text input for grid size (centered, narrower)
        self.grid_text_ax = self.vis.fig.add_axes([0.85, 0.7, 0.09, 0.05])  # Narrower width (0.09)
        self.grid_text_box = TextBox(
            self.grid_text_ax, 
            '',  # Remove label since we added it separately above
            initial=f"{grid_size[0]}",
            textalignment='center'  # Center the text in the box
        )
        
        # Add button to apply grid size (centered under text box)
        self.grid_button_ax = self.vis.fig.add_axes([0.845, 0.63, 0.1, 0.05])
        self.grid_button = Button(self.grid_button_ax, "Apply")
        self.grid_button.on_clicked(self.change_grid_size)

        self.obstacles = set()  # Store obstacle positions
        self.obstacle_mode = False  # Track if obstacle placement mode is active
        
        # Add obstacle controls (left side)
        self.obstacle_button_ax = self.vis.fig.add_axes([0.05, 0.25, 0.15, 0.075])
        self.obstacle_button = Button(self.obstacle_button_ax, "Place Obstacles")
        self.obstacle_button.on_clicked(self.toggle_obstacle_mode)
        
        self.reset_obstacles_ax = self.vis.fig.add_axes([0.05, 0.35, 0.15, 0.075])
        self.reset_obstacles_button = Button(self.reset_obstacles_ax, "Clear Obstacles")
        self.reset_obstacles_button.on_clicked(self.reset_obstacles)

        # Print initialization info
        print(f"Initializing Connected Programmable Matter Agent...")
        print(f"Grid size: {grid_size}")
        print(f"Start positions: {self.start_positions}")
        print(f"Current shape: {self.current_shape}")
        print(f"Topology: {topology}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Constraint: All elements must remain connected during movement")

        # Initialize the agent
        self.agent = ConnectedMatterAgent(grid_size, self.start_positions, self.goal_positions, topology)
        
        # Set up button to start search
        self.vis.button.on_clicked(self.handle_button)
        
        self.dragging = False
        self.drag_start = None
        self.drag_offset = None
        
        # For boundary checking
        self.shape_min_x = 0
        self.shape_max_x = 0
        self.shape_min_y = 0
        self.shape_max_y = 0
        
        # Add mouse events for dragging and selection
        self.vis.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.vis.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.vis.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def toggle_selection_mode(self, event):
        """Toggle between normal mode and goal selection mode"""
        if self.dragging:  # Don't allow mode switch while dragging
            return
            
        self.selection_mode = not self.selection_mode
        self.selection_active = self.selection_mode
        
        if self.selection_mode:
            # Entering selection mode
            self.custom_goal = []  # Clear any previous selection
            self.select_button.label.set_text("Confirm Goal")
            self.vis.update_text("Click on grid cells to define goal state", color="blue")
            
            # Clear any highlighted goal shape
            self.vis.draw_grid(highlight_goal=False)
        else:
            # Exiting selection mode
            if len(self.custom_goal) == len(self.start_positions):
                self.goal_positions = self.custom_goal.copy()  # Make a copy of the custom goal
                self.agent = ConnectedMatterAgent(self.grid_size, self.start_positions, self.goal_positions, self.topology)
                self.select_button.label.set_text("Select Goal")
                
                # Update visualization with the new goal shape
                self.vis.draw_grid()
                self.vis.highlight_goal_shape(self.goal_positions)
                self.vis.update_text(f"Custom goal set with {len(self.goal_positions)} blocks", color="green")
                
                # Reset search state
                self.search_completed = False
                self.vis.animation_started = False
                self.vis.animation_done = False
                self.vis.current_step = 0
                self.vis.path = None
            else:
                self.selection_mode = True  # Stay in selection mode if invalid
                self.select_button.label.set_text("Select Goal")
                self.vis.update_text(f"Invalid goal: Need exactly {len(self.start_positions)} blocks", color="red")

    def on_grid_click(self, event, x, y):
        """Handle grid cell clicks for goal selection"""
        if not self.selection_mode or not self.selection_active:
            return
        
        pos = (x, y)
        
        # Check if position is valid (within bounds and not an obstacle)
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            # Don't allow selection of obstacle positions
            if pos in self.obstacles:
                self.vis.update_text("Cannot place goal on obstacles", color="red")
                return
                
            if pos in self.custom_goal:
                # Remove this position
                self.custom_goal.remove(pos)
            else:
                # Add this position if we haven't reached the limit
                if len(self.custom_goal) < len(self.start_positions):
                    self.custom_goal.append(pos)
                else:
                    # Replace the first position
                    self.custom_goal.pop(0)
                    self.custom_goal.append(pos)
            
            # Redraw the grid with current selection
            self.vis.draw_grid(highlight_goal=False)
            for cell_pos in self.custom_goal:
                self.highlight_cell(cell_pos, color='green')
            
            # Update the counter
            self.vis.update_text(f"Selected {len(self.custom_goal)}/{len(self.start_positions)} blocks", color="blue")
    
    def highlight_cell(self, pos, color='green'):
        """Highlight a grid cell with the specified color"""
        x, y = pos
        self.vis.ax.add_patch(plt.Rectangle((y, x), 1, 1, color=color, alpha=0.7))
        plt.draw()    

    def on_close(self, event):
        """Stops execution when the Matplotlib window is closed."""
        print("\nWindow closed. Exiting program.")
        sys.exit()  # Forcefully stop the script
        
    def on_shape_selected(self, label):
        """Handle shape selection from radio buttons"""
        # Cancel selection mode if active when changing shapes
        if self.selection_mode:
            self.selection_mode = False
            self.selection_active = False
            self.select_button.label.set_text("Select Goal")
        
        self.current_shape = label
        self.goal_positions = self.formations[self.current_shape]
        
        # Update agent with new goal positions and existing obstacles
        self.agent = ConnectedMatterAgent(self.grid_size, self.start_positions, self.goal_positions, self.topology)
        self.agent.set_obstacles(self.obstacles)  # Maintain obstacles
        
        print(f"Selected shape: {self.current_shape}")
        print(f"Goal positions: {self.goal_positions}")
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None
        
        # Update visualization with highlighted goal shape and obstacles
        self.vis.draw_grid()
        self.vis.highlight_goal_shape(self.goal_positions)
        self.vis.button.label.set_text("Search")
        self.vis.update_text(f"Selected {self.current_shape} shape", color="blue")

    def handle_button(self, event):
        """Handle button clicks based on current state"""
        if not self.search_completed:
            self.run_search(event)
        else:
            self.vis.handle_button_click(event)

    def run_search(self, event):
        """Runs the search when the Search button is clicked."""
        # Clear goal shape highlight but maintain obstacles
        self.vis.draw_grid()
        if self.obstacles:  # Redraw obstacles
            self.vis.draw_obstacles(self.obstacles)
        self.vis.update_text("Searching for a path...", color="red")
        plt.pause(1)  # Force update to show "Searching..." before search starts
        
        print("\nSearching for optimal path with connectivity constraint...")
        start_time = time.time()
        path = self.agent.search(self.time_limit)
        search_time = time.time() - start_time
        
        self.search_completed = True

        if path:
            print(f"Path found with {len(path)-1} moves in {search_time:.2f} seconds")
            print(f"Ready for visualization...")
            self.vis.path = path  # Update path in visualizer
            self.vis.button.label.set_text("Start")  # Set button text to Start
            self.vis.update_text(f"Path found ({len(path)-1} moves)", color="green")
            plt.draw()
        else:
            print(f"No path found after {search_time:.2f} seconds")
            self.vis.button.label.set_text("Search")  # Reset button text
            self.vis.update_text("No paths found", color="red")
            plt.draw()

    def on_text_submit(self, text):
        """Handle grid size text submission"""
        self.change_grid_size(None)  # Call change_grid_size when Enter is pressed
            
    def change_grid_size(self, event):
        """Handle grid size change"""
        try:
            n = int(self.grid_text_box.text)
            if n < 10:
                self.vis.update_text("Grid size must be at least 10", color="red")
                return
            if n > 200:
                self.vis.update_text("Grid size cannot exceed 200", color="red")
                return

            # Update grid size
            self.grid_size = (n, n)
            
            # Update text box to show clean number
            self.grid_text_box.set_val(str(n))
            
            # Reinitialize agent with new grid size
            self.agent = ConnectedMatterAgent(self.grid_size, self.start_positions, self.goal_positions, self.topology)
            
            # Update visualization
            self.vis.grid_size = self.grid_size
            self.vis.draw_grid()
            if self.obstacles:  # Maintain obstacles when changing grid size
                self.vis.draw_obstacles(self.obstacles)
            self.vis.update_text(f"Grid size updated to {n}x{n}", color="green")
            
            # Reset search state
            self.search_completed = False
            self.vis.animation_started = False
            self.vis.animation_done = False
            self.vis.current_step = 0
            self.vis.path = None
            
        except ValueError:
            self.vis.update_text("Invalid grid size. Enter a number between 10-200", color="red")

    def toggle_obstacle_mode(self, event):
        """Toggle obstacle placement mode"""
        if self.selection_mode or self.dragging:  # Don't allow while in other modes
            return
            
        self.obstacle_mode = not self.obstacle_mode
        if self.obstacle_mode:
            self.obstacle_button.label.set_text("Confirm Obstacles")
            self.vis.update_text("Click to place/remove obstacles", color="orange")
        else:
            self.obstacle_button.label.set_text("Place Obstacles")
            self.vis.update_text("Obstacles confirmed", color="green")
            # Update both agent and visualizer with obstacles
            self.agent.set_obstacles(self.obstacles)
            self.vis.set_obstacles(self.obstacles)  # New method to update visualizer

    def reset_obstacles(self, event):
        """Clear all obstacles"""
        self.obstacles.clear()
        self.obstacle_mode = False
        self.obstacle_button.label.set_text("Place Obstacles")
        self.agent.set_obstacles(self.obstacles)
        self.vis.set_obstacles(self.obstacles)  # Update visualizer
        self.vis.draw_grid()
        self.vis.update_text("Obstacles cleared", color="blue")

    def on_mouse_press(self, event):
        """Handle mouse press events for dragging, selection, and obstacles"""
        if event.inaxes != self.vis.ax:
            return
            
        # Convert click coordinates to grid cell
        x, y = int(event.ydata), int(event.xdata)
        
        # Handle obstacle placement mode
        if self.obstacle_mode:
            pos = (x, y)
            # Don't allow obstacles on start positions or goal positions
            if pos in self.start_positions or pos in self.goal_positions:
                return
            
            if pos in self.obstacles:
                self.obstacles.remove(pos)
            else:
                self.obstacles.add(pos)
            self.vis.draw_grid()
            self.vis.highlight_goal_shape(self.goal_positions)
            self.vis.draw_obstacles(self.obstacles)
            return
            
        # Handle selection mode separately from dragging
        if self.selection_mode and self.selection_active:
            self.on_grid_click(event, x, y)
            return
        
        # Only allow dragging when not in selection mode
        click_pos = (x, y)
        
        # Check if click is within the goal shape
        if any(abs(gx - x) < 1 and abs(gy - y) < 1 for gx, gy in self.goal_positions):
            self.dragging = True
            self.drag_start = click_pos
            self.drag_offset = []
            
            # Calculate offsets for all points relative to click position
            for gx, gy in self.goal_positions:
                self.drag_offset.append((gx - x, gy - y))
            
            # Calculate shape boundaries for edge checking
            x_coords = [pos[0] for pos in self.goal_positions]
            y_coords = [pos[1] for pos in self.goal_positions]
            self.shape_min_x = min(x_coords)
            self.shape_max_x = max(x_coords)
            self.shape_min_y = min(y_coords)
            self.shape_max_y = max(y_coords)
            
            # Calculate shape dimensions
            shape_width = self.shape_max_x - self.shape_min_x
            shape_height = self.shape_max_y - self.shape_min_y
            
            # Calculate distance from click to shape bounds
            self.bound_left = x - self.shape_min_x
            self.bound_right = self.shape_max_x - x
            self.bound_top = y - self.shape_min_y
            self.bound_bottom = self.shape_max_y - y
    
    def on_mouse_release(self, event):
        """Handle mouse release events for dragging"""
        if self.dragging and not self.selection_mode:
            self.dragging = False
            if event.inaxes == self.vis.ax:
                # Snap to grid
                x, y = int(event.ydata), int(event.xdata)
                # Apply boundary constraints
                x, y = self.constrain_to_boundaries(x, y)
                
                # Calculate new positions
                new_positions = []
                for offset_x, offset_y in self.drag_offset:
                    new_x = x + offset_x
                    new_y = y + offset_y
                    if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                        new_positions.append((new_x, new_y))
                
                # Check for obstacle collisions before finalizing
                if len(new_positions) == len(self.goal_positions) and not self.check_goal_obstacle_collision(new_positions):
                    self.goal_positions = new_positions
                    self.agent = ConnectedMatterAgent(self.grid_size, self.start_positions, 
                                                    self.goal_positions, self.topology)
                    self.agent.set_obstacles(self.obstacles)  # Update agent with obstacles
                    self.search_completed = False
                    self.vis.draw_grid()
                    self.vis.highlight_goal_shape(self.goal_positions)
                    self.vis.draw_obstacles(self.obstacles)  # Redraw obstacles
                    self.vis.update_text("Goal shape moved", color="blue")
                else:
                    # If invalid position, revert to original position
                    self.vis.draw_grid()
                    self.vis.highlight_goal_shape(self.goal_positions)
                    self.vis.draw_obstacles(self.obstacles)
                    self.vis.update_text("Invalid position - overlapping with obstacles", color="red")

    def constrain_to_boundaries(self, x, y):
        """Constrain the drag point to keep the shape within grid boundaries"""
        # Constrain x-coordinate
        min_x = self.bound_left  # Minimum allowed x (to keep left edge in bounds)
        max_x = self.grid_size[0] - 1 - self.bound_right  # Maximum allowed x (to keep right edge in bounds)
        x = max(min_x, min(x, max_x))
        
        # Constrain y-coordinate
        min_y = self.bound_top  # Minimum allowed y (to keep top edge in bounds)
        max_y = self.grid_size[1] - 1 - self.bound_bottom  # Maximum allowed y (to keep bottom edge in bounds)
        y = max(min_y, min(y, max_y))
        
        return x, y
    
    def on_mouse_move(self, event):
        """Handle mouse movement events for dragging"""
        if self.dragging and not self.selection_mode and event.inaxes == self.vis.ax:
            # Get cursor position with boundary constraints
            x, y = event.ydata, event.xdata
            x, y = self.constrain_to_boundaries(x, y)
            
            # Calculate new positions
            temp_positions = []
            for offset_x, offset_y in self.drag_offset:
                new_x = x + offset_x
                new_y = y + offset_y
                temp_positions.append((new_x, new_y))
            
            # Check for obstacle collisions
            if self.check_goal_obstacle_collision(temp_positions):
                return  # Skip updating if there's a collision
            
            # Clear and redraw the grid with obstacles
            self.vis.draw_grid()
            if self.obstacles:
                self.vis.draw_obstacles(self.obstacles)
            
            # Highlight the shape at its temporary position
            self.vis.highlight_goal_shape(temp_positions)

    def check_goal_obstacle_collision(self, positions):
        """Check if any goal position overlaps with obstacles"""
        return any(pos in self.obstacles for pos in positions)

# Example usage
if __name__ == "__main__":
    grid_size = (10, 10)
    
    # Dictionary of formations
    formations = {
        "start": [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4),
                  (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7), (0, 8), (1, 8), (0, 9), (1, 9)],
        
        "Ring": [(7, 4), (7, 5), (6, 3), (6, 4), (6, 5), (6, 6), (5, 2), (5, 3), (5, 6), (5, 7),
                 (4, 2), (4, 3), (4, 6), (4, 7), (3, 3), (3, 4), (3, 5), (3, 6), (2, 4), (2, 5)],
        
        "Rectangle": [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
                      (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)],
        
        "Triangle": [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 2), (3, 3),
                     (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (5, 4), (5, 5)]
    }
    
    controller = SearchController(grid_size, formations, "moore", 1000)

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep window open until manually closed