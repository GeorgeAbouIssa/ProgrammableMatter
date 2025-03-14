import heapq
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import copy

class ConnectedMatterAgent:
    def __init__(self, grid_size, start_positions, goal_positions, topology="moore"):
        self.grid_size = grid_size
        self.start_positions = list(start_positions)
        self.goal_positions = list(goal_positions)
        self.topology = topology
        
        # Set moves based on topology
        if self.topology == "moore":
            self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # Von Neumann
            self.directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            
        # Initialize the start and goal states
        self.start_state = frozenset((x, y) for x, y in start_positions)
        self.goal_state = frozenset((x, y) for x, y in goal_positions)
        
        # Calculate the centroid of the goal positions for block movement phase
        self.goal_centroid = self.calculate_centroid(self.goal_positions)
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # For optimizing the search
        self.articulation_points_cache = {}
        
    def calculate_centroid(self, positions):
        """Calculate the centroid (average position) of a set of positions"""
        if not positions:
            return (0, 0)
        x_sum = sum(pos[0] for pos in positions)
        y_sum = sum(pos[1] for pos in positions)
        return (x_sum / len(positions), y_sum / len(positions))
    
    def is_connected(self, positions):
        """Check if all positions are connected using BFS"""
        if not positions:
            return True
            
        # Convert to set for O(1) lookup
        positions_set = set(positions)
        
        # Start BFS from first position
        start = next(iter(positions_set))
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            # Check all adjacent positions
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in positions_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # All positions should be visited if connected
        return len(visited) == len(positions_set)
    
    def get_articulation_points(self, state_set):
        """
        Find articulation points (critical points that if removed would disconnect the structure)
        Uses a modified DFS algorithm
        """
        state_hash = hash(frozenset(state_set))
        if state_hash in self.articulation_points_cache:
            return self.articulation_points_cache[state_hash]
            
        if len(state_set) <= 2:  # All points are critical in structures of size 1-2
            self.articulation_points_cache[state_hash] = set(state_set)
            return set(state_set)
            
        articulation_points = set()
        visited = set()
        discovery = {}
        low = {}
        parent = {}
        time = [0]  # Using list to allow modification inside nested function
        
        def dfs(u, time):
            children = 0
            visited.add(u)
            discovery[u] = low[u] = time[0]
            time[0] += 1
            
            # Visit all neighbors
            for dx, dy in self.directions:
                v = (u[0] + dx, u[1] + dy)
                if v in state_set:
                    if v not in visited:
                        children += 1
                        parent[v] = u
                        dfs(v, time)
                        
                        # Check if subtree rooted with v has a connection to ancestors of u
                        low[u] = min(low[u], low[v])
                        
                        # u is an articulation point if:
                        # 1) u is root and has two or more children
                        # 2) u is not root and low value of one of its children >= discovery value of u
                        if parent.get(u) is None and children > 1:
                            articulation_points.add(u)
                        if parent.get(u) is not None and low[v] >= discovery[u]:
                            articulation_points.add(u)
                            
                    elif v != parent.get(u):  # Update low value of u for parent function calls
                        low[u] = min(low[u], discovery[v])
        
        # Call DFS for all vertices
        for point in state_set:
            if point not in visited:
                dfs(point, time)
                
        self.articulation_points_cache[state_hash] = articulation_points
        return articulation_points
    
    def get_valid_block_moves(self, state):
        """
        Generate valid moves for the entire block of elements
        A valid block move shifts all elements in the same direction while maintaining connectivity
        """
        valid_moves = []
        state_list = list(state)
        
        # Try moving the entire block in each direction
        for dx, dy in self.directions:
            # Calculate new positions after moving
            new_positions = [(pos[0] + dx, pos[1] + dy) for pos in state_list]
            
            # Check if all new positions are valid (within bounds and not occupied)
            all_valid = all(0 <= pos[0] < self.grid_size[0] and 
                            0 <= pos[1] < self.grid_size[1] for pos in new_positions)
            
            # Only consider moves that keep all positions within bounds
            if all_valid:
                new_state = frozenset(new_positions)
                valid_moves.append(new_state)
        
        return valid_moves
    
    def get_valid_morphing_moves(self, state):
        """
        Generate valid morphing moves that maintain connectivity
        Uses a priority queue to focus on promising moves first
        """
        state_key = hash(state)
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
            
        valid_moves = []
        state_set = set(state)
        
        # Find non-critical points that can move without breaking connectivity
        articulation_points = self.get_articulation_points(state_set)
        movable_points = state_set - articulation_points
        
        # If no non-critical points (all are articulation points), try carefully moving articulation points
        if not movable_points and articulation_points:
            for pos in articulation_points:
                # Try removing this point, check if structure remains connected
                temp_state = state_set - {pos}
                if self.is_connected(temp_state):  # If still connected without this point
                    movable_points.add(pos)  # This articulation point can be safely moved
        
        # For each movable point, find valid target positions that maintain connectivity
        positions_to_try = movable_points if movable_points else state_set
        
        # Prioritize moves that get cells closer to the goal state
        priority_queue = []
        for pos in positions_to_try:
            # Find the closest goal position for this point
            min_dist_to_goal = float('inf')
            for goal_pos in self.goal_state:
                dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                if dist < min_dist_to_goal:
                    min_dist_to_goal = dist
            
            # Add to priority queue with the minimum distance as priority
            heapq.heappush(priority_queue, (min_dist_to_goal, pos))
        
        # Process movable points by priority
        while priority_queue:
            _, pos = heapq.heappop(priority_queue)
            
            # Try to move this point in each direction
            for dx, dy in self.directions:
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                # Skip if out of bounds or already occupied
                if not (0 <= new_pos[0] < self.grid_size[0] and 
                        0 <= new_pos[1] < self.grid_size[1]):
                    continue
                if new_pos in state_set:
                    continue
                
                # Create new state with this element moved
                new_state_set = state_set.copy()
                new_state_set.remove(pos)
                new_state_set.add(new_pos)
                
                # Check if the new position is adjacent to at least one other element
                has_neighbor = False
                for dx2, dy2 in self.directions:
                    neighbor = (new_pos[0] + dx2, new_pos[1] + dy2)
                    if neighbor in new_state_set and neighbor != pos:
                        has_neighbor = True
                        break
                
                # Only consider moves that maintain connectivity
                if has_neighbor and self.is_connected(new_state_set):
                    new_state_frozen = frozenset(new_state_set)
                    valid_moves.append(new_state_frozen)
        
        # Cache and return results
        self.valid_moves_cache[state_key] = valid_moves
        return valid_moves
    
    def get_optimized_morphing_moves(self, state):
        """
        Generate more optimized morphing moves by considering multiple element moves
        that maintain connectivity and focus on moving toward goal positions
        """
        # Get basic single-element moves
        single_moves = self.get_valid_morphing_moves(state)
        state_set = set(state)
        
        # Try to find valid multi-element moves (up to 2 elements at once)
        # This helps navigate tight spaces and overcome local minima
        multi_moves = []
        
        # Find pairs of elements that can move together
        non_critical_points = state_set - self.get_articulation_points(state_set)
        
        # If we have enough non-critical points, try moving pairs
        if len(non_critical_points) >= 2:
            point_list = list(non_critical_points)
            for i in range(min(len(point_list), 5)):  # Limit to 5 points for efficiency
                pos1 = point_list[i]
                
                # Try moving this element in each direction
                for dx1, dy1 in self.directions:
                    new_pos1 = (pos1[0] + dx1, pos1[1] + dy1)
                    
                    # Skip if invalid
                    if not (0 <= new_pos1[0] < self.grid_size[0] and 
                            0 <= new_pos1[1] < self.grid_size[1]):
                        continue
                    if new_pos1 in state_set:
                        continue
                    
                    # Create intermediate state with first element moved
                    intermediate_state = state_set.copy()
                    intermediate_state.remove(pos1)
                    intermediate_state.add(new_pos1)
                    
                    # Skip if this breaks connectivity
                    if not self.is_connected(intermediate_state):
                        continue
                    
                    # Now try moving a second non-critical element
                    for j in range(i+1, min(len(point_list), 5)):
                        pos2 = point_list[j]
                        
                        # Don't consider pos2 if it's no longer non-critical in the intermediate state
                        if pos2 not in intermediate_state - self.get_articulation_points(intermediate_state):
                            continue
                        
                        for dx2, dy2 in self.directions:
                            new_pos2 = (pos2[0] + dx2, pos2[1] + dy2)
                            
                            # Skip if invalid
                            if not (0 <= new_pos2[0] < self.grid_size[0] and 
                                    0 <= new_pos2[1] < self.grid_size[1]):
                                continue
                            if new_pos2 in intermediate_state or new_pos2 == new_pos1:
                                continue
                            
                            # Create final state with both elements moved
                            final_state = intermediate_state.copy()
                            final_state.remove(pos2)
                            final_state.add(new_pos2)
                            
                            # Only add if connected
                            if self.is_connected(final_state):
                                multi_moves.append(frozenset(final_state))
        
        # Return combined moves
        return single_moves + multi_moves
    
    def block_heuristic(self, state):
        """
        Heuristic for block movement phase:
        Calculate Manhattan distance from current centroid to goal centroid
        """
        if not state:
            return float('inf')
            
        current_centroid = self.calculate_centroid(state)
        
        # Manhattan distance between centroids
        return abs(current_centroid[0] - self.goal_centroid[0]) + \
               abs(current_centroid[1] - self.goal_centroid[1])
    
    def morphing_heuristic(self, state):
        """
        Enhanced heuristic for morphing phase:
        Uses bipartite matching to find optimal assignment
        """
        if not state:
            return float('inf')
            
        state_list = list(state)
        goal_list = list(self.goal_state)
        
        # Early exit if states have different sizes
        if len(state_list) != len(goal_list):
            return float('inf')
        
        # Build distance matrix
        distances = []
        for pos in state_list:
            row = []
            for goal_pos in goal_list:
                # Manhattan distance
                dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                row.append(dist)
            distances.append(row)
        
        # Use Hungarian algorithm to find minimum cost assignment
        # This is a simplified implementation - for production code,
        # consider using scipy.optimize.linear_sum_assignment
        assigned = [-1] * len(state_list)
        min_cost = self.find_min_cost_assignment(distances, assigned)
        
        # Add penalty for disconnected structures
        connectivity_penalty = 0 if self.is_connected(state) else 1000
        
        return min_cost + connectivity_penalty
    
    def find_min_cost_assignment(self, cost_matrix, assignment):
        """
        Simple greedy assignment algorithm (for optimal results, use Hungarian algorithm)
        """
        n = len(cost_matrix)
        total_cost = 0
        assigned_cols = set()
        
        # For each row, find the lowest cost unassigned column
        for i in range(n):
            min_cost = float('inf')
            min_col = -1
            
            for j in range(n):
                if j not in assigned_cols and cost_matrix[i][j] < min_cost:
                    min_cost = cost_matrix[i][j]
                    min_col = j
            
            assignment[i] = min_col
            assigned_cols.add(min_col)
            total_cost += min_cost
        
        return total_cost

    def block_movement_phase(self, time_limit=15):
        """
        Phase 1: Move the entire block toward the goal centroid
        Returns the path of states to get near the goal area
        """
        print("Starting Block Movement Phase...")
        start_time = time.time()
    
        # Initialize A* search
        open_set = [(self.block_heuristic(self.start_state), 0, self.start_state)]
        closed_set = set()
    
        # Track path and g-scores
        g_score = {self.start_state: 0}
        came_from = {self.start_state: None}
    
    # Target area reached when centroid distance is small enough
        target_threshold = max(2, min(self.grid_size) // 5)  # Adaptive threshold based on grid size
    
        while open_set and time.time() - start_time < time_limit:
        # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
        
        # Skip if already processed
            if current in closed_set:
                continue
            
        # Check if we're close enough to the goal centroid
            current_centroid = self.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - self.goal_centroid[0]) + 
                            abs(current_centroid[1] - self.goal_centroid[1]))
                            
            if centroid_distance <= target_threshold:
                return self.reconstruct_path(came_from, current)
            
            closed_set.add(current)
        
        # Process neighbor states (block moves)
            for neighbor in self.get_valid_block_moves(current):
                if neighbor in closed_set:
                    continue
                
            # Calculate tentative g-score
                tentative_g = g_score[current] + 1
            
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.block_heuristic(neighbor)
                
                # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print("Block movement phase timed out!")
        
    # Return the best state we found
        if came_from:
        # Find state with minimum heuristic value
            best_state = min(came_from.keys(), key=self.block_heuristic)
            return self.reconstruct_path(came_from, best_state)
        
        return [self.start_state]  # No movement possible

    def morphing_phase(self, start_state, time_limit=15):
        """
        Phase 2: Morph the block into the final shape while maintaining connectivity
        Returns the path from the provided start state to the goal shape
        """
        print("Starting Morphing Phase...")
        start_time = time.time()
    
    # Initialize A* search
        open_set = [(self.morphing_heuristic(start_state), 0, start_state)]
        closed_set = set()
    
    # Track path and g-scores
        g_score = {start_state: 0}
        came_from = {start_state: None}
    
    # Track best solution so far for anytime behavior
        best_state = start_state
        best_score = self.morphing_heuristic(start_state)
    
        while open_set and time.time() - start_time < time_limit:
        # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
        
        # Skip if already processed
            if current in closed_set:
                continue
            
        # Check if this is the best state so far
            current_score = self.morphing_heuristic(current)
            if current_score < best_score:
                best_score = current_score
                best_state = current
            
        # Check if goal reached
            if current == self.goal_state:
                return self.reconstruct_path(came_from, current)
            
            closed_set.add(current)
        
        # Process neighbor states - use optimized morphing moves to maintain connectivity
        # This is the key change to ensure blocks stay connected during morphing
            for neighbor in self.get_optimized_morphing_moves(current):
                if neighbor in closed_set:
                    continue
                
            # Calculate tentative g-score
                tentative_g = g_score[current] + 1
            
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.morphing_heuristic(neighbor)
                
                # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print("Morphing phase timed out!")
    
    # Return the best solution found so far
        return self.reconstruct_path(came_from, best_state)

    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from start to goal.
        Returns a list of states (each state is a set of positions).
        """
        path = []
        while current:
        # Convert frozenset to list of tuples for visualization
            path.append([pos for pos in current])
            current = came_from.get(current)
    
        path.reverse()  # Start to goal
        return path
    
    def search(self, time_limit=30):
        """
        Main search method implementing the two-phase approach
        with improved connectivity maintenance during morphing
        """
    # Allocate time for each phase
        block_time_limit = time_limit * 0.4  # 40% of time for block movement
        morphing_time_limit = time_limit * 0.6  # 60% of time for morphing
    
    # Phase 1: Block Movement
        block_path = self.block_movement_phase(block_time_limit)
    
        if not block_path:
            print("Block movement phase failed to find a path")
            return None
        
    # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])
    
    # Phase 2: Morphing with connectivity maintenance
        morphing_path = self.morphing_phase(block_final_state, morphing_time_limit)
    
        if not morphing_path:
            print("Morphing phase failed to find a path")
            return block_path  # Return just the block movement path
        
    # Combine paths (remove duplicate state at transition)
        combined_path = block_path[:-1] + morphing_path
    
        return combined_path

    def visualize_path(self, path, interval=0.5):
        """
        Visualize the path as an animation
        """
        if not path:
            print("No path to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.ion()  # Turn on interactive mode
    
    # Get bounds for plotting
        min_x, max_x = 0, self.grid_size[0] - 1
        min_y, max_y = 0, self.grid_size[1] - 1
    
    # Show initial state
        ax.clear()
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.grid(True)
    
    # Draw goal positions (as outlines)
        for pos in self.goal_positions:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
    
    # Draw current positions (blue squares)
        current_positions = path[0]
        rects = []
        for pos in current_positions:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
            ax.add_patch(rect)
            rects.append(rect)
        
        ax.set_title(f"Step 0/{len(path)-1}")
        plt.draw()
        plt.pause(interval)
    
    # Animate the path
        for i in range(1, len(path)):
        # Update positions
            new_positions = path[i]
        
        # Clear previous positions
            for rect in rects:
                rect.remove()
        
        # Draw new positions
            rects = []
            for pos in new_positions:
                rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
                ax.add_patch(rect)
            rects.append(rect)
            
            ax.set_title(f"Step {i}/{len(path)-1}")
            plt.draw()
            plt.pause(interval)
    
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True)