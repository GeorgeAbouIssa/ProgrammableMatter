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
        Generate valid morphing moves where individual elements move
        while maintaining connectivity of the entire structure
        """
        state_key = hash(state)
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
            
        valid_moves = []
        state_set = set(state)
        
        # Try moving each element in each direction
        for pos in state:
            for dx, dy in self.directions:
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                # Check if the new position is valid (within bounds and not occupied)
                if (0 <= new_pos[0] < self.grid_size[0] and 
                    0 <= new_pos[1] < self.grid_size[1] and 
                    new_pos not in state_set):
                    
                    # Create new state with this element moved
                    new_state_set = state_set.copy()
                    new_state_set.remove(pos)
                    new_state_set.add(new_pos)
                    
                    # Only add if the new state maintains connectivity
                    if self.is_connected(new_state_set):
                        new_state_frozen = frozenset(new_state_set)
                        valid_moves.append(new_state_frozen)
        
        # Cache result
        self.valid_moves_cache[state_key] = valid_moves
        return valid_moves
    
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
        Heuristic for morphing phase:
        Sum of minimum distances of each element to an unassigned goal position
        """
        if not state:
            return float('inf')
            
        state_list = list(state)
        total = 0
        
        # Create a mapping of current state positions to closest unassigned targets
        unassigned_targets = list(self.goal_state)
        
        for pos in state_list:
            best_dist = float('inf')
            best_target = None
            
            for target in unassigned_targets:
                dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                if dist < best_dist:
                    best_dist = dist
                    best_target = target
                    
            if best_target:
                total += best_dist
                unassigned_targets.remove(best_target)
                
        return total
    
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
        Phase 2: Morph the block into the final shape
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
        
        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            # Check if goal reached
            if current == self.goal_state:
                return self.reconstruct_path(came_from, current)
                
            closed_set.add(current)
            
            # Process neighbor states (morphing moves)
            for neighbor in self.get_valid_morphing_moves(current):
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
            
        # Return the best state we found
        if came_from:
            # Find state with minimum heuristic value
            best_state = min(came_from.keys(), key=self.morphing_heuristic)
            return self.reconstruct_path(came_from, best_state)
            
        return [start_state]  # No movement possible
    
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
        
        # Phase 2: Morphing
        morphing_path = self.morphing_phase(block_final_state, morphing_time_limit)
        
        if not morphing_path:
            print("Morphing phase failed to find a path")
            return block_path  # Return just the block movement path
            
        # Combine paths (remove duplicate state at transition)
        combined_path = block_path[:-1] + morphing_path
        
        return combined_path
