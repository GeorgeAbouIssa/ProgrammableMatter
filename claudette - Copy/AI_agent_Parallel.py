import heapq
from collections import defaultdict
import time
import itertools

class AI_Agent_Parallel:
    def __init__(self, grid_size, start, goal, topology="moore", max_parallel_moves=6):
        self.grid_size = grid_size
        self.start_positions = list(start)  # Keep original format for reference
        self.goal_positions = list(goal)    # Keep original format for reference
        self.topology = topology
        self.max_parallel_moves = max_parallel_moves  # Maximum number of elements that can move in parallel
        
        # Convert positions to more efficient representation for path finding
        self.start_state = frozenset((x, y) for x, y in start)
        self.goal_state = frozenset((x, y) for x, y in goal)
        
        # Pre-compute and store element-target assignments
        self.assignments = self.compute_optimal_assignments()
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # Set moves based on topology
        if self.topology == "moore":
            self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # Von Neumann
            self.directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    def compute_optimal_assignments(self):
        """
        Compute optimal assignments of start positions to goal positions,
        prioritizing minimizing total Manhattan distance.
        Uses a greedy approach that works well for this specific problem.
        """
        # Sort goal positions by Y coordinate (top-down)
        sorted_goals = sorted(self.goal_positions, key=lambda pos: (-pos[0], pos[1]))
        
        # Sort start positions to prioritize ones that are easy to move first
        # (typically those at the edges)
        sorted_starts = sorted(self.start_positions, 
                               key=lambda pos: (min(pos[0], self.grid_size[0]-1-pos[0]) + 
                                               min(pos[1], self.grid_size[1]-1-pos[1])))
        
        assignments = {}
        for start_pos in sorted_starts:
            # Find the closest unassigned goal position
            best_goal = None
            best_dist = float('inf')
            
            for goal_pos in sorted_goals:
                dist = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_goal = goal_pos
                    
            # Assign this start position to the best goal
            if best_goal:
                assignments[(start_pos[0], start_pos[1])] = (best_goal[0], best_goal[1])
                sorted_goals.remove(best_goal)
                
        return assignments

    def heuristic(self, state):
        """
        Calculate an admissible heuristic for the A* search.
        Uses Manhattan distance with weighted priorities for top positions.
        """
        if not state:
            return float('inf')
            
        state_list = list(state)
        total = 0
        
        # Create a mapping from current state positions to their closest unassigned targets
        unassigned_targets = list(self.goal_state)
        assigned_targets = {}
        
        for pos in state_list:
            best_target = None
            best_dist = float('inf')
            
            for target in unassigned_targets:
                dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                if dist < best_dist:
                    best_dist = dist
                    best_target = target
                    
            if best_target:
                assigned_targets[pos] = best_target
                unassigned_targets.remove(best_target)
                
                # Top positions (smaller x) get higher priority weight
                priority = 1.0 + 0.1 * (self.grid_size[0] - best_target[0])
                total += best_dist * priority
                
        return total

    def get_valid_parallel_moves(self, state):
        """
        Generate all valid parallel moves from the current state.
        Returns a list of next states where multiple elements can move simultaneously.
        """
        state_key = hash(state)
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
            
        # First, get all possible individual element moves
        state_set = set(state)
        element_moves = {}  # Maps positions to their possible moves
        
        for pos in state:
            possible_moves = []
            for dx, dy in self.directions:
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                # Check if the new position is valid (within bounds and not occupied)
                if (0 <= new_pos[0] < self.grid_size[0] and 
                    0 <= new_pos[1] < self.grid_size[1] and 
                    new_pos not in state_set):
                    possible_moves.append(new_pos)
            
            if possible_moves:
                element_moves[pos] = possible_moves
        
        # Generate all valid combinations of parallel moves
        valid_moves = []
        
        # Iterate through different numbers of elements moving in parallel
        for num_moving in range(1, min(len(element_moves) + 1, self.max_parallel_moves + 1)):
            # Consider all possible combinations of elements to move
            for elements_to_move in itertools.combinations(element_moves.keys(), num_moving):
                # For each combination, consider all possible move combinations
                move_options = [element_moves[pos] for pos in elements_to_move]
                
                # Generate all combinations of these moves
                for move_combination in itertools.product(*move_options):
                    # Check if this combination would result in overlapping positions
                    new_positions = set(move_combination)
                    if len(new_positions) != len(move_combination):
                        continue  # Skip if there are overlaps
                    
                    # Create the new state
                    new_state = state_set.copy()
                    for i, pos in enumerate(elements_to_move):
                        new_state.remove(pos)
                        new_state.add(move_combination[i])
                    
                    # Convert to frozenset for immutability
                    new_state_frozen = frozenset(new_state)
                    valid_moves.append(new_state_frozen)
        
        # Cache result
        self.valid_moves_cache[state_key] = valid_moves
        return valid_moves

    def a_star_search(self, time_limit=30):
        """
        A* search algorithm with parallel movement capability.
        Returns the path from start to goal if found within time limit.
        """
        start_time = time.time()
        
        open_set = [(self.heuristic(self.start_state), 0, self.start_state)]
        heapq.heapify(open_set)
        closed_set = set()
        
        # Track path and g-scores
        g_score = {self.start_state: 0}
        came_from = {self.start_state: None}
        
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
            
            # Process neighbor states with parallel moves
            for neighbor in self.get_valid_parallel_moves(current):
                if neighbor in closed_set:
                    continue
                    
                # Calculate tentative g-score (cost is 1 per step, regardless of how many elements move)
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print("Search timed out!")
        return None

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
        """Main search method - interface to maintain compatibility"""
        return self.a_star_search(time_limit)
