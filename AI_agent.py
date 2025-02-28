import heapq

class AI_Agent:
    def __init__(self, grid_size, start, goal, topology="moore"):
        self.grid_size = grid_size
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.topology = topology
        self.moves = self.get_moves()

    def get_moves(self):
        if self.topology == "moore":
            return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            return [(-1, 0), (0, -1), (0, 1), (1, 0)]

    def heuristic(self, pos):
        return sum(abs(px - gx) + abs(py - gy) for (px, py), (gx, gy) in zip(pos, self.goal))

    def is_valid(self, positions):
        return all(0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] for x, y in positions)

    def get_neighbors(self, current):
        neighbors = []
        for move in self.moves:
            new_positions = tuple((x + move[0], y + move[1]) for x, y in current)
            if self.is_valid(new_positions) and len(set(new_positions)) == len(current):
                neighbors.append(new_positions)
        return neighbors

    def search(self):
        open_list = []
        heapq.heappush(open_list, (0, self.start))
        came_from = {self.start: None}
        g_score = {self.start: 0}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == self.goal:
                return self.reconstruct_path(came_from)
            
            for neighbor in self.get_neighbors(current):
                temp_g_score = g_score[current] + 1
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    g_score[neighbor] = temp_g_score
                    f_score = temp_g_score + self.heuristic(neighbor)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current
        return None

    def reconstruct_path(self, came_from):
        path = []
        current = self.goal
        while current:
            path.append(current)
            current = came_from[current]
        return path[::-1]