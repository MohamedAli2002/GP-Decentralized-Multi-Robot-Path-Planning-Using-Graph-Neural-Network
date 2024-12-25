import heapq
from collections import deque
import copy
import numpy as np
class MoStar:
    def __init__(self, grid, start_positions, goal_positions):
        self.grid = grid  # 2D list representing the grid map (0: free, 1: obstacle)
        self.start_positions = start_positions  # List of (x, y) start positions for each agent
        self.goal_positions = goal_positions  # List of (x, y) goal positions for each agent
        self.num_agents = len(start_positions)
        self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible moves: right, down, left, up
        self.t_grid = {(-1,-1,-1): -1}
        self.et = {(-1,-1,-1,-1,-1): -1}

    def is_valid(self, x, y,t):
        """Check if a position is within bounds and not an obstacle."""
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] == 0 and (x,y,t) not in self.t_grid
    
    def heuristic(self, pos, goal):
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def a_star(self, agent_id):
        """A* search for a single agent, considering dependencies."""
        start = self.start_positions[agent_id]
        goal = self.goal_positions[agent_id]
        visited = set()
        pq = []
        heapq.heappush(pq, (0,start, []))  # (cost, current_position, path)

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)
            new_path = path + [current]

            if current == goal:
                return new_path

            for move in self.moves: #[(0, 1), (1, 0), (0, -1), (-1, 0)]
                nx, ny = current[0] + move[0], current[1] + move[1]
                if self.is_valid(nx, ny,len(new_path)) and (nx, ny) not in visited:
                    if (current[0],current[1],nx,ny,len(new_path)-0.5) not in self.et and (nx,ny,current[0],current[1],len(new_path)-0.5) not in self.et:
                        new_cost = cost + 1 + self.heuristic((nx, ny), goal)
                        heapq.heappush(pq, (new_cost, (nx, ny), new_path))
                        if (nx, ny) == goal:
                            return new_path + [(nx, ny)]
        return None  # No path found
    
    def remove_from_t_grid(self,agent):
        keys_to_remove = [key for key, value in self.t_grid.items() if value == agent]
        for key in keys_to_remove:
            del self.t_grid[key]
            
    def remove_from_et(self,agent):
        keys_to_remove = [key for key, value in self.et.items() if value == agent]
        for key in keys_to_remove:
            del self.et[key]
            
    def check_confilicts(self,paths):
        if None in paths:
            return 0
        num_of_confilicts = 0
        list_of_confilicts = []
        max_len = max(paths)
        for step in range(len(max_len)):
            step_nodes = {}
            for path in paths:
                if step < len(path):
                    if (path[step][0],path[step][1],step) in step_nodes:
                        num_of_confilicts+=1
                        list_of_confilicts.append((path[step][0],path[step][1],step))
                    else:
                        step_nodes[(path[step][0],path[step][1],step)] = 1 
                else:
                    if (path[-1][0],path[-1][1],step) in step_nodes:
                        num_of_confilicts+=1
                        list_of_confilicts.append((path[-1][0],path[-1][1],step))
                    else:
                        step_nodes[(path[-1][0],path[-1][1],step)] = 1
            step_nodes.clear()
        return num_of_confilicts
    
    def resolve_confilict_in_goals_nodes_waiting(self,paths):
        all_paths = []
        for goal_index in range(self.num_agents):
            new_paths = []
            for path_index in range(self.num_agents):
                if goal_index != path_index:
                    if self.goal_positions[goal_index] in paths[path_index]:
                        if (paths[path_index].index(self.goal_positions[goal_index])) >= (len(paths[goal_index])-1):
                            self.remove_from_t_grid(path_index)
                            self.remove_from_et(path_index)
                            path = self.a_star(path_index)
                            if path != None:
                                size = len(path)
                                for i in range(len(self.grid)*5):
                                    if i < size:
                                        pos = path[i]
                                    else:
                                        pos = path[-1]
                                    self.t_grid[(pos[0],pos[1],i)] = path_index
                                for i in range(size-1):
                                    self.et[(path[i][0],path[i][1],path[i+1][0],path[i+1][1],i+0.5)] = path_index
                            new_paths.append(path)
                        else:
                            new_paths.append(paths[path_index])
                    else:
                        new_paths.append(paths[path_index])
                else:
                    new_paths.append(paths[path_index])
            all_paths.append(new_paths)
        return all_paths
    def plan(self):
        """Plan paths for all agents."""
        paths = []
        for agent in range(self.num_agents):
            path = self.a_star(agent)
            if path != None:
                size = len(path)
                for i in range(len(self.grid)*5):
                    if i < size:
                        pos = path[i]
                    else:
                        pos = path[-1]
                    self.t_grid[(pos[0],pos[1],i)] = agent
                for i in range(size-1):
                    self.et[(path[i][0],path[i][1],path[i+1][0],path[i+1][1],i+0.5)] = agent
            paths.append(path)
        n_conflicts = self.check_confilicts(paths)
        if None in paths:
            return [None]
        n_paths = [row[:] for row in paths]
        # print(n_paths)
        while(True):
            if n_conflicts > 0:
                x = self.resolve_confilict_in_goals_nodes_waiting(n_paths)
                con = 100
                idx = 0
                for i in range(len(x)):
                    v = self.check_confilicts(x[i])
                    if v == 0:
                        return x[i]
                    if v < con:
                        con = v
                        idx =i
                n_paths = [row[:] for row in x[idx]]
                # n_conflicts = self.check_confilicts(n_paths)
                # print(n_conflicts)
            else:
                paths = [row[:] for row in n_paths]
                break
        return paths
    