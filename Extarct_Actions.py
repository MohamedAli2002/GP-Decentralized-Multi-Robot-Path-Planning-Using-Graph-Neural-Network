class Action_Extractor:
    def __init__(self, cases, num_of_robots):
        self.cases = cases
        self.num_of_robots = num_of_robots
    def get_max_length(self,paths):
        max_length = 0
        for path in paths:
            if len(path) > max_length:
                max_length = len(path)
        return max_length
    def unify_paths_length(self,paths, max_length):
        for i in range(max_length):
            for path in paths:
                if i >= len(path):
                    path.append(path[-1])
        return paths
    def extract(self):
        movements = {}
        for itr,case in enumerate(self.cases):
            paths = case['paths']
            max_len = self.get_max_length(paths)
            new_paths = self.unify_paths_length(paths,max_len)
            movements_step = {}
            for s in range(max_len-1):
                movements_robots = []
                for path in new_paths:
                    action = -1
                    if   path[s+1] == (path[s][0]-1,path[s][1]): # going left
                    if   path[s+1] == [path[s][0]-1,path[s][1]]: # going left
                        action = 1
                    elif path[s+1] == (path[s][0],path[s][1]+1): # going up
                    elif path[s+1] == [path[s][0],path[s][1]+1]: # going up
                        action = 2
                    elif path[s+1] == (path[s][0]+1,path[s][1]): # going right
                    elif path[s+1] == [path[s][0]+1,path[s][1]]: # going right
                        action = 3
                    elif path[s+1] == (path[s][0],path[s][1]-1): # going down
                    elif path[s+1] == [path[s][0],path[s][1]-1]: # going down
                        action = 4
                    elif path[s+1] == (path[s][0],path[s][1]): # staying in the same position
                    elif path[s+1] == [path[s][0],path[s][1]]: # staying in the same position
                        action = 0
                    movements_robots.append(action)
                movements_step[s] = movements_robots
            movements_step[s+1] = [0 for _ in range(self.num_of_robots)]
            movements[itr] = movements_step
        return movements