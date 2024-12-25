import json
import random
from Mo_Star import MoStar
class DatasetGenerator:
    def __init__(self, num_cases, num_agents, grid):
        self.num_cases = num_cases
        self.num_agents = num_agents
        self.grid = grid
    
    def generate_cases(self):
        """
    Generate multiple cases for agents with random start and goal positions.

    Args:
    - num_cases: Number of cases to generate.
    - num_agents: Number of agents per case.
    - grid_size: Size of the grid (rows, cols).

    Returns:
    - List of generated cases.
    """
        num_cases = self.num_cases
        num_agents = self.num_agents
        grid = self.grid
        cases = []
        grid_size = (len(grid),len(grid))
        counter = 0
        for i in range(num_cases):
            case = {"start_positions": [], "goal_positions": [],"paths":[]}
            used_positions = set() 
            paths = [None]
            start_points = []
            goal_points = []
            while None in paths:
                # Generate random start positions
                start_points = []
                goal_points = []
                for _ in range(num_agents):
                    while True:
                        start = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
                        if start not in used_positions:
                            used_positions.add(start)
                            start_points.append(start)
                            # case["start_positions"].append(start)
                            break

                # Generate random goal positions
                for _ in range(num_agents):
                    while True:
                        goal = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
                        if goal not in used_positions:
                            used_positions.add(goal)
                            goal_points.append(goal)
                            # case["goal_positions"].append(goal)
                            break
                # Generate paths
                print("case ",i)
                planner = MoStar(grid, start_points, goal_points)
                # planner = MoStar(grid, case["start_positions"], case["goal_positions"])
                paths = planner.plan()
                counter+=1
            
            case["start_positions"] = start_points
            case["goal_positions"] = goal_points
            case["paths"]=paths
            cases.append(case)
            print("case added")
        # print(counter)
        return cases

    def save_cases_to_file(self,cases, filename):
        """Save generated cases to a file in JSON format."""
        with open(filename, 'w') as file:
            json.dump(cases, file)
    def load_cases_from_file(self,filename):
        """Load cases from a JSON file."""
        with open(filename, 'r') as file:
            return json.load(file)