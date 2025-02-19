import numpy as np
class Preprocessing:
    def __init__(self, grid, cases, r_fov):
        self.cases = cases
        self.r_fov = r_fov
        self.grid = grid
        obstacles = []
        self.num_agents = len(cases[0]["paths"])
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    obstacles.append((i, j))
        self.obstacles = obstacles
    def get_channel_3(self,case,s):
        channels = []
        #create grid
        grid_step = np.zeros((len(self.grid),len(self.grid)))
        nodes = []
        for path in self.cases[case]['paths']:
            # print(path[step])
            if s < len(path):
                grid_step[path[s][0]][path[s][1]] = 1
                nodes.append((path[s][0],path[s][1]))
            else:
                grid_step[path[-1][0]][path[-1][1]] = 1
                nodes.append((path[-1][0],path[-1][1]))
        #get channel3
        for node in nodes:
            i,j = node
            channel = np.zeros((((self.r_fov*2)+1),((self.r_fov*2)+1)))
            for k in range((self.r_fov*2)+1): # iterate on every node in the created window
                for l in range((self.r_fov*2)+1):
                    if (i-self.r_fov+k) >= 0 and (i-self.r_fov+k) < len(grid_step) and (j-self.r_fov+l)>=0 and (j-self.r_fov+l) < len(grid_step): # check if node is in the grid
                        channel[k][l] = grid_step[i-self.r_fov+k][j-self.r_fov+l] 
                    else:# node is out of the grid so we set it to 1
                        channel[k][l] = 1
            # print(channel)
            # print((i,j))
            np_channel = np.array(channel)
            padding_channel = np.pad(np_channel, pad_width=1, mode='constant', constant_values=0)
            channels.append(padding_channel)
        return channels
    def get_channel_2(self,case,s):
        channels = []
        nodes = []
        for path in self.cases[case]['paths']:
            if s < len(path):
                nodes.append((path[s][0],path[s][1]))
            else:
                nodes.append((path[-1][0],path[-1][1]))
        #get channel2
        for n in range(len(nodes)):
            i,j = nodes[n]
            # print(nodes[n])
            goal_x,goal_y = self.cases[case]['goal_positions'][n]
            channel = np.zeros((((self.r_fov*2)+3),((self.r_fov*2)+3)))
            if abs(i-goal_x) > self.r_fov+1 or abs(j-goal_y) > self.r_fov+1:
                # print("(i,j) = ",(i,j))
                direction_to_goal = [goal_x-i,goal_y-j]
                # print("direction_to_goal= ",direction_to_goal)
                abs_direction_to_goal = [abs(goal_x-i),abs(goal_y-j)]
                # print("abs_direction_to_goal= ",abs_direction_to_goal)
                normalized_direction = (direction_to_goal[0]/max(abs_direction_to_goal),direction_to_goal[1]/max(abs_direction_to_goal))
                # print("normalized_direction= ",normalized_direction)
                projection_goal = (i+((self.r_fov+1)*normalized_direction[0]),j+((self.r_fov+1)*normalized_direction[1]))
                # print("projection_goal= ",projection_goal)
                boundary_position = (round(projection_goal[0]),round(projection_goal[1]))
                # print("boundary_position= ",boundary_position)
                difference = (boundary_position[0]-i,boundary_position[1]-j)
                # print("difference= ",difference)
                position_on_channel = ((self.r_fov+1)+difference[0],(self.r_fov+1)+difference[1])
                # print(position_on_channel)
                channel[position_on_channel[0]][position_on_channel[1]] = 1
            else:
                difference = (goal_x-i,goal_y-j)
                position_on_channel = ((self.r_fov+1)+difference[0],(self.r_fov+1)+difference[1])
                channel[position_on_channel[0]][position_on_channel[1]] = 1
            # print(channel)
            # print((i,j))
            channels.append(channel)
        return channels
    def get_channel_1(self,case,s):
        channels = []
        #create grid
        # grid_step = np.zeros((len(self.grid),len(self.grid)))
        nodes = []
        for path in self.cases[case]['paths']:
            # print(path[step])
            if s < len(path):
                nodes.append((path[s][0],path[s][1]))
            else:
                nodes.append((path[-1][0],path[-1][1]))
        #get channel1
        for node in nodes:
            i,j = node
            channel = np.zeros((((self.r_fov*2)+1),((self.r_fov*2)+1)))
            for k in range((self.r_fov*2)+1): # iterate on every node in the created window
                for l in range((self.r_fov*2)+1):
                    if (i-self.r_fov+k) >= 0 and (i-self.r_fov+k) < len(self.grid) and (j-self.r_fov+l)>=0 and (j-self.r_fov+l) < len(self.grid): # check if node is in the grid
                        channel[k][l] = self.grid[i-self.r_fov+k][j-self.r_fov+l]
                    else:# node is out of the grid so we set it to 1
                        channel[k][l] = 1
            # print(channel)
            # print((i,j))
            channels.append(np.pad(channel, pad_width=1, mode='constant', constant_values=0))
        return channels
    def get_max_length(self,paths):
        max_length = 0
        for path in paths:
            if len(path) > max_length:
                max_length = len(path)
        return max_length
    def begin(self):
        cases_tensors = {}
        for c in range(len(self.cases)):# iterate on every case to get the channels
            channels_1_all_agents_all_steps = {}
            channels_2_all_agents_all_steps = {}
            channels_3_all_agents_all_steps = {}
            # max_path = len(max(self.cases[c]['paths']))
            max_path = self.get_max_length(self.cases[c]['paths'])
            for step in range(max_path):# iterate on every step to get the matrices              
                channels_1_all_agents_all_steps[step] = self.get_channel_1(c,step)
                channels_2_all_agents_all_steps[step] = self.get_channel_2(c,step)
                channels_3_all_agents_all_steps[step] = self.get_channel_3(c,step)
        
            cases_tensors[c] = {'channel 1':channels_1_all_agents_all_steps,
                                'channel 2':channels_2_all_agents_all_steps,
                                'channel 3':channels_3_all_agents_all_steps}
        return cases_tensors