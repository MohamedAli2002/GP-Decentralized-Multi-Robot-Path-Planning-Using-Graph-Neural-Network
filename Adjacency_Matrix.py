import numpy as np
class adj_mat:
    def __init__(self, cases, rfov):
        self.cases = cases
        self.rfov = rfov
    def get_max_length(self,paths):
        max_length = 0
        for path in paths:
            if len(path) > max_length:
                max_length = len(path)
        return max_length
    def get_adj_mat(self):
        num_of_agents = self.cases[0]['paths']
        matrices = {}
        for itr,case in enumerate(self.cases):
            paths = self.cases[itr]['paths']
            max_len = self.get_max_length(paths)
            # print(max_len)
            matrices_steps = {}
            for s in range(max_len):
                matrix = np.zeros((len(num_of_agents),len(num_of_agents)))
                for i in range(len(num_of_agents)-1):
                    for j in range(i+1,len(num_of_agents)):
                        if s >= len(paths[i]):
                            node_i = paths[i][-1]
                        else:
                            node_i = paths[i][s]
                        if s >= len(paths[j]):
                            node_j = paths[j][-1]
                        else:
                            node_j = paths[j][s]
                        if abs(node_i[0]-node_j[0]) <= self.rfov and abs(node_i[1]-node_j[1]) <= self.rfov:

                            matrix[i][j] = 1
                            matrix[j][i] = 1
                matrices_steps[s] = matrix
            matrices[itr] = matrices_steps
        return matrices