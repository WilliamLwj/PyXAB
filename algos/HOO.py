import math
import numpy as np
import pdb
from algos.Algo import Algorithm


### Implementation of truncated-HOO (Bubeck et al, 2011)


class T_HOO(Algorithm):

    def __init__(self, nu=1, rho=0.75, rounds=1000, partition=None):
        super(T_HOO, self).__init__(partition)

        self.iteration = 0
        self.nu = nu
        self.rho = rho
        self.rounds = rounds

        # List of values that are important

        self.Bvalues = [[np.inf]]
        self.Uvalues = [[np.inf]]
        self.Rewards = [[0]]
        self.visitedTimes = [[0]]
        self.visited = [[True]]
        self.expand(self.partition.get_root())

    def optTraverse(self):
        curr_node = self.partition.get_root()
        path = [curr_node]

        while curr_node.get_children() is not None:
            children = curr_node.get_children()
            maxchild = 1
            for child in children:
                depth = child.get_depth()
                index = child.get_index()

                # If the child is never visited or prepared to be visited, denote maxchild = -1
                if not self.visited[depth][index - 1]:
                    maxchild = -1
                    break
                if self.Bvalues[depth][index - 1] >= self.Bvalues[depth][maxchild - 1]:
                    maxchild = index

            # If we find that the child is never visited
            if maxchild == -1:
                break
            else:
                curr_node = self.partition.get_node(curr_node.get_depth()+1, maxchild)
                path.append(curr_node)

        return curr_node, path

    def updateRewardTree(self, path, reward):

        for node in path:
            depth = node.get_depth()
            index = node.get_index()

            # Update the visited times and the average reward of the pulled node

            self.visitedTimes[depth][index - 1] += 1
            self.Rewards[depth][index - 1] = \
                ((self.visitedTimes[depth][index - 1] - 1) / self.visitedTimes[depth][index - 1]
                * self.Rewards[depth][index - 1]) + (reward / self.visitedTimes[depth][index - 1])

        self.iteration += 1

    def updateUvalueTree(self):
        node_list = self.partition.get_node_list()
        for layer in node_list:
            for node in layer:
                depth = node.get_depth()
                index = node.get_index()

                if self.visitedTimes[depth][index - 1] == 0:
                    continue
                else:
                    UCB = math.sqrt(2 * math.log(self.rounds) / self.visitedTimes[depth][index - 1])
                    self.Uvalues[depth][index - 1] = self.Rewards[depth][index - 1] + UCB + self.nu * (self.rho ** depth)


    def updateBackwardTree(self):

        nodes = self.partition.get_node_list()

        for i in range(1, self.partition.get_depth()+1):

            layer = nodes[-i]
            for node in layer:
                depth = node.get_depth()
                index = node.get_index()

                # If no children or if children not visitied, use its own U value
                children = node.get_children()
                if children is None:
                    self.Bvalues[depth][index - 1] = self.Uvalues[depth][index - 1]
                else:
                    c_depth = children[0].depth
                    c_index = children[0].index
                    if not self.visited[c_depth][c_index]:
                        self.Bvalues[depth][index - 1] = self.Uvalues[depth][index - 1]
                    else:
                        tempB = 0
                        for child in node.get_children():
                            c_depth = child.get_depth()
                            c_index = child.get_index()
                            tempB = np.maximum(tempB, self.Bvalues[c_depth][c_index - 1])

                        self.Bvalues[depth][index - 1] = np.minimum(self.Uvalues[depth][index - 1], tempB)

    def expand(self, parent):

        if parent.get_depth() > self.partition.get_depth():
            raise ValueError
        elif parent.get_depth() == self.partition.get_depth():
            self.partition.deepen()
            num_nodes = len(self.partition.get_node_list()[-1])
            self.Uvalues.append([np.inf] * num_nodes)
            self.Bvalues.append([np.inf] * num_nodes)
            self.visited.append([False] * num_nodes)
            self.visitedTimes.append([0] * num_nodes)
            self.Rewards.append([0] * num_nodes)

        children = parent.get_children()
        if children is None:
            raise ValueError
        else:
            for child in children:
                c_depth = child.get_depth()
                c_index = child.get_index()
                self.visited[c_depth][c_index - 1] = True


    def updateAllTree(self, path, reward):

        self.updateRewardTree(path, reward)
        self.updateUvalueTree()
        # Truncate or not
        if path[-1].depth <= np.ceil((np.log(self.rounds)/2 - np.log(1/self.nu))/np.log(1/self.rho)):
            self.expand(path[-1])
        self.updateBackwardTree()

    def pull(self, time):

        curr_node, self.path = self.optTraverse()
        sample_range = curr_node.get_domain()
        point = []
        for j in range(len(sample_range)):
            # uniformly sample one point
            x = (sample_range[j][0] + sample_range[j][1]) / 2
            # x = np.random.uniform(sample_range[j][0], sample_range[j][1])
            point.append(x)

        return point

    def receive_reward(self, time, reward):

        self.updateAllTree(self.path, reward)


class HOO_Node:
    def __init__(self, depth, index, parent, range):
        self.depth = depth
        self.index = index
        self.meanReward = 0
        self.visitedTimes = 0
        self.parent = parent
        self.children = None
        self.Uvalue = np.inf
        self.Bvalue = np.inf
        self.range = range

    def updateReward(self, reward):

        self.visitedTimes += 1
        self.meanReward = ((self.visitedTimes - 1) / self.visitedTimes * self.meanReward) + (reward / self.visitedTimes)

    def updateBackward(self):

        if self.children is None:
            self.Bvalue = self.Uvalue
        else:
            tempB = 0
            for child in self.children:
                tempB = np.maximum(tempB, child.Bvalue)
            self.Bvalue = np.minimum(self.Uvalue, tempB)



class HOO_tree:

    def __init__(self, nu, rho, range, rounds):
        self.root = HOO_Node(0, 1, None, range)
        self.list = [[self.root]]
        self.iteration = 0
        self.nu = nu
        self.rho = rho
        self.rounds = rounds
        self.createNewNodes(self.root)

    def optTraverse(self):
        curr_point = self.root
        path = []
        while True:
            path.append(curr_point)
            if curr_point.children is not None:
                child1 = curr_point.children[0]
                child2 = curr_point.children[1]
                if child1.Bvalue >= child2.Bvalue:
                    curr_point = child1
                else:
                    curr_point = child2
            else:
                break

        return curr_point, path

    def updateRewardTree(self, path, reward):

        for node in path:
            node.updateReward(reward)

        self.iteration += 1

    def updateUvalueTree(self):

        for layer in self.list:
            for node in layer:

                if node.visitedTimes == 0:
                    continue
                else:
                    node.Uvalue = node.meanReward + math.sqrt(2 * math.log(self.rounds) / node.visitedTimes) + \
                              self.nu * (self.rho ** node.depth)


    def updateBackwardTree(self):

        for i in range(1, len(self.list)+1):

            layer = self.list[-i]
            for node in layer:
                node.updateBackward()

    def createNewNodes(self, parent):

        dim = np.random.randint(0, len(parent.range))
        selected_dim = parent.range[dim]

        range1 = parent.range.copy()
        range2 = parent.range.copy()

        range1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1])/2]
        range2[dim] = [(selected_dim[0] + selected_dim[1])/2, selected_dim[1]]

        node1 = HOO_Node(parent.depth+1, 2 * parent.index, parent, range1)
        node2 = HOO_Node(parent.depth+1, 2 * parent.index - 1, parent, range2)

        parent.children = [node1, node2]

        if len(self.list) <= parent.depth + 1:
            self.list.append([node1, node2])
        else:
            self.list[parent.depth + 1].append(node1)
            self.list[parent.depth + 1].append(node2)

    def updateAllTree(self, path, reward):

        self.updateRewardTree(path, reward)
        self.updateUvalueTree()
        # Truncate or not
        if path[-1].depth <= np.ceil((np.log(self.rounds)/2 - np.log(1/self.nu))/np.log(1/self.rho) ):
            self.createNewNodes(path[-1])
        self.updateBackwardTree()








