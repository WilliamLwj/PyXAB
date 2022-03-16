import math
import numpy as np
import pdb



### Implementation of truncated-HOO (Bubeck et al, 2011)

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








