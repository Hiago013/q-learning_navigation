import numpy as np
from itertools import product
class qlAgent:
    e = 2.7183
    def __init__(self, actions, alpha=.1, gamma=.99, epsilon=.1):
        self.actions = actions # list of actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def create_q_table(self, states):
        self.q_table = dict()
        n = len(states)
        list_of_dimensions = []
        # initializing list of list
        for i in range(n):
            list_of_dimensions.append(np.arange(states[i]))

        # printing lists
        print ("The original states are : " + str(list_of_dimensions))

        # using itertools.product()
        # to compute all possible permutations
        res = list(product(*list_of_dimensions))

        # Filling q_table
        for state in res:
            self.q_table[state] = np.zeros(len(self.actions))

    def setEpsilon(self, mode = 0, intervals = [1, .1, 500]):
        '''
        @param: mode: 0, 1 or 2
        @param intervals: [max, min, totalEpisodes]
        '''
        if mode == 0:
            self.epsilonFunction = lambda episode: self.epsilon
        elif mode == 1:
            self.epsilonFunction = lambda episode: max(intervals[0] - intervals[0]/intervals[2] * episode, intervals[1])
        elif mode == 2:
            a = intervals[0]
            b =  - np.log(intervals[1]) / intervals[2]
            self.epsilonFunction = lambda episode: max(a * self.EXP**(-b * episode), intervals[1])
        else:
            self.epsilonFunction = lambda episode: self.epsilon

    def chooseAction(self, state, episode):
        if np.random.rand() < self.epsilonFunction(episode):
            action = np.random.choice(self.actions)
            return action
        else:
            action = np.argmax(self.q_table[state])
            return action




if __name__ == "__main__":
    a = qlAgent([0, 1, 2, 3])
    c = (2,2,3)
    a.create_q_table(c)
    a.setEpsilon()
    print(a.chooseAction((0,0,0), 490))
    a.q_table[(0,0,0)][0] = 1
    print(a.q_table.keys())
    