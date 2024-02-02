import numpy as np

class qlAgent:
    e = 2.7183
    def __init__(self, actions, alpha=.1, gamma=.99, epsilon=.1):
        self.actions = actions # list of actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def create_q_table(states):
        n = len(states)
        list_of_dimensions = []
        for i in range(n):
            list_of_dimensions[i].append()
        #for state in range(len(states)):


