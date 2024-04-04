import numpy as np
from gridworld import GridWorld

class Agent:
    def __init__(self, alpha=.1, gamma=.99, epsilon=.1, double = False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.EXP = 2.7183
        self.double = double
        if double == True:
            self.__currentQTable = 1
            self.__best_action = 0


    def setEnvironment(self, environment:GridWorld):
        self.environment = environment
        self.setPossibleStates()

    def setPossibleStates(self):
        self.states_ = np.arange(self.environment.rows * self.environment.cols * 4 * 16) # 4 devido aos angulos)
        obstacle = self.environment.obstacles
        aux = [self.environment.s2c(obs) for obs in obstacle]
        aux2 = list()
        for item in aux:
            for angle in range(4):
                for g1 in range(2):
                    for g2 in range(2):
                        for g3 in range(2):
                            for g4 in range(2):
                                aux2.append(self.environment.cart2s((g1, g2, g3, g4, item[1], item[0], angle)))
        self.states_ = np.delete(self.states_, aux2)
        print('oi')

    def removeStates(self, states):
        self.states_ = np.setdiff1d(self.states_, states)

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
            # y = ae^(-bx)
            a = intervals[0]
            b =  - np.log(intervals[1]) / intervals[2]
            self.epsilonFunction = lambda episode: max(a * self.EXP**(-b * episode), intervals[1])
        else:
            self.epsilonFunction = lambda episode: self.epsilon


    def setAlpha(self, mode = 0, intervals = [1, .1, 500]):
        '''
        @param: mode: 0, 1 or 2
        @param intervals: [max, min, totalEpisodes]
        '''
        if mode == 0:
            self.alphaFunction = lambda episode: self.alpha
        elif mode == 1:
            self.alphaFunction = lambda episode: max(intervals[0] - intervals[0]/intervals[2] * episode, intervals[1])
        elif mode == 2:
            # y = ae^(-bx)
            a = intervals[0]
            b =  - np.log(intervals[1]) / intervals[2]
            self.alphaFunction = lambda episode: max(a * self.EXP**(-b * episode), intervals[1])
        else:
            self.alphaFunction = lambda episode: self.alpha

    def setQtable(self, numTotalStates, numActions):
        '''
        Create q table
        '''
        self.Q = np.zeros((numTotalStates, numActions))

    def exploringStarts(self, rows, cols, origin):
        '''
        Set of possible states given the initial exploration constraints
        '''
        xo, yo, zo = origin
        x = np.arange(xo, xo+rows)
        y = np.arange(yo, yo+cols)
        totalStates = len(x) * len(y)
        self.states_ = np.zeros(totalStates, dtype=np.ushort)
        step = 0
        for row in x:
            for col in y:
                self.states_[step] = self.environment.cart2s((row, col, 0))
                step += 1
        #self.removeStates(self.environment.obstacles)



    def chooseAction(self, state, episode):
        '''
        chooses a action-state based on possible action-states
        '''
        if self.double == True:
            self.Q = self.Q1 + self.Q2

        if np.random.rand() < self.epsilonFunction(episode):
            action = np.random.choice(self.environment.actions)
            return action
        else:
            action = np.argmax(self.Q[state,:])
            return action

    def chooseBestAction(self, state):
        if self.double == True:
            self.Q = self.Q1 + self.Q2
        return np.argmax(self.Q[state,:])

    def chooseInitialState(self):
        '''
        chooses a random initial state based on possible states
        '''
        initialState = np.random.choice(self.states_)
        #vec_initial = self.environment.s2cart(initialState)
        #while vec_initial[:4] == (1, 1, 1, 1):
        #    initialState = np.random.choice(self.states_)
        #    vec_initial = self.environment.s2cart(initialState)
        self.updateEnvironment(initialState)
        return initialState


    def updateAction(self, action):
        '''
        Update action in grid world
        '''
        self.environment.current_action = action


    def move(self, action):
        '''
        Move agent in grid world
        '''
        newState, reward, done = self.environment.step(action)
        return newState, reward, done

    def updateEnvironment(self, state):
        '''
        Update position of the agent in grid world
        '''
        g1, g2, g3, g4, row, col, psi = self.environment.s2cart(state)
        self.environment.g1 = g1
        self.environment.g2 = g2
        self.environment.g3 = g3
        self.environment.g4 = g4
        self.environment.row = row
        self.environment.col = col
        self.environment.psi = psi
        self.environment.last_action = self.environment.current_action


    def updateQTable(self, state, action, reward, newState,done, episode):
        '''
        Update q-table
        '''
        self.Q[state, action] = (1 - self.alphaFunction(episode)) * self.Q[state, action] + self.alphaFunction(episode) * (reward + (1-done)*self.gamma * max(self.Q[newState, :]))

    def updateDQtable(self, state, action, reward, newState, done, episode):

        if np.random.rand() <= .5:
            self.__currentQTable = 1
            self.__best_action = np.argmax(self.Q1[newState, :])
            self.Q1[state, action] = (1 - self.alphaFunction(episode)) * self.Q1[state, action] + self.alphaFunction(episode) * (reward + (1-done)*self.gamma * self.Q2[newState, self.__best_action])
        else:
            self.__currentQTable = 2
            self.__best_action = np.argmax(self.Q2[state, :])
            self.Q2[state, action] = (1 - self.alphaFunction(episode)) * self.Q2[state, action] + self.alphaFunction(episode) * (reward + (1-done)*self.gamma * self.Q1[newState, self.__best_action])



    def reset(self,start=0):
        '''
        reset environment
        '''
        self.environment.reset(start)

    def plot(self, scorePerEpisode, stepsPerEpisode, TDerrorPerEpisode):
        '''
        this method plots to statistics during training, like: score per episode, steps per episode and 50 periods moving average
        '''
        import matplotlib.pyplot as plt

        moving_average = lambda data, periods: np.convolve(data, np.ones(periods), 'valid') / periods

        period = 50
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        ax1.plot(moving_average(TDerrorPerEpisode, period))
        ax1.set_title("TD error per episode")
        ax2.plot(moving_average(stepsPerEpisode, period))
        ax2.set_title("Steps per episode")
        ax3.plot(moving_average(scorePerEpisode, period))
        ax3.set_title("Reward per episode")
        fig.tight_layout()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        plt.show()



    def train(self, numEpisodes, frequency, plot = False):
        '''
        This method train the agent
        '''
        if self.double == 0:
            update = self.updateQTable
        else:
            update = self.updateDQtable
        scorePerEpisode = np.zeros(numEpisodes)
        stepsPerEpisode = np.zeros(numEpisodes)
        TDerrorPerEpisode = np.zeros(numEpisodes)
        for episode in range(numEpisodes):
            step = 0
            score = 0
            done = False
            TDerror = 0
            self.reset()
            print(f'Episode {episode}', end='\r')
            if episode % frequency == 0:
                self.chooseInitialState()
            while done == False:
                g1, g2, g3, g4 = self.environment.g1, self.environment.g2, self.environment.g3, self.environment.g4
                state = self.environment.cart2s((g1, g2, g3, g4, self.environment.row, self.environment.col, self.environment.psi))
                action = self.chooseAction(state, episode)
                self.updateAction(action)
                newState, reward, done = self.move(action)
                newState = self.environment.cart2s(newState)
                TDerror += self.__getTDerror(state, action, reward, newState, done, plot)
                update(state, action, reward, newState, done, episode)
                state = newState
                self.updateEnvironment(state)
                step += 1
                score += reward
            scorePerEpisode[episode] = score
            stepsPerEpisode[episode] = step
            TDerrorPerEpisode[episode] = TDerror
        if self.double == True:
            self.Q = self.Q1 + self.Q2
        if plot == True:
            self.plot(scorePerEpisode, stepsPerEpisode, TDerrorPerEpisode)

        np.savetxt('qtable.txt', self.Q)

    def __getTDerror(self, state, action, reward, newState, done, plot):
        if plot == False:
            return 0
        if self.double == True:
            if self.__currentQTable == 1:
                return reward + (1 - done) * self.gamma * self.Q2[newState, self.__best_action] - self.Q1[state, action]
            else:
                return reward + (1 - done) * self.gamma * self.Q1[newState, self.__best_action] - self.Q2[state, action]

        return reward + (1 - done) * self.gamma * max(self.Q[newState, :]) - self.Q[state, action]

    def getQtable(self):
        '''
        Return q table
        '''
        return self.Q

    def getBestPath(self, startPos):
        path = self.environment.getBestPath(startPos, self.Q)
        path.insert(0, self.environment.cart2s(startPos))
        return path

    def getStats(self, startPos):
        print(f'Fail rate: {self.environment.testConvergence(self.Q)}%' )
        print('\nQ learning stats:')
        print('Lenght: %.2f' %self.environment.get_distance(self.getBestPath(startPos)))
        print('Mean Distance: %.2f ' % self.environment.get_meanDist(self.getBestPath(startPos)))


        print('\nDijkstra stats:')
        print('Lenght: %.2f' % self.environment.get_distance(self.getDijkstraPath(startPos)))
        print('Mean Distance: %.2f' % self.environment.get_meanDist(self.getDijkstraPath(startPos)))

    def getRank(self, path):
        distanceRank = self.getRankDist(path)
        energyRank = self.getRankEnergy(path)
        proximityRank = self.getRankObstaclesProximity(path)
        return distanceRank, energyRank, proximityRank

    def getRankDist(self, path):
        xo, yo, zo = self.environment.s2cart(path[0])
        xf, yf, zf = self.environment.s2cart(path[-1])
        lengthPath = self.environment.get_distance(path)
        euclideanDistance = np.linalg.norm([xf - xo, yf - yo])
        return euclideanDistance/lengthPath

    def getRankEnergy(self, path):
        last_action = self.chooseBestAction(path[0])
        current_action = 0
        energyCost = 0
        for index_state in range(1, len(path) - 1):
            current_action = self.chooseBestAction(path[index_state])
            energyCost += self.__getEnergyCost(last_action, current_action)
            last_action = current_action
        return energyCost

    def getRankObstaclesProximity(self, path):
        return self.environment.get_meanDist(path)

    def __getEnergyCost(self, last_action, current_action):
        energyCost = 0
        if last_action != current_action:
            try:
                energyCost += self.environment.energyCost[(last_action, current_action)]
            except:
                energyCost += self.environment.energyCost[(current_action, last_action)]
        return energyCost

    def __OneStep(self, initialState: int) -> list:
        maxsteps = 100
        step = 0
        state = initialState
        path = [initialState]
        last_state = 66
        done = False
        self.environment.reset()
        while done == False and last_state != state:
            last_state = state
            self.updateEnvironment(state)
            action = self.chooseBestAction(state)
            self.updateAction(action)
            newState, reward, done = self.move(action)
            newState = self.environment.cart2s(newState)
            state = newState
            self.updateEnvironment(state)
            step += 1
            if last_state != newState:
                path.append(state)
            else:
                return []
        return path

    def FindInPolicy(self, position : tuple, size : int = 1) -> dict:
        x, y, z = position
        x_possibles = np.arange(x - size, x + size + 1)
        y_possibles = np.arange(y - size, y + size + 1)
        z_possibles = np.arange(z - size, z + size + 1)
        possibles_states = []
        for x in x_possibles:
            for y in y_possibles:
                if x >= 0 and y >=0:
                    possibles_states.append(self.environment.cart2s((x, y, 0)))

        for state in possibles_states:
            if state in self.environment.obstacles:
                possibles_states = np.setdiff1d(possibles_states, state)

        possibles_states = np.setdiff1d(possibles_states, self.environment.cart2s(position))

        states = []
        step = 0
        for state in possibles_states:
            if np.array(self.__OneStep(state)) != []:
                states.append(np.array(self.__OneStep(state)))
        return self.__rankPath(states)

    def __rankPath(self, paths : list) -> dict:
        rank_path = dict()
        for i, path in enumerate(paths):
            rank_path[i] = {path: self.getRank(path)}

if __name__ == '__main__':
    env = GridWorld(9,9,(0,0,0), -1, -1, 10)
    #env.set_obstacles([6, 8, 16, 18])
    env.set_obstacles([1, 3, 5, 7,
                       19, 21, 23, 25,
                       37, 39, 41, 43,
                       55, 57, 59, 61,
                       73, 75, 77, 79])
    #env.set_goals([3, 11, 13, 24])
    env.set_goals([10, 32, 46, 80])

    agent = Agent()
    agent.setEnvironment(env)
    agent.setQtable(9*9*4*16, 3)
    agent.setEpsilon(1, [1, .1, 50000])
    agent.setAlpha()

    agent.train(50000, 10, 1)
    print('oi')