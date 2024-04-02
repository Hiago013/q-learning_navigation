import numpy as np

class GridWorld:
    ANGLES2STATE = {0  : 0,
                    90 : 1,
                    180: 2,
                    -90: 3}

    STATE2ANGLE = {0 : 0,
                   1: 90,
                   2: 180,
                   3: -90}
    def __init__(self, rows, cols, start, Ks, Kt, Kg):
        '''
        Defines Rewards according to 3 parameters
            Ks --> Constant reward  for each step
            Kt --> Constant reward for changing routes this should encourage straigh lines
            Kg --> Reward for reaching the goal
            start --> Initial position (x, y, z)
        '''
        #Define reward
        self.Ks = Ks
        self.Kt = Kt
        self.Kg = Kg
        self.reward = 0

        # set grid world size
        self.rows = rows #number of rows, so if rows=4, we will have rows labeld from 0-3
        self.cols = cols #number of columns, so if cols=4, we will have columns labeld from 0-3

        # set initial position
        self.start = start
        self.row = start[0] #starting x
        self.col = start[1] #starting y
        self.psi = 0
        self.g1 = 0
        self.g2 = 0
        self.g3 = 0
        self.g4 = 0

        # set goal position
        self.goal = (self.rows-1,self.cols-1, 0)
        self.goals = dict()
        self.flag_goals = False

        # set done verify
        self.done = False # Means the episode hasn't ended

        # set obtacles position
        self.obstacles = []

        # set last action
        self.last_action = 0

        # list of actions taken
        self.current_action = 0

        # action space
        self.actions = [0, 1, 2]
        self.QTable=[]

        self.states = np.arange(self.rows * self.cols * 4 * 16)
        self.states = self.states.reshape([2, 2, 2, 2, self.rows, self.cols, 4])

    def get_obstacles(self):
        return self.obstacles

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def get_current_position(self):
        return self.row, self.col, self.psi

    def set_current_position(self, position:tuple):
        self.row, self.col, self.psi = position

    def c2s(self, cartesian_position):
        ''' returns the position in the cell given the 3D Cartesian coordinate
        (i, j) -> s
        '''
        i, j = cartesian_position
        s = self.cols * i + j
        return s

    def s2c(self, s_position):
        ''' returns the position in the 2D Cartesian coordinate given the cell
        s -> (i, j)
        '''
        i = int(s_position /(self.cols))
        j = s_position % (self.cols)
        return (j, i)


    def cart2s(self, cartesian_position):
        ''' returns the position in the cell given the its pose
        (row, col, psi) -> s
        '''
        return self.states[cartesian_position]

    def s2cart(self, s_position):
        ''' returns the pose in the 2D Cartesian coordinate given the cell
        s -> (row, col, psi)
        '''
        vec = np.where(self.states == s_position)
        g1, g2, g3, g4, row, col, psi = vec[0][0], vec[1][0], vec[2][0], vec[3][0], vec[4][0], vec[5][0], vec[6][0]
        return (g1, g2, g3, g4, row, col, psi)

    def set_goals(self, goals:list):
        for i, item in enumerate(goals):
            aux1, aux2 = self.s2c(item)
            self.goals[str(i)] = aux2, aux1
        self.flag_goals = True

    def next_orientation(self, current, action):
        if action == 1:
            current += 90
        elif action == 2:
            current -= 90
        else:
            raise("Ação Incorreta!\nAções permitidas: [1, 2].")

        if current > 180:
            current -= 360
        elif current == -180:
            current = 180

        return self.ANGLES2STATE[current]


    def step(self, action):
        ''' return new state, reward and boolean value to verify if the new state = goal
            0:go straight
            1:Turn Right 90
            2:Turn Left -90
        '''
        reward = 0
        row, col, psi = self.get_current_position()
        if action == 0:
            if psi == 0:
                col = col + 1
            elif psi == 1:
                row = row - 1
            elif psi == 2:
                col = col - 1
            elif psi == 3:
                row = row + 1
        elif action == 1 or action == 2:
            psi = self.next_orientation(self.STATE2ANGLE[psi], action)
        else:
            print("Ação inválida.")

        if not self.is_onboard((row, col, psi)):
            # Cancel action
            row = self.row
            col = self.col
            psi = self.psi
        # Retirei este caso para testar
        # Agora o episodio acaba quando o agente bate no obstaculo
        # E isto está implementado dentro da função get_reward
        #if self.c2s((new_row, new_col)) in self.obstacles:
        #    # Cancel action
        #    new_row =  self.row
        #    new_col =  self.col
        #    new_psi =  self.psi
        #    reward = -self.Kg
        #    self.done = True
        new_state = (self.g1, self.g2, self.g3, self.g4, row, col, psi)
        reward += self.get_reward(action, new_state) # reward of distance and energy
        new_state = (self.g1, self.g2, self.g3, self.g4, row, col, psi)

        if self.is_done(new_state):
            self.done = True
        return (new_state), reward, self.done

    def is_done(self, state):

        if (self.g1 == 1) and (self.g2 == 1) and (self.g3 == 1) and (self.g4 == 1):
            return 1


        # Comentei esses caras abaixos devido a estar utilizando mais de um destino
        #row, col, psi = state
        #row_g, col_g, psi_g = self.goal
        #if (row, col) == (row_g, col_g):
        #    return 1
        #return 0

    def get_reward(self, action, new_state):
        '''
        this function return a reward for each step
        '''
        g1, g2, g3, g4, row, col, _ = new_state
        row_g, col_g, _ = self.goal
        reward = self.Ks

        if action == 0:
            reward += 0
        else:
            reward += self.Kt

        if self.flag_goals == True:
            for i, item in enumerate(list(self.goals.values())):
                if ((row, col) == item) and i == 0 and g1 == 0:
                    self.g1 = 1
                    reward += self.Kg
                elif ((row, col) == item) and i == 1 and g2 == 0:
                    self.g2 = 1
                    reward += self.Kg

                elif ((row, col) == item) and i == 2 and g3 == 0:
                    self.g3 = 1
                    reward += self.Kg

                elif ((row, col) == item) and i == 3 and g4 == 0:
                    self.g4 = 1
                    reward += self.Kg

                if (self.g1 == 1) and (self.g2 == 1) and (self.g3 == 1) and (self.g4 == 1):
                    reward += self.Kg
                    self.done = True

        else:
            if (row, col) == (row_g, col_g):
                reward += self.Kg
                self.done = True

        if self.c2s((row, col)) in self.obstacles:
            reward -= self.Kg
            #self.done = True
        return reward


    def is_onboard(self, cartesian_position):
        '''
        checks if the agent is in the environment and return true or false
        '''
        x, y, z = cartesian_position
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return 0
        return 1

    def reset(self, start=0):
        self.g1 = self.g2 = self.g3 = self.g4 = 0
        if start==0:
            self.row = self.start[0]
            self.col = self.start[1]
            self.psi = self.start[2]
        else:
            self.row = start[0]
            self.col = start[1]
            self.psi = start[2]
        self.done = False

    def PrintBestAction(self, Q, k): # Preciso arrumar esse cara.
        # Os prints não estão corretos.
            for i in range(0, self.rows):
                print(72*'-')
                for j in range(0, self.cols):
                    if self.c2s((i,j)) in self.obstacles:
                        print('  X  ',end='|')
                    else:
                        if np.argmax(Q[self.cart2s((i,j,k)),:])==0:
                            print('  D  ',end='|')
                        elif np.argmax(Q[self.cart2s((i,j,k)),:])==1:
                            print('  U  ',end='|')
                        elif np.argmax(Q[self.cart2s((i,j,k)),:])==2:
                            print('  R  ',end='|')
                        elif np.argmax(Q[self.cart2s((i, j, k)),:])==3:
                            print('  L  ',end='|')
                        elif np.argmax(Q[self.cart2s((i, j, k)),:])==4:
                            print('  DR ',end='|')
                        elif np.argmax(Q[self.cart2s((i, j, k)),:])==5:
                            print('  DL ',end='|')
                        elif np.argmax(Q[self.cart2s((i, j, k)),:])==6:
                            print('  UR ',end='|')
                        elif np.argmax(Q[self.cart2s((i, j, k)),:])==7:
                            print('  UL ',end='|')
                        elif np.argmax(Q[self.cart2s((i, j, k)),:])==8:
                            print('  S  ',end='|')
                        elif np.argmax(Q[self.cart2s((i, j, k)),:])==9:
                            print('  B  ',end='|')

                    if j==(self.cols- 1):
                        print()
            print()

    def debug(self, q_table, origin = (0,0,0,0,0,0,0), reset = False):
        actions = {0:'Go',
                    1:'Left',
                    2:'Right'}
        if reset == True:

            self.g1, self.g2, self.g3, self.g4, self.row, self.col, self.psi = origin
            self.done = False
        else:
            best_action = np.argmax(q_table[self.cart2s((self.g1, self.g2, self.g3, self.g4, self.row, self.col, self.psi))])
            #print(f'action taken: {actions[best_action]}')
            state, reward, done = self.step(best_action)
            self.g1, self.g2, self.g3, self.g4, self.row, self.col, self.psi = state

    def get_distance(self,Path):
        dist=[]
        for i in range(0,len(Path)):
            try:
                cart1=np.array(self.s2cart(Path[i]))
                #cart1[2]/=2
                cart2=np.array(self.s2cart(Path[i+1]))
                #cart2[2]/=2
                dist.append(np.linalg.norm(cart1-cart2))
            except:
                pass
        return sum(dist)

    def get_meanDist(self,Path):
        #get mean of minimum distance
        dist=[]
        for cell in Path:
            obstacleList = []
            i,j,k=self.s2cart(cell)
            k/=2
            for obstacle in self.obstacles:
                ox, oy = self.s2c(obstacle)
                #oz=oz/2
                distance = np.sqrt((i - ox)**2 + (j - oy)**2)
                obstacleList.append(distance)
            obstacleList.sort()
            dist.append(sum(obstacleList[0:10]))
        return np.mean(dist)


    def onMap(self):
        k = self.psi
        for i in range(self.rows):
            print(72 * '-')
            for j in range(self.cols):
                if self.cart2s((i, j, k)) in self.obstacles:
                    print('  X  ', end = '|')
                elif (i, j, k) == (self.row, self.col, self.psi):
                    print('  A  ', end = '|')
                else:
                    print('     ', end='|')
            print('')

    def getGrafo(self):
        row = self.rows
        col = self.cols
        height = 0
        grafo = {}
        for k in range(height):
            for i in range(row):
                for j in range(col):
                    # if k > 0 and k < height-1:
                        if i > 0 and i < row-1 and j>0 and j < col-1:
                            grafo[(i, j, k)] = {(i+1,j,k) : 1,
                                                (i-1,j,k): 1,
                                                (i,j+1,k): 1,
                                                (i,j-1,k): 1,
                                                (i+1,j+1,k): np.sqrt(2),
                                                (i+1,j-1,k): np.sqrt(2),
                                                (i-1,j-1,k): np.sqrt(2),
                                                (i-1,j+1,k): np.sqrt(2),
                                                (i,j,k+1): .5,
                                                (i,j,k-1): .5
                                                }
                        elif (i == 0) and (j > 0 and j < col-1):
                            grafo[(i, j, k)] = {(i+1,j,k): 1,
                                                (i, j-1, k): 1,
                                                (i, j+1, k): 1,
                                                (i+1, j+1, k): np.sqrt(2),
                                                (i+1, j-1, k): np.sqrt(2),
                                                (i, j, k+1): .5,
                                                (i, j, k-1): .5}
                        elif (i == row-1) and (j > 0 and j < col-1):
                            grafo[(i, j, k)] = {(i-1,j,k): 1,
                                                (i, j-1, k): 1,
                                                (i, j+1, k): 1,
                                                (i-1, j+1, k): np.sqrt(2),
                                                (i-1, j-1, k): np.sqrt(2),
                                                (i, j, k+1): .5,
                                                (i, j, k-1): .5}
                        elif (j == 0) and (i>0 and i < row-1):
                            grafo[(i, j, k)] = {(i+1,j,k):1,
                                                (i-1,j,k): 1,
                                                (i,j+1,k): 1,
                                                (i+1,j+1,k): np.sqrt(2),
                                                (i-1,j+1,k): np.sqrt(2),
                                                (i,j,k+1): .5,
                                                (i,j,k-1): .5}
                        elif (j == col-1) and (i>0 and i < row-1):
                            grafo[(i, j, k)] = {(i+1,j,k): 1,
                                                (i-1,j,k): 1,
                                                (i,j-1,k): 1,
                                                (i+1,j-1,k): np.sqrt(2),
                                                (i-1,j-1,k): np.sqrt(2),
                                                (i,j,k+1): .5,
                                                (i,j,k-1): .5}
                        elif (i == 0 and j == 0):
                            grafo[(i, j, k)] = {(i+1,j,k): 1,
                                                (i+1,j+1,k): np.sqrt(2),
                                                (i,j+1,k): 1,
                                                (i,j,k+1): .5,
                                                (i,j,k-1): .5}
                        elif (i == 0 and j == col-1):
                            grafo[(i, j, k)] = {(i+1,j,k): 1,
                                                (i+1,j-1,k): np.sqrt(2),
                                                (i,j-1,k): 1,
                                                (i,j,k+1): .5,
                                                (i,j,k-1): .5}
                        elif (i == row -1 and j == 0):
                            grafo[(i, j, k)] = {(i-1,j,k): 1,
                                                (i-1,j+1,k): np.sqrt(2),
                                                (i,j+1,k): 1,
                                                (i,j,k+1): .5,
                                                (i,j,k-1): .5}
                        elif (i == row-1 and j == col -1):
                            grafo[(i, j, k)] = {(i-1,j,k): 1,
                                                (i-1,j-1,k): np.sqrt(2),
                                                (i,j-1,k): 1,
                                                (i,j,k+1): .5,
                                                (i,j,k-1): .5}
                        if k == 0:
                            del grafo[(i, j, k)][(i,j,k-1)]
                        if k == height-1:
                            del grafo[(i, j, k)][(i,j,k+1)]
        new_grafo = {}
        for key in grafo.keys():
            auxState = []
            auxDistance = []
            for item in grafo[key].items():
                if self.cart2s(item[0]) not in self.obstacles:
                    auxState.append(self.cart2s(item[0]))
                    auxDistance.append(item[1])
            new_grafo[self.cart2s(key)] = dict(zip(auxState, auxDistance))
        return new_grafo

    def getBestPath(self, initial, Q_table):
        '''
        this function return the best path for the robot to follow using Q-learning
        '''
        bestPath = []
        choosen_actions = []

        # get the best path
        self.debug(Q_table, initial, reset=True)
        self.done = False
        while not self.done:
            state = self.cart2s((self.g1, self.g2, self.g3, self.g4, self.row, self.col, self.psi))
            best_action = np.argmax(Q_table[state])

            newState, reward, done = self.step(best_action)
            print(self.g1, self.g2, self.g3, self.g4, "POSITION", self.row, self.col,'ACTION' , best_action, "REWARD", reward)
            self.g1, self.g2, self.g3, self.g4, self.row, self.col, self.psi = newState
            bestPath.append(self.cart2s(newState))
        return bestPath, best_action

    def testConvergence(self, Q_table):
        '''
        this function test if the robot has converged to a stable state
        '''

        Fail = 0
        maxSteps = self.rows * self.cols * self.height
        countStep = 0
        countStates = 0

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.height):
                    if self.cart2s((i,j,k)) not in self.obstacles and self.is_onboard((i, j, k)):
                        countStates += 1
                        countStep = 0
                        self.debug(Q_table, (i, j, k), reset=True)
                        while (not self.done) and (maxSteps > countStep):
                            state = self.cart2s((self.row, self.col, self.psi))
                            best_action = np.argmax(Q_table[state])
                            newState, reward, done = self.step(best_action)
                            self.row, self.col, self.psi = newState
                            countStep += 1
                            self.done = done
                        if self.done == False:
                            Fail += 1
                            #print("Fail: ", i,j,k)
        return np.round(100 * Fail / countStates, 2)

    def PrintBestPath(self, Q, k, path):
            print(self.cols*'---', end='')
            for i in range(0, self.rows):
                print('')
                for j in range(0, self.cols):
                    if self.cart2s((i,j,k)) in self.obstacles:
                        print(' X ',end='')
                    else:
                        if self.cart2s((i,j,k)) in path:
                            if np.argmax(Q[self.cart2s((i,j,k)),:])==0:
                                print(' v ',end='')
                            elif np.argmax(Q[self.cart2s((i,j,k)),:])==1:
                                print(' ^ ',end='')
                            elif np.argmax(Q[self.cart2s((i,j,k)),:])==2:
                                print(' > ',end='')
                            elif np.argmax(Q[self.cart2s((i, j, k)),:])==3:
                                print(' < ',end='')
                            elif np.argmax(Q[self.cart2s((i, j, k)),:])==4:
                                print(' \ ',end='')
                            elif np.argmax(Q[self.cart2s((i, j, k)),:])==5:
                                print(' / ',end='')
                            elif np.argmax(Q[self.cart2s((i, j, k)),:])==6:
                                print(' / ',end='')
                            elif np.argmax(Q[self.cart2s((i, j, k)),:])==7:
                                print(' \ ',end='')
                        else:
                            print('   ', end='')

                    if j==(self.cols- 1):
                        print("|")
            print(self.cols*'---')

    def printMiniGrid(self, initialPosition, endPosition, globalPath = None, localPath = None, qtableLocal = None):
        '''
        @param: initialPosition e.g. (0,0,0)
        @param: endPosition e.g. (9,9,0)
        @param: globalPath e.g. [0, 9, 18, 28, 38, 47, 57, 58, 68, 78, 79, 80]
        @param: localPath e.g. [0, 9, 17, 18]
        @param: qtableLocal
        @return None
        '''
        grid = np.zeros((self.rows+2, self.cols+2))
        verticalEsquerda = [(row, initialPosition[1]) for row in range(initialPosition[0], endPosition[0] + 1)]
        verticalDireita = [(row, endPosition[1]) for row in range(initialPosition[0], endPosition[0] + 1)]
        horizontalSuperior = [(initialPosition[0], col) for col in range(initialPosition[1], endPosition[1] + 1)]
        horizontalInferior = [(endPosition[0], col) for col in range(initialPosition[1], endPosition[1] + 1)]

        horizontal = set()
        horizontal.update(horizontalInferior + horizontalSuperior)
        vertical = set()
        vertical.update(verticalDireita + verticalEsquerda)

        Q = self.QTable

        verifyGlobal = True if globalPath != None else False
        verifyLocal = True if localPath != None else False
        for row in range(self.rows):
            print('')
            for col in range(self.cols):
                cell = self.cart2s((row, col, 0))
                if (verifyLocal) and (cell in localPath):
                        if np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==0:
                            print(' v"',end='')
                        elif np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==1:
                            print(' ^"',end='')
                        elif np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==2:
                            print(' >"',end='')
                        elif np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==3:
                            print(' <"',end='')
                        elif np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==4:
                            print(' \\"',end='')
                        elif np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==5:
                            print(' /"',end='')
                        elif np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==6:
                            print(' /"',end='')
                        elif np.argmax(qtableLocal[self.cart2s((row, col, 0)),:])==7:
                            print(' \\"',end='')

                elif (verifyGlobal) and (cell in globalPath):
                        if np.argmax(Q[self.cart2s((row, col, 0)),:])==0:
                            print(' v ',end='')
                        elif np.argmax(Q[self.cart2s((row, col, 0)),:])==1:
                            print(' ^ ',end='')
                        elif np.argmax(Q[self.cart2s((row, col, 0)),:])==2:
                            print(' > ',end='')
                        elif np.argmax(Q[self.cart2s((row, col, 0)),:])==3:
                            print(' < ',end='')
                        elif np.argmax(Q[self.cart2s((row, col, 0)),:])==4:
                            print(' \ ',end='')
                        elif np.argmax(Q[self.cart2s((row, col, 0)),:])==5:
                            print(' / ',end='')
                        elif np.argmax(Q[self.cart2s((row, col, 0)),:])==6:
                            print(' / ',end='')
                        elif np.argmax(Q[self.cart2s((row, col, 0)),:])==7:
                            print(' \ ',end='')
                elif cell in self.obstacles:
                    print(' O ', end='')

                elif (row, col) in horizontal:
                    print(' = ', end='')
                elif (row, col) in vertical:
                    print(' | ', end='')
                else:
                    print('   ', end='')
            if col == (self.cols - 1):
                print("|", end='')
        print('\n',self.cols*'---')




if __name__ == "__main__":
    env = GridWorld(9,9, (0,0,0), -1, -1, 10)
    env.set_obstacles([9, 6])
    env.set_goals([10, 32, 46, 80])
    env.step(0)
    print('oi')