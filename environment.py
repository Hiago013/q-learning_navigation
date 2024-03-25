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
            start --> Initial position (x, y, psi)
        '''
        #Define reward
        self.Ks = Ks
        self.Kt = Kt
        self.Kg = Kg
        self.reward = 0

        # set grid world size
        self.rows = rows # number of rows, so if rows = 4, we'll have rows labeld from 0-3
        self.cols = cols # number of cols, so if cols = 4, we'll have columns labeld from 0-3

        # set initial position
        self.start = start
        self.row = start[0] #starting i
        self.col = start[1] #starting j
        self.psi = 0

        # set goal position
        self.goal = (self.rows-1, self.cols-1, 0)

        # set done verify
        self.done = False # Means the episode hasn't ended

        # set obtacles position
        self.obstacles = set()

        # set last action
        self.last_action = 0

        # list of actions taken
        self.current_action = 0

        # action space
        self.actions = {}

        self.states = np.arange(self.rows * self.cols * 4)
        self.states = self.states.reshape([self.rows, self.cols, 4])

    def get_obstacles(self):
        return self.obstacles

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def get_current_state(self):
        return self.row, self.col, self.psi

    def set_current_state(self, state:tuple):
        self.row, self.col, self.psi = state

    def vec2state(self, vec):
        return self.states[vec]

    def state2vec(self, state):
        vec = np.where(self.states == state)
        row, col, psi = vec[0][0], vec[1][0], vec[2][0]
        return (row, col, psi)


    def step(self, action):
        ''' return new state, reward and boolean value to verify if the new state = goal
            0: Go Forward
            1: Turn Right 90
            2: Turn Left -90
        '''
        reward = 0
        row, col, psi = self.get_current_state()
        if action == 0:
            if psi == 0:
                new_col = self.col + 1
            elif psi == 1:
                new_row = self.row - 1
            elif psi == 2:
                new_col = self.col - 1
            elif psi == 3:
                new_row = self.row + 1
            new_psi = psi
        elif action == 1 or action == 2:
            new_psi = self.next_orientation(self.STATE2ANGLE[psi], action)
        else:
            print("Ação inválida.")

        if not self.is_onboard((new_row, new_col, new_psi)):
            # Cancel action
            new_row = row
            new_col = col
            new_psi = psi

        reward = self.get_reward() # reward of distance and energy
        new_state = (new_row, new_col, new_psi)
        ###################################
        # IS DONE PRECISO EDITAR
        self.done = self.is_done(new_state)
        ###################################


        return (new_state), reward, self.done

    def is_done(self, state):
        row, col, psi = state
        row_g, col_g, psi_g = self.goal
        if (row, col) == (row_g, col_g):
            return 1

        if self.is_in_the_obstacle(state):
            return 1

        return 0

    def get_reward(self, action, new_state):
        '''
        this function return a reward for each step
        '''
        row, col, _ = new_state
        row_g, col_g, _ = self.goal
        reward = self.Ks

        if action == 0:
            reward += 0
        else:
            reward += self.Kt

        if (row, col) == (row_g, col_g):
            reward += self.Kg
            self.done = True

        if self.is_in_the_obstacle(new_state):
            reward -= self.Kg
            self.done = True
        return reward



    def is_onboard(self, cartesian_position):
        '''
        checks if the agent is in the environment and return true or false
        '''
        i, j, _ = cartesian_position
        if i < 0 or i >= self.rows or j < 0 or j >= self.cols:
            return 0
        return 1

    def is_in_the_obstacle(self, cartesian_position):
        '''
        checks if the agent is in the environment and return true or false
        '''
        i, j, _ = cartesian_position
        state = (i, j)
        if state in self.obstacles:
            return 1
        return 0


    def reset(self, start=0):
        if start==0:
            self.row = self.start[0]
            self.col = self.start[1]
            self.psi = self.start[2]
        else:
            self.row = start[0]
            self.col = start[1]
            self.psi = start[2]
        self.done = False

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

if __name__ == '__main__':
    env = GridWorld(4, 4, (0,0,0), -1, -1, 10)
    env.states