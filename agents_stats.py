from agent import Agent
from gridworld import GridWorld
import numpy as np
from time import time
class agents_stats(object):
    def __init__(self, agent : Agent, env : GridWorld):
        self.agent = agent
        self.env = env
        self.qtable = agent.getQtable().copy()

    def get_success_rate(self):
        all_states = self.agent.states_.copy()
        visitados = np.zeros(self.env.rows * self.env.cols * 4 * 16)
        success_fail = 0
        for state in all_states:
            if visitados[state] == 1:
                continue
            visitados[state] = 1
            state_vec = self.env.s2cart(state)
            (g1, g2, g3, g4, row, col, psi) = state_vec
            self.env.debug(self.qtable, state_vec, reset = True)
            done = False
            max_steps = 0
            if g1 == g2 == g3 == g4 == 1:
                continue
            while not done and max_steps < 100:
                s = self.env.cart2s((g1, g2, g3, g4, row, col, psi))
                visitados[s] = 1
                best_action = np.argmax(self.qtable[s])
                newState, reward, done = self.env.step(best_action)

                self.env.debug(self.qtable, (g1, g2, g3, g4, row, col, psi) ,reset = True)
                self.env.debug(self.qtable, reset = False)

                (g1, g2, g3, g4, row, col, psi) = newState

                max_steps += 1

            if max_steps == 100:
                #print("fail", (g1, g2, g3, g4, row, col, psi), self.env.cart2s((g1, g2, g3, g4, row, col, psi)))
                success_fail += 1

        return 100 * (len(all_states) - success_fail) / len(all_states)

    def get_stats(self):
        # - Listar Estados que eu quero testar que não sejam nos obstaculos
        # Iterar sobre as linhas, colunas e orientações
        vector_states = []
        goals = [0, 0, 0, 0]
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if not self.env.c2s((r, c)) in self.env.obstacles:
                    vector_states.append(goals + [r, c, 0])
                    vector_states.append(goals + [r, c, 1])
                    vector_states.append(goals + [r, c, 2])
                    vector_states.append(goals + [r, c, 3])
        vector_states = np.array(vector_states)

        # Calcular distância para cada estado no vector_states
        distancias = []
        turns = []
        planning_time = []
        for init_state in vector_states:
            state = init_state.copy()
            self.env.debug(self.qtable, state, reset = True)
            done = False
            max_steps = 0
            length = 0
            turn = 0
            check_orientation = 0
            init_timer = time()
            while not done and max_steps < 100:
                s = self.env.cart2s(tuple(state))
                best_action = np.argmax(self.qtable[s])
                newState, reward, done = self.env.step(best_action)

                self.env.debug(self.qtable, state ,reset = True)
                self.env.debug(self.qtable)
                state = newState
                max_steps += 1
                if best_action == 0:
                    length += .5
                    check_orientation = 1
                else:
                    if check_orientation == 1:
                        turn += 1
            if not done:
                #print("fail", init_state)
                continue

            distancias.append(length)
            turns.append(turn)
            planning_time.append(time() - init_timer)
        return np.array(distancias), np.array(turns), np.array(planning_time)


    def get_path(self, origin=None):
        if not origin:
            origin = self.env.start

        state = origin
        self.env.debug(self.qtable, state, reset = True)
        done = False
        max_steps = 0
        length = 0
        turn = 0
        init_timer = time()
        actions = []
        path = [(origin[-3:-1])]
        check_orientation = 0
        while not done and max_steps < 100:
            s = self.env.cart2s(tuple(state))
            best_action = np.argmax(self.qtable[s])
            newState, reward, done = self.env.step(best_action)
            path.append(state[-3:-1])
            self.env.debug(self.qtable, state ,reset = True)
            self.env.debug(self.qtable)
            actions.append(best_action)
            max_steps += 1
            state = newState
            if best_action == 0:
                length += .5
                check_orientation = 1
            else:
                if check_orientation == 1:
                    turn += 1
        if not done:
            print("fail")


        return path, actions, length, turn, time() - init_timer

    def __str__(self):
        return "name: %s, value: %s, type: %s" % (self.name, self.value, self.type)

    def __repr__(self):
        return self.__str__()

