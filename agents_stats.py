from agent import Agent
from gridworld import GridWorld
import numpy as np
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
                (g1, g2, g3, g4, row, col, psi) = newState
                self.env.debug(self.qtable, reset = False)
                max_steps += 1

            if max_steps == 100:
                print("fail", (g1, g2, g3, g4, row, col, psi), self.env.cart2s((g1, g2, g3, g4, row, col, psi)))
                success_fail += 1

        return 100 * (len(all_states) - success_fail) / len(all_states)


    def __str__(self):
        return "name: %s, value: %s, type: %s" % (self.name, self.value, self.type)

    def __repr__(self):
        return self.__str__()