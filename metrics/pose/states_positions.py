from .metrics_interface import metrics_interface
from environment.src import transition_orientation
from targets import goal_position
from typing import Tuple, List
import numpy as np

class states_positions(metrics_interface):
    @staticmethod
    def run(qtable,
            target_state: goal_position,
            start_state: Tuple,
            trans_model: transition_orientation,
            ) -> List[Tuple[int, int]]:

        n = 500 # Preciso arrumar a verificacao do GOAL
        #################################################
        next_state = start_state
        states = [(start_state[0], start_state[1])]
        while not(target_state.isdone(next_state)) and n > 0:
            best_action = np.argmax(qtable[next_state])
            row, col, psi = next_state[0], next_state[1], next_state[2]
            nrow, ncol, npsi = trans_model.step((row, col, psi), best_action)
            target_state.isgoal((nrow, ncol))
            next_state = [nrow, ncol, npsi] + target_state.get_visited_state()
            next_state = tuple(next_state)
            if best_action == 0:
                states.append(tuple([next_state[0], next_state[1]]))
            n -= 1

        return states
