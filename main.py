from gridworld import GridWorld
from agent import Agent
from openinstance import OpenInstance
from agents_stats import agents_stats
import numpy as np

def main():
    # Load map
    map = OpenInstance('maps/map2.txt')
    header, _, obstacles = map.read()
    row, col = header


    # Define initial state
    start_state = (0, 0, 0)

    # Define living and turning penalty plus goal reward
    r_l = -1
    r_t = -1
    r_g = 10

    # Setting four goals state
    #goals = [10, 32, 46, 80]
    goals = [18, 49, 56, 86]

    # Loading environment
    env = GridWorld(row, col, start_state, r_l, r_t, r_g)
    env.set_obstacles(obstacles)
    env.set_goals(goals)


    # Loading an Agent
    agent = Agent()

    # Setting hyperparameters
    n_episodes = 50000
    agent.setEnvironment(env)
    agent.setQtable(row*col*2**6, 3)
    agent.setEpsilon(1, [1, .1, n_episodes])
    agent.setAlpha()

    # Applying Q-table inicialization
    #agent.Q[:,0] = 1

    #agent.Q = np.loadtxt('qtable.txt')

    agent.train(n_episodes, 1, 1)


    # Loading agents stats
    metrics = agents_stats(agent, env)
    print(metrics.get_success_rate())


if __name__ == '__main__':
    main()