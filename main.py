from gridworld import GridWorld
from agent import Agent
from openinstance import OpenInstance
from agents_stats import agents_stats
import numpy as np
import matplotlib.pyplot as plt

def main():
    import time

    # Load map
    map = OpenInstance('maps/map2.txt')
    header, _, obstacles = map.read()
    row, col = header


    # Define initial state
    start_state = (0, 0, 0)

    # Define living and turning penalty plus goal reward
    r_l = -1
    r_t = -2
    r_g = 10

    # Setting four goals state
    #goals = [10, 32, 46, 80]
    goals = [18, 49, 56, 86]

    for ii in range(1, 21):
        print('Training number: ', ii)
        # Loading environment
        env = GridWorld(row, col, start_state, r_l, r_t, r_g)
        env.set_obstacles(obstacles)
        env.set_goals(goals)


        # Loading an Agent
        agent = Agent()

        # Setting hyperparameters
        n_episodes = 80_000
        agent.setEnvironment(env)
        agent.setQtable(row*col*2**6, 3)
        agent.setEpsilon(1, [1, .1, n_episodes])
        agent.setAlpha()

        # Applying Q-table inicialization
        #agent.Q[:,0] = 1

        #agent.Q = np.loadtxt('qtable.txt')
        init = time.time()
        agent.train(n_episodes, 1, 0, ii)
        fim = time.time() - init

        # Loading agents stats
        metrics = agents_stats(agent, env)
        #path, act, length, turn, time = metrics.get_path((0,0,0,0,0,0,0))
        print(metrics.get_success_rate())
        dist, turns, planning_time = metrics.get_stats()
        print(np.mean(dist), np.mean(turns), np.mean(planning_time * 1000))

        np.save(f'results/distances/dist{ii}.npy', dist)
        np.save(f'results/turns/turns{ii}.npy', turns)
        np.save(f'results/planning_time/planning_time{ii}.npy', planning_time)
        print('Training time: ', fim)
        np.save(f'results/training_time/training_time{ii}.npy', np.array([fim]))
        np.save(f'results/success_rate/sucess_rate{ii}.npy', np.array([metrics.get_success_rate()]))
        #plt.boxplot(dist)
        #plt.show()

if __name__ == '__main__':
    main()