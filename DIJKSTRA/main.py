#! /usr/bin/env python3
from openinstance import OpenInstance# adsdsad sa
from gridworld import GridWorld
from qlAgentCursed import qlAgent as qla
from image_process import image_process

from sys import platform
opr=platform

import time
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import numpy as np

from math import sqrt, acos
from numpy import sign

import matplotlib.pyplot as plt


## load map
index = 1
if opr!='linux':
            print('We are working on a Windows system')
            path = f"mapasICUAS\mapaEscolhido{index}.txt"

else:
    print('We are working on a Linux system')
    path = f"mapasICUAS/mapaEscolhido{index}.txt"

maps = OpenInstance(path)
header, numObstacle, obstacles = maps.run()


row = header[0]
col = header[1]
try:
    height = header[2]
except:
    height = 1



## Create GridWorld
startPos = (0, 0, 0)# Posição inicial do agente
kd, ks, kt, kg = -.4, -.1, -.4, 100
grid_world = GridWorld(row, col, height, startPos,
                                kd, ks, kt, kg)

grid_world.set_obstacles(obstacles)
grid_world.get_reward_safety()
grid_world.goal = (5, 1, 0)

## Create agent
alpha = .1
numEpisode = 2500

localAgent=qla(alpha=alpha, epsilon=.3)
localAgent.setEnviroment(grid_world)
localAgent.setQtable(row * col, 8)
localAgent.setEpsilon(1, [.8, .1, numEpisode])
localAgent.setAlpha(0, [alpha, .1, numEpisode])

#localAgent.exploringStarts(length_i, length_j, mini_grid.initGrid)
localAgent.exploringStarts(1, 1, startPos)


# get position
distances = np.zeros(row*col)
timers = np.zeros(row*col)
turns = np.zeros(row*col)
for idx, state in enumerate(np.arange(row * col)):
    startPos = np.array(grid_world.s2cart(state))
    goals_vec = np.array([[5, 1, 0], [4, 5, 0], [1, 7, 0], [7, 9, 0]])
    goals_state = np.array([grid_world.cart2s(goal) for goal in goals_vec])
    visitados = np.array([0, 0, 0, 0])

    path = [grid_world.cart2s(startPos)]


    init = time.time()
    while not np.all(visitados == 1):
        menor = np.inf
        for i, goal in enumerate(goals_vec):
            if (np.linalg.norm(startPos - goal) < menor) and visitados[i] == 0:
                menor = np.linalg.norm(startPos - goal)
                index = i
        visitados[index] = 1

        crr_goal = goals_vec[index]
        grid_world.goal = crr_goal

        dijkstra_path = localAgent.getDijkstraPath(startPos)

        for item in dijkstra_path:
            bools = item == goals_state
            goals_state[bools] = 1



        startPos = crr_goal
        path = np.concatenate([np.array(path), np.array(dijkstra_path[1:])])

    distances[idx] = (len(path) - 1) * .5
    timers[idx] = (time.time() - init) * 1000
    for dijkstra_idx in range(2, len(path)):
        crr = np.array(grid_world.s2cart(path[dijkstra_idx])[:2])
        prev = np.array(grid_world.s2cart(path[dijkstra_idx-1])[:2])
        prev_v = np.array(grid_world.s2cart(path[dijkstra_idx-2])[:2])

        crr_dif = crr - prev
        prev_dif = prev - prev_v

        if np.all( (crr_dif + prev_dif) == (0, 0)):
            turns[idx] += 2

        if not np.any(crr_dif == prev_dif):
            turns[idx] += 1


plt.boxplot(distances)
plt.show()