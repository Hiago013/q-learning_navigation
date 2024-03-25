from environment import GridWorld
from brain import qlAgent as qla

row = 4
col = 4
start = (0,0,0)
ks = -1
kt = 0
kg = 10

env = GridWorld(row,col,start,ks,kt,kg)
env.goal = (3,3,0)
env.obstacles = []

agent = qla(.1)

agent.t