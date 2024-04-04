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

def create_shadow(x_position, y_position, velocity_x, velocity_y):
    """
    Creates a shadow of obstacles based on the given parameters.

    :param x_position: The x-coordinate of the current position.
    :param y_position: The y-coordinate of the current position.
    :param velocity_x: The velocity in the x-direction.
    :param velocity_y: The velocity in the y-direction.
    :return: List of obstacle coordinates representing the shadow.
    """
    cell_size = 0.5

    obstacle_list = [(x_position, y_position)]

    is_highest_velocity = 1 if abs(velocity_x) > abs(velocity_y) else 0

    highest_velocity = max(abs(velocity_x), abs(velocity_y))
    lowest_velocity = min(abs(velocity_x), abs(velocity_y))
    velocity_module = sqrt(highest_velocity**2 + lowest_velocity**2)

    if highest_velocity == 0:
        return obstacle_list

    angle_theta = acos((highest_velocity)**2 / (highest_velocity * velocity_module)) * 180 / 3.1415

    if 40 < angle_theta < 50:
        obstacle_list.append((x_position + sign(velocity_x) * cell_size, y_position))
        obstacle_list.append((x_position, y_position + sign(velocity_y) * cell_size))
        obstacle_list.append((x_position + sign(velocity_x) * cell_size, y_position + sign(velocity_y) * cell_size))
        return obstacle_list

    if is_highest_velocity == 1:
        obstacle_list.append((x_position + sign(velocity_x) * cell_size, y_position))
    else:
        obstacle_list.append((x_position, y_position + sign(velocity_y) * cell_size))

    return obstacle_list

def cell2position_simulator(cells):
    positions = []
    for cell in cells:
        i, j, k = cell2opt(8, 5, .5, cell)
        positions.append([i, j, k, 0, 3.1415/2, 0])
    np.savetxt('Mapa3.txt', np.array(positions))

def opt2position(x_max :float, y_max:float, size:float, position:tuple)->tuple:
    n_cols = int(x_max / size)
    n_rows = int(y_max / size)

    current_10x = position[0] * 10
    current_10y = position[1] * 10
    size_10 = size * 10

    xcell, ycell = int(current_10x/size_10), int(current_10y/size_10)
    cell = (n_cols) * int(current_10y/size_10) + int(current_10x/size_10)

    return xcell, ycell, cell

def cell2opt(x_max:float, y_max:float, cell_size:float, cell:int):
    n_cols = int(x_max / cell_size)

    i,j= divmod(cell, n_cols)
    y_cell, x_cell=i,j
    return (cell_size * x_cell + cell_size/2, cell_size * y_cell + cell_size/2, 1)

class MiniGrid(GridWorld):
    def __init__(self, rows, cols, height, start, Kd, Ks, Kt, Kg):
         super().__init__(rows, cols, height, start, Kd, Ks, Kt, Kg)
         self.initGrid=(0,0,0)
         self.endGrid=(rows-1, cols-1, height-1)
    def setPathGlobal(self, pathGlobal):
        self.pathGlobal = pathGlobal

    def setGridSize(self, localSize):
        self.localSize = localSize

    def setGoal(self, goal):
        self.goal = goal

    def setMiniGrid(self, initGrid, endGrid):
        self.initGrid = initGrid
        self.endGrid = endGrid

    def setObstacles(self, obstacles):
        self.obstacles = obstacles

    def setPosition(self, cartesian_position):
        self.i, self.j, self.k = cartesian_position

    def updateObstacles(self, obstacle:list):
        if obstacle not in (self.obstacles):
            self.obstacles.append(obstacle)

    def is_onboard(self, cartesian_position):
        '''
        checks if the agent is in the environment and return true or false
        init grid and endgrid (i,j)
        '''
        i, j, k = cartesian_position
        if i < self.initGrid[0]  or i > self.endGrid[0] or j < self.initGrid[1]  or j > self.endGrid[1]:
            return 0
        return 1

    def actionSpace(self):
        '''
        This function set the possible action space for each state removing actions that
        would leave out of the board or towards a obstacle
        '''
        self.actions = {}

        for state in range(0, self.cart2s((self.rows-1, self.cols-1, self.height-1))+1):
            self.actions[state] = list((0,1,2,3,4,5,6,7))
            action=0
            while action<=7:
                ni,nj,nk = self.test_move(action,state)
                if not self.is_onboard((ni,nj,nk)) or not self.is_onboard(self.s2cart(state)):
                    self.removeAction(action,state)
                action+=1


    def set_bounds(self, startPosition, row, col, grid_size):
        i, j, k = startPosition

        new_i, new_j, new_k = i - int(grid_size/2), j - int(grid_size/2), 0
        if grid_size % 2 == 0: #If grid size is a even number
            end_i, end_j, end_k = i + int(grid_size/2) - 1, j + int(grid_size/2) - 1, 0
        else:
            end_i, end_j, end_k = i + int(grid_size/2) - 0 , j + int(grid_size/2)- 0 , 0

        if new_i < 0:
            new_i = 0
            end_i = i + (grid_size - 1)

        if new_j < 0:
            new_j = 0
            end_j = j + (grid_size - 1)

        if end_i > row -1:
            new_i = row - grid_size
            end_i = row - 1

        if end_j > col -1:
            new_j = col - grid_size
            end_j = col - 1

        self.setMiniGrid((new_i, new_j, new_k), (end_i, end_j, end_k))

        return [(new_i, new_j, new_k), (end_i, end_j, end_k)]

    def get_obstacles_matrix(self):
        ii, ji, _ = self.initGrid
        ie, je, _ = self.endGrid
        obstacles_matrix = np.ones((self.rows, self.cols), dtype=np.uint8)
        for obstacle in self.obstacles:
            i, j, _ = self.s2cart(obstacle)
            obstacles_matrix[i, j] = 0
        return obstacles_matrix[ii:ie+1, ji:je+1]

    def get_minigrid_objects(self) -> np.ndarray:
        ii, ji, _ = self.initGrid
        ie, je, _ = self.endGrid
        ia, ja, _ = self.start
        ig, jg, _ = self.goal
        obstacle_matrix = np.ones((self.rows, self.cols), dtype=np.uint8)
        obstacle_matrix[ia, ja] = 2
        obstacle_matrix[ig, jg] = 3
        for obstacle in self.obstacles:
            i, j, _ = self.s2cart(obstacle)
            obstacle_matrix[i, j] = 0
        return obstacle_matrix[ii:ie+1, ji:je+1]

    def get_empty_cell_minigrid(self, init:tuple, end:tuple) -> list:
        ii, ji, _ = init
        ie, je, _ = end
        nonobstacles = []
        for i in range(ii, ie + 1):
            for j in range(ji, je + 1):
                if self.cart2s((i, j, 0)) not in self.get_obstacles():
                    nonobstacles.append(self.cart2s((i, j, 0)))
        return nonobstacles


    def set_mindist_goal(self, start_grid:tuple, end_grid:tuple, goal_position:int, minigrid_size:int):
        ii, ji, ki = start_grid
        ie, je, ke = end_grid
        ig, jg, kg = goal_position
        print('minigrid', goal_position)
        min_dist = 10000
        for i in range(ii, ie + 1):
            for j in range(ji, je + 1):
                if abs(i - ig) + abs(j - jg) < min_dist:
                    if self.cart2s((i, j, 0)) not in self.get_obstacles():
                        min_dist = abs(i - ig) + abs(j - jg)
                        goal = (i, j, 0)
        self.setGoal(goal=goal)
        print('GOAL: ', goal)
        return goal

    def get_bounds(self, start_position:tuple, row:int, col:int, action:int, grid_size:int) -> list:
        i, j, k = start_position

        if action == 0: # If action equals zero, so the agente must go down
            start_i, start_j, start_k = i , j- int(grid_size/2) , 0
            end_i, end_j, end_k = i + (grid_size-1), j + int(grid_size/2), 0


        elif action == 1: # If action equals one, so the agente must go up
            start_i, start_j, start_k = i - (grid_size-1), j- int(grid_size/2) , 0
            end_i, end_j, end_k = i , j + int(grid_size/2), 0



        elif action == 2: # If action equals two, so the agent must go right
            start_i, start_j, start_k = i-int(grid_size/2), j , 0
            end_i, end_j, end_k = i +  int(grid_size/2),  j+(grid_size-1), 0


        elif action == 3: # If action equals three, so the agent must go left
            start_i, start_j, start_k = i-int(grid_size/2), j - (grid_size-1), 0
            end_i, end_j, end_k = i+int(grid_size/2), j , 0


        elif action == 4: # If action equals four, so the agent must go down-right
            start_i, start_j, start_k = i , j , 0
            end_i, end_j, end_k = i + grid_size - 1, j + grid_size - 1, 0


        elif action == 5: # If action equals five, so the agent must go down-left
            start_i, start_j, start_k = i, j-(grid_size-1), 0
            end_i, end_j, end_k = i+(grid_size-1), j, 0


        elif action == 6: # If action equals six, so the agent must go up-right
            start_i, start_j, start_k = i-(grid_size-1) , j , 0
            end_i, end_j, end_k = i ,j+(grid_size-1) , 0


        elif action == 7: # If action equals seven, so the agent must go up-left
            start_i, start_j, start_k = i - (grid_size-1), j - (grid_size-1), 0
            end_i, end_j, end_k = i, j, 0


        else: # Otherwise the agent must be in the center of grid
            start_i, start_j, start_k = i - int(grid_size/2), j - int(grid_size/2), 0
            end_i, end_j, end_k = i + int(grid_size/2), j + int(grid_size/2), 0

        return [(max(0, start_i), max(0, start_j), 0), (min(row - 1, end_i), min(col - 1, end_j), 0)]

    def get_last_action(self, previous_state:tuple, current_state:tuple) -> int:
        i_p, j_p, k_p = previous_state
        i_c, j_c, k_c = current_state

        if (i_c == i_p + 1) and (j_c == j_p):
            return 0

        elif (i_c == i_p - 1) and (j_c == j_p):
            return 1

        elif (i_c == i_p) and (j_c == j_p + 1):
            return 2

        elif (i_c == i_p) and (j_c == j_p - 1):
            return 3

        elif (i_c == i_p + 1) and (j_c == j_p + 1):
            return 4

        elif (i_c == i_p + 1) and (j_c == j_p - 1):
            return 5

        elif (i_c == i_p - 1) and (j_c == j_p + 1):
            return 6

        elif (i_c == i_p - 1) and (j_c == j_p - 1):
            return 7


class local_planner:
    def __init__(self, index:int, kd:float, ks:float, kt:float, kg:float, startPos=(0,0,0), grid_size=7, numEpisode=1000,
                 alpha = .2):


        self.logger = {'position': [],
                       'local_grid_size': [],
                       'local_goal': []}



         #Creating our node,publisher and subscriber
        rospy.init_node('topic_publisher', anonymous=True)
        self.global_path = rospy.Publisher('/GlobalPath', Float64MultiArray, queue_size=1)
        self.orientation = rospy.Publisher('/Orientation', Float64MultiArray, queue_size=1)
        self.pub_log_position = rospy.Publisher('/log_position', Float64MultiArray, queue_size=1)
        self.pub_log_local_grid_size = rospy.Publisher('/log_local_grid_size', Float64MultiArray, queue_size=1)
        self.pub_log_local_goal = rospy.Publisher('/log_local_goal', Float64MultiArray, queue_size=1)
        self.pub_dijkstra = rospy.Publisher('/DijkstraPath', Float64MultiArray, queue_size=1)
        self.opt_info = rospy.Subscriber('/B1/ObstaclePosition', Float64MultiArray, self.callback)
        self.uav_position = rospy.Subscriber('/B1/UAVposition', Point, self.callbackpos)
        self.dynamic_pos = rospy.Subscriber('DynamicObstacle', Float64MultiArray, self.callback_dynamic)
        self.opt_pos = rospy.Subscriber('/OptPos', Point, self.callbackpos)
        self.flag = 0
        self.flag_minigrid = 0

        self.rate = rospy.Rate(10)
        self.multi_array = Float64MultiArray()
        self.__current_position = Point()
        self.best_path = Float64MultiArray()
        self.best_path_index = Float64MultiArray()
        self._current_action = Float64MultiArray()
        self.log_agent_position = Float64MultiArray()
        self.log_local_grid_size = Float64MultiArray()
        self.log_local_goal = Float64MultiArray()
        self.dijkstra_path = Float64MultiArray()
        self.dynamic_obstacles = Float64MultiArray()


        self._visited_states = list()
        self._last_action = None
        self.__index = index
        self.__ks = ks
        self.__kd = kd
        self.__kt = kt
        self.__kg = kg
        self.__startPos = list(startPos)
        self.__grid_size = grid_size
        self.__numEpisode = numEpisode
        self.__alpha = alpha
        self.__current_state = 0
        self._image_proc = None
        self.__previous_empty_cells = []

        self.action2angle = {0:-90,
                            1:90,
                            2: 0,
                            3:180,
                            4:-45,
                            5:-135,
                            6:45,
                            7:135}

        self.angle2action = {-90:0,
                             90:1,
                             0:2,
                             180:3,
                             -45:4,
                             -135:5,
                             45:6,
                             135:7}


        self.__load_our_map()
        self.__create_our_grid_world()
        self.__create_our_mini_grid()
        self.__create_our_agent()
        self.__train_agent()

    def callback_dynamic(self, data:Float64MultiArray):
        self.dynamic_obstacles = data


    def callbackpos(self, msg:Point):
        self.__current_position = msg


    def callback(self, data:Float64MultiArray):
        if self.flag == 0:
            self.__load_our_map()
            self.__create_our_grid_world()
            self.__create_our_mini_grid()
            self.__create_our_agent()
            self.__train_agent()


        ### Esta etapa é para previnir colisões durante o experimento, adicionando mais obstaculos
        ### Na direção de movimento do robô
        if len(self.dynamic_obstacles.data) > 0:
            dynamic_aux = []
            for i in range(0, len(self.dynamic_obstacles.data), 4):
                x = self.dynamic_obstacles.data[i]
                y = self.dynamic_obstacles.data[i+1]
                vx = self.dynamic_obstacles.data[i+2]
                vy = self.dynamic_obstacles.data[i+3]

                #print(x, y, vx, vy)

                print(create_shadow(x, y, vx, vy))

        x_max = 8
        y_max = 5
        cell_size = .5
        self.multi_array = data
       # self.multi_array.data = tuple(list(self.multi_array.data) + list(self.dynamic_obstacles.data))
        state_list = list()

        for i in range(0, len(self.multi_array.data), 3):
            _, _, cell = opt2position(x_max, y_max, cell_size, self.multi_array.data[i:i+3])
            state_list.append(cell)
        self.multi_array.data = state_list


        xcell, ycell, current_position = opt2position(x_max, y_max, cell_size, (self.__current_position.x, self.__current_position.y))
        i_c, j_c = ycell, xcell
        self.__current_state = current_position
        self.__startPos[0] = i_c
        self.__startPos[1] = j_c

        if current_position not in self._visited_states:
            self._visited_states.append(current_position)


        if len(self._visited_states) == 2:
            self._last_action = self.__localAgent.chooseBestAction(self.__current_state)
            ####self._last_action = self.__mini_grid.get_last_action(self.__mini_grid.s2cart(self._visited_states[0]), self.__mini_grid.s2cart(self._visited_states[1]))
            print('\n', self._last_action, '\n')
            del self._visited_states[0]
        else:
            self._last_action = self.__localAgent.chooseBestAction(self.__current_state)

        # Atualizar obstaculos no minigrid e no grid_world
        self.__mini_grid.set_obstacles(state_list)
        self.__grid_world.set_obstacles(state_list)

        # Atualizar a posicao do agente
        self.__mini_grid.start = self.__startPos
        self.__grid_world.start = self.__startPos



        try:
            if current_position != self.logger['position'][-1]:
                self.logger['position'] += [current_position]

                # Pegar as celulas vazias da primeira vez que ele criou o minigrid

                init, end = self.__mini_grid.get_bounds(self.__startPos, self.__row, self.__col, self._last_action, self.__grid_size)
                self.__mini_grid.setMiniGrid(init, end)
                self.__previous_empty_cells = self.__mini_grid.get_empty_cell_minigrid(init, end)
                I_bw = self.__mini_grid.get_obstacles_matrix()

                print(I_bw, '\n')

        except IndexError:
            self.logger['position'] += [current_position]



        #init, end = self.__mini_grid.get_bounds(self.__startPos, self.__row, self.__col, self._last_action, self.__grid_size)
        #previous_empty_cells = self.__mini_grid.get_empty_cell_minigrid(init, end)


        obs_in_path = set(self.best_path.data) & set(self.multi_array.data)           # Interseção entre dois conjuntos
        obs_in_minigrid = set(self.__previous_empty_cells) & set(self.multi_array.data)

        # Utilizado porque o mapa que iniciamos é vazio, pois os obstaculos vem do MATLAB
        if self.flag_minigrid == 0:
            init, end = self.__mini_grid.get_bounds(self.__startPos, self.__row, self.__col, self._last_action, self.__grid_size)
            self.__mini_grid.setMiniGrid(init, end)
            self.__previous_empty_cells = self.__mini_grid.get_empty_cell_minigrid(init, end)
            #I_bw = self.__mini_grid.get_obstacles_matrix()
            #print(I_bw, '\n')

            obs_in_minigrid = {25}
            self.flag_minigrid = 1

        #obs_in_minigrid = set(previous_empty_cells) & set(self.multi_array.data)      # Interseção entre dois conjunto




        #print(self.best_path.data)
        #print(obs_in_path)
        print(f"obs: {1}, drone:{current_position} {i_c, j_c}, obsminigrid: {obs_in_minigrid}", end='\r')



        try:
            if current_position in self.best_path.data[-2:] and (self.__mini_grid.cart2s(self.get_goal_position()) not in self.best_path.data):
                init, end = self.__mini_grid.get_bounds(self.__startPos, self.__row, self.__col, self._last_action, self.__grid_size)
                self.__mini_grid.setMiniGrid(init, end)
                self.__previous_empty_cells = self.__mini_grid.get_empty_cell_minigrid(init, end)
                #I_bw = self.__mini_grid.get_obstacles_matrix()
                #print(I_bw, '\n')
                self.__obstacles = sorted(self.multi_array.data)
                self.__create_our_grid_world()
                self.__create_our_mini_grid()
                self.__create_our_agent()
                self.__train_agent()
        except ValueError:
            pass

        if (len(obs_in_minigrid) > 0):
            init, end = self.__mini_grid.get_bounds(self.__startPos, self.__row, self.__col, self._last_action, self.__grid_size)
            self.__mini_grid.setMiniGrid(init, end)
            self.__previous_empty_cells = self.__mini_grid.get_empty_cell_minigrid(init, end)
            I_bw = self.__mini_grid.get_obstacles_matrix()
            print('\n', I_bw, '\n')
            self.__obstacles = sorted(self.multi_array.data)
            self.__create_our_grid_world()
            self.__create_our_mini_grid()
            self.__create_our_agent()
            self.__train_agent()

        index_list = self.state_list2index_list(self.best_path.data)
        self.best_path_index.data = index_list
        self._current_action.data = [self.action2angle[self.__localAgent.chooseBestAction(item)] for item in self.best_path.data]


        self.orientation.publish(self._current_action)
        self.global_path.publish(self.best_path_index)

        # Log
        self.log_agent_position.data = self.logger['position']
        self.log_local_goal.data = self.logger['local_goal']
        self.log_local_grid_size.data = self.logger['local_grid_size']

        self.pub_log_position.publish(self.log_agent_position)
        self.pub_log_local_goal.publish(self.log_local_goal)
        self.pub_log_local_grid_size.publish(self.log_local_grid_size)
        self.pub_dijkstra.publish(self.dijkstra_path)



    def set_goal_position(self, goal_position:tuple) -> None:
        self.__goal_position = goal_position

    def get_goal_position(self)-> tuple:
        return self.__goal_position

    def set_minigrid_size(self, grid_size:int)->None:
        self.__grid_size = grid_size

    def __load_our_map(self):
        if opr!='linux':
            print('We are working on a Windows system')
            path = f"mapasICUAS\mapaEscolhido{self.__index}.txt"

        else:
            print('We are working on a Linux system')
            path = f"mapasICUAS/mapaEscolhido{self.__index}.txt"

        self.__maps = OpenInstance(path)
        self.__header, self.__numObstacle, self.__obstacles = self.__maps.run()


        self.__row = self.__header[0]
        self.__col = self.__header[1]
        try:
            self.__height = self.__header[2]
        except:
            self.__height = 1

    def __create_our_grid_world(self):
        try:
            self.get_goal_position()
        except AttributeError:
              self.set_goal_position((self.__row - 1, self.__col - 1, 0))# Objetivo global do agente

        self.__grid_world = GridWorld(self.__row, self.__col, self.__height, self.__startPos,
                                self.__kd, self.__ks, self.__kt, self.__kg)

        self.__grid_world.set_obstacles(self.__obstacles)
        self.__grid_world.get_reward_safety()

    def __create_our_mini_grid(self):
        # Configurar Minigrid
        self.__mini_grid = MiniGrid(self.__row, self.__col, self.__height, self.__startPos, self.__kd, self.__ks, self.__kt, self.__kg)
        self.__mini_grid.setObstacles(self.__grid_world.get_obstacles())

        # Modificacao novo minigrid
        try:
            if self._last_action != None:
                action = self.__localAgent.chooseBestAction(self.__current_state)
                ####action = self._last_action
            else:
                action = self.__localAgent.chooseBestAction(self.__current_state)
            init_grid, end_grid = self.__mini_grid.get_bounds(self.__startPos, self.__row, self.__col, action, self.__grid_size)
            print(f'\ncurrent action: {action}')
            print(f'\ninit: {init_grid}\nend: {end_grid}\n')
            self.__mini_grid.setMiniGrid(init_grid, end_grid)
        except AttributeError:
            action = 0
            init_grid, end_grid = self.__mini_grid.set_bounds(self.__startPos, self.__row, self.__col, self.__grid_size)

        self.__mini_grid.setPosition(self.__startPos)
        self.__mini_grid.set_mindist_goal(init_grid, end_grid, self.__goal_position, self.__grid_size)
        self.search_path(action)
        self.__mini_grid.actionSpace()

        if self._last_action != None:
            self.__mini_grid.last_action = self._last_action
            self.__mini_grid.get_energyCost()

        self.__mini_grid.get_reward_safety()

        cell2position_simulator(self.__mini_grid.get_obstacles())

        # Getting log
        ii, jj, __ = self.__mini_grid.initGrid
        ie, je, __ = self.__mini_grid.endGrid
        ig, jg, __ = self.__mini_grid.goal
        self.logger['local_grid_size'] += [ii, jj, ie, je]
        self.logger['local_goal'] += [ig, jg]



    def check_connected(self) -> bool:
        I_bw = self.__mini_grid.get_obstacles_matrix()
        #print(I_bw)
        mini_grid_objects = self.__mini_grid.get_minigrid_objects()

        if not np.any(mini_grid_objects==2): # The goal is the same place of the agent
            return False

        agent_result = np.where(mini_grid_objects==2)
        agent_coordinates= list(zip(agent_result[0], agent_result[1]))[0]

        goal_result = np.where(mini_grid_objects==3)
        goal_coordinates= list(zip(goal_result[0], goal_result[1]))[0]

        self._image_proc = image_process(I_bw)
        self._image_proc.two_pass()
        check_connected = self._image_proc.check_elements_connecteds(goal_coordinates, agent_coordinates)
        return check_connected



    def search_path(self, action:int):
        '''
        This function search for available paths
        '''
        next_action = action
        isconnected = self.check_connected()
        count = 0
        while not isconnected:
            angle = self.action2angle[next_action]
            next_action = self.angle2action[angle + 45] if angle < 180 else self.angle2action[-135]

            init_grid, end_grid = self.__mini_grid.get_bounds(self.__startPos, self.__row, self.__col, next_action, self.__grid_size)
            print(f'\ncurrent action: {next_action}')
            print(f'\ninit: {init_grid}\nend: {end_grid}\n')
            self.__mini_grid.setMiniGrid(init_grid, end_grid)
            self.__mini_grid.setPosition(self.__startPos)
            self.__mini_grid.set_mindist_goal(init_grid, end_grid, self.__goal_position, self.__grid_size)

            isconnected = self.check_connected()
            count += 1
            if count == 8:
                raise Exception("Don't there any path available")

        print(f'The path is connected')




        #print(self.__mini_grid.get_bounds((0,0,0), 10, 10, 4, 5))

    def __create_our_agent(self):
        self.__localAgent=qla(alpha=self.__alpha, epsilon=.3)
        self.__localAgent.setEnviroment(self.__mini_grid)
        self.__localAgent.setQtable(self.__row * self.__col, 8)
        self.__localAgent.setEpsilon(1, [.8, .1, self.__numEpisode])
        self.__localAgent.setAlpha(0, [self.__alpha, .1, self.__numEpisode])

        length_i = self.__mini_grid.endGrid[0] - self.__mini_grid.initGrid[0] + 1
        length_j = self.__mini_grid.endGrid[1] - self.__mini_grid.initGrid[1] + 1

        #self.__localAgent.exploringStarts(length_i, length_j, self.__mini_grid.initGrid)
        self.__localAgent.exploringStarts(1, 1, self.__startPos)

    def __train_agent(self):
        if self.get_goal_position() != self.__startPos:
            init=time.time() * 1000
            self.__localAgent.train(self.__numEpisode, 1, plot=0)
            fim=time.time() * 1000 - init
            print(fim)


            self.__grid_world.PrintBestPath(self.__localAgent.getQtable(), 0, self.__localAgent.getBestPath(self.__startPos))
            self.best_path.data = self.__localAgent.getBestPath(self.__startPos)
            dijkstra_path = self.__localAgent.getDijkstraPath(self.__startPos)
            print(f'Dijkstra Path: {self.__localAgent.getDijkstraPath(self.__startPos)}')
            print(f'startPos:{self.__startPos}, Best path:{self.best_path.data}')

            index_list = self.state_list2index_list(self.best_path.data)
            self.best_path_index.data = index_list
            self._current_action.data = [self.action2angle[self.__localAgent.chooseBestAction(item)] for item in self.best_path.data]
            self.dijkstra_path.data = self.state_list2index_list(dijkstra_path)
            self.orientation.publish(self._current_action)
            self.global_path.publish(self.best_path_index)
            self.pub_dijkstra.publish(self.dijkstra_path)
            self.flag = 1
            aa = []
            # for obstacle in self.__mini_grid.obstacles:
            #    a, b, c = cell2opt(8, 5, .5, obstacle)
            #    aa.append(a)
            #    aa.append(b)
            #    aa.append(c)
            # print(aa)


    def _check_diagonal_obstacles(self, obstacles_list:list)-> list:
        new_list_obstacles = obstacles_list.copy()
        for obstacle in obstacles_list:
            if (obstacle + 1 - self.__col in obstacles_list) and (obstacle % self.__col < self.__col):
                new_list_obstacles.append(obstacle + 1)
                new_list_obstacles.append(obstacle + self.__col)
            if (obstacle - 1 - self.__col in obstacles_list) and (obstacle % self.__col > 0):
                new_list_obstacles.append(obstacle - 1)
                new_list_obstacles.append(obstacle - self.__col)

        return list(set(new_list_obstacles))

    def state_list2index_list(self, state_list:list) ->list:
        index_list = list()
        for state in state_list:
            i, j, _ = self.__mini_grid.s2cart(state)
            index_list.append(j)
            index_list.append(i)
        return index_list

    def send_messages(self):
        while self.get_goal_position() != self.__startPos:
            self.orientation.publish(self._current_action)
            self.global_path.publish(self.best_path_index)




if __name__ == '__main__':
    try:
        a1 = local_planner(1, -.4, -.1, -.4, 100, alpha=.2, grid_size=5, numEpisode=2500)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

