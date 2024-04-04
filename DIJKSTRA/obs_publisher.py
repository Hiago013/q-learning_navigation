#! /usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
from openinstance import OpenInstance
from sys import platform
opr=platform

def cell2opt(x_max:float, y_max:float, cell_size:float, cell:int):
    n_cols = int(x_max / cell_size)

    i,j= divmod(cell, n_cols)
    y_cell, x_cell=i,j
    return (cell_size * x_cell + cell_size/2, cell_size * y_cell + cell_size/2, 1)

def load_map(index):
        if opr!='linux':
            print('We are working on a Windows system')
            path = f"mapasJINT\map{index}.txt"

        else:
            print('We are working on a Linux system')
            path = f"mapasJINT/map{index}.txt"

        maps = OpenInstance(path)
        _, _, obstacles = maps.run()
        return obstacles

def obstacler(data):
    list_obstacles = Float64MultiArray()

    pub = rospy.Publisher('/B1/ObstaclePosition', Float64MultiArray, queue_size=1)
    rospy.init_node('node_sObstacle', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        obstacle = data
        list_obstacles.data = obstacle
        pub.publish(list_obstacles)
        rate.sleep()

if __name__ == '__main__':
    data = []
    obstacles = load_map(2)
    for obstacle in obstacles:
        i,j,k = cell2opt(8, 5, .5, obstacle)
        data += [i, j, k]
    print(data)
    try:
        obstacler(data)
    except rospy.ROSInterruptException:
        pass
