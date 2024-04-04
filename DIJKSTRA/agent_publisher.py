#!/usr/bin/env python
# license removed for brevity

import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import numpy as np
import time

def agent():
    global path
    path = list()
    current = Point()
    pub = rospy.Publisher('/B1/UAVposition', Point, queue_size=10)
    sub = rospy.Subscriber('/GlobalPath', Float64MultiArray, callback)
    rospy.init_node('node_cPos', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        new_path = []
        aux = []
        for i in range(len(path)):
            print(path[i])
            aux.append(path[i])
            if (i+1) % 2 == 0:
                new_path.append(tuple(aux))
                aux.clear()

        for i in range(len(new_path) - 1):
            try:
                x, y = new_path[i+1]
                cx, cy = new_path[i]

                x_pos = np.linspace(cx, x, 100)
                y_pos = np.linspace(cy, y, 100)


                for j in range(100):
                    current.x = 1#x_pos[j]
                    current.y = 1#y_pos[j]
                    current.z = 1
                    pub.publish(current)
                    rate.sleep()

            except IndexError:
                pass


def callback(data:Float64MultiArray):
    global path
    path = [value for value in data.data]
if __name__ == '__main__':
    try:
        agent()
    except rospy.ROSInterruptException:
        pass