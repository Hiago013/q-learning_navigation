import numpy as np
import matplotlib.pyplot as plt


x = np.arange(50)

m = -1/len(x)

y = m * x + 1

#plt.plot(x, y)
#plt.grid()
#plt.show()

def next_orientation( current, action):
    if action == 1:
        current += 90
    elif action == 2:
        current -= 90
    else:
        raise("Ação Incorreta!\nAções permitidas: [1, 2].")

    if current > 180:
        current -= 360
    elif current == -180:
        current = 180

    return current


print(next_orientation(0, 2))
