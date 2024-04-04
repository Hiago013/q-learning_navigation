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

def cell2opt(x_max:float, y_max:float, cell_size:float, cell:int):
    n_cols = int(x_max / cell_size)

    i,j= divmod(cell, n_cols)
    y_cell, x_cell=i,j
    return (cell_size * x_cell + cell_size/2, cell_size * y_cell + cell_size/2, 1)


print(create_shadow(0,0,-3,2))
