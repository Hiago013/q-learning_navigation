import numpy as np
import csv

def load_q_table(file_name):
    q_table = np.loadtxt(fname=file_name+'.txt')
    return q_table

def create_conversor(row, col):
    states = np.arange(row * col * 4 *16)
    states = states.reshape([2, 2, 2, 2, row, col, 4])
    return states

def list2string(arr:list):
    str_list = ''
    for item in arr:
        str_list += str_list.join([str(item) + ','])
    str_list = str_list[:-1]
    return str_list

def s2cart(state:int, cnversor):
    vec = np.where(state == cnversor)
    g1, g2, g3, g4, row, col, psi = vec[0][0], vec[1][0], vec[2][0], vec[3][0], vec[4][0], vec[5][0], vec[6][0]
    return [g1, g2, g3, g4, row, col, psi]

def main():
    row, col = 9, 9
    q_table = load_q_table('qtable')
    conversor = create_conversor(row, col)
    header = ['g1', 'g2', 'g3', 'g4', 'row', 'col', 'psi', 'action']
    data = []

    for s in np.arange(row * col * 4 * 16):
        aux = s2cart(s, conversor)
        aux.append(np.argmax(q_table[s,:]))
        data.append(aux)
        aux = []

    with open('qtable.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)






if __name__ == '__main__':
    q_table = load_q_table('qtable')
    conversor = create_conversor(9, 9)
    main()

    print(s2cart(0, conversor))
