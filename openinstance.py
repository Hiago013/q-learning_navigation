class OpenInstance:
    def __init__(self, instanceName):
        self.instanceName = instanceName


    def read(self):
        maps = open(self.instanceName, 'r')
        self.header = maps.readline().rstrip('\n').split(' ')
        self.header = tuple([int(num) for num in self.header])
        self.walls = []
        line = 'aux'
        num_obstacles = 0
        while line != '':
            line = maps.readline().rstrip('\n')
            try:
                self.walls.append(int(line))
                num_obstacles += 1
            except:
                pass
        maps.close()

        return(self.header, num_obstacles, list(self.walls))

