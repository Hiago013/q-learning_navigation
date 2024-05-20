import numpy as np
from environment.src import transition_position
from typing import Tuple, Dict

# This class likely represents a data structure to store key-value pairs with the ability to access 
# elements by their position in the dictionary.
class graph_2d():
    def __init__(self,
                 nrow:int,
                 ncol:int,
                 model:transition_position,
                 actions:list
                 ) -> None:
        
        self.nrow = nrow
        self.ncol = ncol
        self.model = model
        self.actions = actions

    def __check_position(self, state:Tuple[int, int]) -> bool:
        """
        The function __check_position() takes a state tuple as input and returns True if the position is
        within the specified boundaries, otherwise False.
        """
        if (state[0] < 0) or (state[1] < 0): 
            return False
        if (state[0] >= self.nrow) or (state[1] >= self.ncol):
            return False
        return True
               
    def get_graph(self) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
        """
        The function `dict2D` creates a 2D dictionary representing a graph where each cell has a list of
        neighboring cells based on certain conditions.
        """
        graph = dict()
        for row in range (self.nrow):
            for col in range (self.ncol):
                graph[(row, col)] = dict()
                for action in self.actions:
                    next_state = self.model.step((row,col), action)
                    if self.__check_position(next_state):
                        graph[(row,col)][next_state] = 1     
        return graph
    
    
        
        
    