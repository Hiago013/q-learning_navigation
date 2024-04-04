import numpy as np

class image_process:
    '''
    This class performs the image processing
    '''

    def __init__(self, img:np.ndarray):
        self.img = img
        self.labels = None
    
    # def _get_connected_elements(self, row : int, col : int) -> np.ndarray:
    #     '''
    #     This function returns the connected elements of the given element using 8-connectivity
    #     :param row:
    #     :param col:
    #     :return:
    #     '''

    #     value = self.img[row, col]
    #     neighbours = []

    #     for r in range(np.max([row - 1, 0]), row + 1):
    #         for c in range(np.max([col - 1, 0]), col + 1):
    #             if self.img[r, c] == value and not (r == row and c == col):
    #                 neighbours.append((r, c))
    #     return np.array(neighbours)

    def _get_connected_elements(self, row : int, col : int) -> np.ndarray:
        '''
        This function returns the connected elements of the given element using 8-connectivity
        :param row:
        :param col:
        :return:
        '''

        value = self.img[row, col]
        neighbours = []

        min_row = np.max([row-1, 0])
        min_col = np.max([col-1, 0])

        if self.img[row, min_col] == value and not(min_col == col):
            neighbours.append((row, min_col))
        
        if self.img[min_row, col] == value and not(min_row == row):
            neighbours.append((min_row, col))
            
        return np.array(neighbours)
    
    def two_pass(self) -> np.ndarray:
        '''
        This function performs the two pass algorithm to get the connected elements
        :return: Image labeled
        '''

        h, w = self.img.shape
        labels = np.zeros((h, w), dtype=np.uint8)
        new_label = 1

        # First pass
        for row in range(h):
            for col in range(w):
                if self.img[row, col] == 0:
                    continue
                else:
                    neighbours = self._get_connected_elements(row, col)
                    if neighbours.size == 0:
                        labels[row, col] = new_label
                        new_label += 1
                    else:
                        neighbours_labels = labels[neighbours[:, 0], neighbours[:, 1]]
                        min_label = np.min(neighbours_labels)
                        labels[row, col] = min_label
        
        # Second pass
        for row in range(h):
            for col in range(w):
                if labels[row, col] == 0:
                    continue
                else:
                    neighbours = self._get_connected_elements(row, col)
                if neighbours.size == 0:
                    continue
                else:
                    neighbours_labels = labels[neighbours[:, 0], neighbours[:, 1]]
                    min_label = np.min(neighbours_labels)
                    if np.all(neighbours_labels == min_label):
                        continue
                    else:
                        for label in neighbours_labels:
                            if label != min_label:
                                labels[labels == label] = min_label
        
        self.labels = labels

        return labels
    
    def get_labels(self) -> np.ndarray:
        '''
        This function returns the labels
        '''

        if self.labels is None:
            raise Exception("Labels not found")
        else:
            return self.labels
    
    def check_elements_connecteds(self, pixel_1 : tuple, pixel_2 : tuple) -> bool:
        '''
        This function checks if two elements are connected
        :param p1:
        :param p2:
        :return:
        '''
        if self.labels is None:
            raise Exception("Labels not found")
        if self.labels[pixel_1] == 0 or self.labels[pixel_2] == 0:
            raise Exception("One of the pixels is not labeled")

        if self.labels[pixel_1] == self.labels[pixel_2]:
            return True
        else:
            return False

if __name__ == '__main__':
    None
    #a = [[1, 1, 1, 0, 1],
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 1, 0, 1],
    #     [1, 0, 0, 0, 1],
    #     [1, 1, 1, 1, 1]]
    #img = np.array(a)
    #b = image_process(img)
    #print(b.two_pass())
    #print(b.check_elements_connecteds((0,2), (4,4)))
