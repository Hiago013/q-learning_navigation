# CLASS GRID
import pygame, sys
import os

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
class create_grid():
    def __init__(self, row, col, width, height, margem = 1):
        """
        This Python function initializes an object with specified row, column, width, height, and margin
        attributes.
        """
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.margem = margem
        self.grid = []
        
    
    def save(self, path : str = ''):
        """
        The `save` function writes the coordinates of cells with a value of 2 in a grid to a file
        specified by the `path` parameter.
        """
        with open(path, 'w') as f:
                        for i in range(len(self.grid)):
                            for j in range(len(self.grid[0])):
                                if self.grid[i][j] == 2:
                                    f.write(f'{i} {j}\n')
        pygame.quit()
        
         
         
    def main(self):
        """
        The main function initializes a grid, sets up a Pygame window for path planning, handles mouse
        events to update the grid, and continuously updates the display.
        """
        for linha in range(self.row):
            self.grid.append([])
            for coluna in range(self.col):
                self.grid[linha].append(0) 
        
        pygame.init()
        janela = pygame.display.set_mode(((self.col*self.height) + self.col + 1, (self.row*self.width) + self.row+1))
        pygame.display.set_caption("Path Planning")  
        
        FPS = 30
        timer = pygame.time.Clock()
        done = True
        while done:
            for evento in pygame.event.get(): 
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        
                # click do mouse
                elif evento.type == pygame.MOUSEBUTTONDOWN:
                    # obtém a posição do clique do mouse
                    pos = pygame.mouse.get_pos()
                    # converte a posição do mouse para linha e coluna na self.grid
                    coluna = pos[0] // (self.width + self.margem)
                    linha = pos[1] // (self.height + self.margem)
                    if evento.button == 1:  # Botão esquerdo
            
                        if self.grid[linha][coluna] == 2:
                            self.grid[linha][coluna] = 0
                        else:  
                            self.grid[linha][coluna] = 2

                    print("Click ", pos, "Coordinates on grid: ", linha, coluna)
                            
                elif evento.type == 768:
                    done = False
                                    
                    
            janela.fill(BLACK)
            for linha in range(self.row):
                for coluna in range(self.col):
                    cor = WHITE
                    if self.grid[linha][coluna] == 2:
                        cor = BLACK

                    pygame.draw.rect(janela, cor, [(self.margem + self.width) * coluna + self.margem,
                    (self.margem + self.height) * linha + self.margem, self.width, self.height])
            
            timer.tick(FPS)
            pygame.display.flip()
