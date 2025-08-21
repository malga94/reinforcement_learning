
import numpy as np
class GridEnv():

    def __init__(self,grid_dim):

        self.grid = np.zeros((grid_dim,grid_dim))
        self.grid = self.grid_creation(grid_dim)
        self.start = (0, 0)
        self.goal = (grid_dim-1, grid_dim-1)
        self.position = self.start
        self.possible_actions = ["up", "down", "left", "right"]
        self.n_states = grid_dim * grid_dim
        self.grid_dim = grid_dim

    def grid_creation(self,grid_dim):
        for i in range(grid_dim):
            for j in range(grid_dim):
                if (i == 0 and j == 0) or (i == grid_dim-1 and j == grid_dim-1):
                    # Keep the corners as zero
                    continue
                self.grid[i, j] = 0
        return self.grid
    
    def step(self,position, direction):
        """Given a state and action (int), return next_state, reward, done."""

        x, y = position
        if position == self.goal or position == self.start:
            return position, 0, True  # terminal
        
        if direction == "up" and x > 0:
            x -= 1
        elif direction == "down" and x < self.grid_dim - 1:
            x += 1
        elif direction == "left" and y > 0:
            y -= 1
        elif direction == "right" and y < self.grid_dim - 1:
            y += 1
        # else: no move, stay in place if on border

        new_position = (x, y)
        reward = 0 if new_position == self.goal or new_position== self.start else -1
        done = new_position == self.goal
        return new_position ,  reward , done
    
        
    def reset(self):
        self.position = self.start
        return self.position
    
    def get_position (self):
        return self.position

    def get_goal(self):
        return self.goal

    def get_grid(self):
        return self.grid
    