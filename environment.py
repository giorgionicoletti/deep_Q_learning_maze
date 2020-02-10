import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy

class MazeEnvironment:    
    def __init__(self, maze, init_position, goal):
        x = len(maze)
        y = len(maze)
        
        self.boundary = np.asarray([x, y])
        self.init_position = init_position
        self.current_position = np.asarray(init_position)
        self.goal = goal
        self.maze = maze
        
        self.visited = set()
        self.visited.add(tuple(self.current_position))
                
        # initialize the empty cells and the euclidean distance from
        # the goal (removing the goal cell itself)
        self.allowed_states = np.asarray(np.where(self.maze == 0)).T.tolist()
        self.distances = np.sqrt(np.sum((np.array(self.allowed_states) -
                                         np.asarray(self.goal))**2,
                                         axis = 1))
        
        del(self.allowed_states[np.where(self.distances == 0)[0][0]])
        self.distances = np.delete(self.distances, np.where(self.distances == 0)[0][0])
                
        self.action_map = {0: [0, 1],
                           1: [0, -1],
                           2: [1, 0],
                           3: [-1, 0]}
        
        self.directions = {0: '→',
                           1: '←',
                           2: '↓ ',
                           3: '↑'}
        
        # the agent makes an action from the following:
        # 1 -> right, 2 -> left
        # 3 -> down, 4 -> up
        
    # introduce a reset policy, so that for high epsilon the initial
    # position is nearer to the goal (useful for large mazes)
    def reset_policy(self, eps, reg = 7):
        return sp.softmax(-self.distances/(reg*(1-eps**(2/reg)))**(reg/2)).squeeze()
    
    # reset the environment when the game is completed
    # with probability prand the reset is random, otherwise
    # the reset policy at the given epsilon is used
    def reset(self, epsilon, prand = 0):
        if np.random.rand() < prand:
            idx = np.random.choice(len(self.allowed_states))
        else:
            p = self.reset_policy(epsilon)
            idx = np.random.choice(len(self.allowed_states), p = p)

        self.current_position = np.asarray(self.allowed_states[idx])
        
        self.visited = set()
        self.visited.add(tuple(self.current_position))

        return self.state()
    
    
    def state_update(self, action):
        isgameon = True
        
        # each move costs -0.05
        reward = -0.05
        
        move = self.action_map[action]
        next_position = self.current_position + np.asarray(move)
        
        # if the goals has been reached, the reward is 1
        if (self.current_position == self.goal).all():
                reward = 1
                isgameon = False
                return [self.state(), reward, isgameon]
            
        # if the cell has been visited before, the reward is -0.2
        else:
            if tuple(self.current_position) in self.visited:
                reward = -0.2
        
        # if the moves goes out of the maze or to a wall, the
        # reward is -1
        if self.is_state_valid(next_position):
            self.current_position = next_position
        else:
            reward = -1
        
        self.visited.add(tuple(self.current_position))
        return [self.state(), reward, isgameon]

    # return the state to be feeded to the network
    def state(self):
        state = copy.deepcopy(self.maze)
        state[tuple(self.current_position)] = 2
        return state
        
    
    def check_boundaries(self, position):
        out = len([num for num in position if num < 0])
        out += len([num for num in (self.boundary - np.asarray(position)) if num <= 0])
        return out > 0
    
    
    def check_walls(self, position):
        return self.maze[tuple(position)] == 1
    
    
    def is_state_valid(self, next_position):
        if self.check_boundaries(next_position):
            return False
        elif self.check_walls(next_position):
            return False
        return True
    
    
    def draw(self, filename):
        plt.figure()
        im = plt.imshow(self.maze, interpolation='none', aspect='equal', cmap='Greys');
        ax = plt.gca();

        plt.xticks([], [])
        plt.yticks([], [])

        ax.plot(self.goal[1], self.goal[0],
                'bs', markersize = 4)
        ax.plot(self.current_position[1], self.current_position[0],
                'rs', markersize = 4)
        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
        plt.show()
