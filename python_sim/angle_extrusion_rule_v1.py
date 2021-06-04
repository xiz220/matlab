import numpy as np
from math import pi

class AngleExtrusionRule_v1:

    def __init__(self, line_length=10):
        self.n_agents = None
        self.deposition = None
        self.actions_list = None
        self.prev_x = [None, None]
        self.t = 0
        self.line_length = line_length
        self.jump_direction = 1
        self.random_jump = 0
        self.radian = pi/4
        self.action_x = 0
        self.action_y = 0
        self.angle = 45
    
    def check_jump_direction(self, obs_grid):
        '''
        Parameters
        ----------
        obs_grid : TYPE - np array 
            DESCRIPTION. - sensor_readings of dimensions [row_dim, col_dim]

        Returns 
        -------
        Integer:
        Quadrant number of where the wall/material was detected
        '''
        
        if ((obs_grid[2,3]==1 or obs_grid[2,4]==1 or obs_grid[3,4]==1) and 
                (obs_grid[1,3]==1 or obs_grid[1,4]==1 or obs_grid[1,5]==1 or obs_grid[2,5]==1 or obs_grid[3,5]==1)):
            self.jump_direction = 3 #diagnolly opposit to quadrant 1
        elif ((obs_grid[2,2]==1 or obs_grid[2,3]==1 or obs_grid[3,2]==1) and 
                (obs_grid[1,3]==1 or obs_grid[1,1]==1 or obs_grid[1,2]==1 or obs_grid[2,1]==1 or obs_grid[3,1]==1)):
            self.jump_direction = 4 #diagnollay opposite to quadrant 2
        elif ((obs_grid[3,2]==1 or obs_grid[4,2]==1 or obs_grid[4,3]==1) and 
                (obs_grid[3,1]==1 or obs_grid[4,1]==1 or obs_grid[5,1]==1 or obs_grid[5,2]==1 or obs_grid[5,3]==1)):
            self.jump_direction = 1 #diagnolly opposite to quadrant 3
        elif ((obs_grid[3,4]==1 or obs_grid[4,4]==1 or obs_grid[4,3]==1) and 
                (obs_grid[5,3]==1 or obs_grid[5,4]==1 or obs_grid[5,5]==1 or obs_grid[4,5]==1 or obs_grid[3,5]==1)):
            self.jump_direction = 2 #diagnolly opposite to quadrant 4
        else :
            self.jump_direction = 0
        
        # if ((obs_grid[2,4]==1 or obs_grid[3,4]==1) and 
        #         (obs_grid[1,4]==1 or obs_grid[1,5]==1 or obs_grid[2,5]==1 or obs_grid[3,5]==1)):
        #     self.jump_direction = 3 #diagnolly opposit to quadrant 1
        # elif ((obs_grid[2,2]==1 or obs_grid[2,3]==1) and 
        #         (obs_grid[1,3]==1 or obs_grid[1,1]==1 or obs_grid[1,2]==1 or obs_grid[2,1]==1)):
        #     self.jump_direction = 4 #diagnollay opposite to quadrant 2
        # elif ((obs_grid[3,2]==1 or obs_grid[4,2]==1) and 
        #         (obs_grid[3,1]==1 or obs_grid[4,1]==1 or obs_grid[5,1]==1 or obs_grid[5,2]==1)):
        #     self.jump_direction = 1 #diagnolly opposite to quadrant 3
        # elif ((obs_grid[4,4]==1 or obs_grid[4,3]==1) and 
        #         (obs_grid[5,3]==1 or obs_grid[5,4]==1 or obs_grid[5,5]==1 or obs_grid[4,5]==1)):
        #     self.jump_direction = 2 #diagnolly opposite to quadrant 4
        # else :
        #     self.jump_direction = 0
        
        
            
        # if ((obs_grid[2,3]==1 or obs_grid[2,4]==1 or obs_grid[3,4]==1) and
        #         (obs_grid[3,2]==1 or obs_grid[4,2]==1 or obs_grid[4,3]==1)):
        #     self.random_jump = 1 # to jump in quadrants other than 1 and 3
        # elif ((obs_grid[2,2]==1 or obs_grid[2,3]==1 or obs_grid[3,2]==1) and
        #         (obs_grid[3,4]==1 or obs_grid[4,4]==1 or obs_grid[4,3]==1)):
        #     self.random_jump = 2 # to jump in quadrants other than 2 and 4
            
        return self.jump_direction
        
        
    def get_action(self, obs):
        """
        Takes in an observation and returns the next action: move randomly, deposit if adjacent to existing material.
        :param obs: observation dict {'x': robot_states (n_agents x 2 np array), 'sensor_readings': (n_agents x row_dim x col_dim) np array}
        :return: n_agents x 3 np array, where row i represents the i'th robot's: [x_action, y_action, deposition_action]
        """
    
        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition = np.zeros((self.n_agents,))
            ##self.actions_list = []
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_jump = np.zeros((self.n_agents,1)) 
        deposition_action = []
        x = obs['x']
        self.radian = np.deg2rad(self.angle)
        
        
        
        for i in range(self.n_agents):
            obs = sensor_reading[i, :, :]
            jump_dir = self.check_jump_direction(obs)
            
            temp_x = np.random.rand()
            temp_y = (np.tan(self.radian))*(temp_x)
            
            # if (jump_dir != 0):
            #     if (jump_dir == 3):
            #         self.action_y = -temp_y
            #         self.action_x = -temp_x
            #     elif (jump_dir == 4):
            #         self.action_y = -temp_y
            #         self.action_x = temp_x
            #     elif (jump_dir == 1):
            #         self.action_y = temp_y
            #         self.action_x = temp_x
            #     elif (jump_dir == 2):
            #         self.action_y = temp_y
            #         self.action_x = -temp_x
            
            
            if (jump_dir != 0):
                if ((jump_dir == 3) and (self.prev_jump[i] == 3)):
                    self.action_y = -temp_y
                    self.action_x = -temp_x
                elif ((jump_dir == 4) and (self.prev_jump[i] == 4)):
                    self.action_y = -temp_y
                    self.action_x = temp_x
                elif ((jump_dir == 1) and (self.prev_jump[i] == 1)):
                    self.action_y = temp_y
                    self.action_x = temp_x
                elif ((jump_dir == 2) and (self.prev_jump[i] == 2)):
                    self.action_y = temp_y
                    self.action_x = -temp_x
                
                self.actions_list[i] = np.array([self.action_x, self.action_y])
                deposition_action.append(1)
                self.prev_jump[i] = jump_dir
                
            else:
                ##temp_action_list = (np.random.rand()-0.5)*4
                self.actions_list[i] = (np.random.rand(2)-0.5)*10
                deposition_action.append(0)
                self.prev_jump[i] = np.random.choice([1,2,3,4])         
        # action = np.array(self.actions_list)
        # action = action.reshape(self.n_agents, 2)
        
        #return np.concatenate((action, np.array(deposition_action).reshape(self.n_agents, 1)),
        #                      axis=1)
        return np.concatenate((np.array(self.actions_list), np.array(deposition_action).reshape(self.n_agents, 1)),
                              axis=1)
            
            
            

        