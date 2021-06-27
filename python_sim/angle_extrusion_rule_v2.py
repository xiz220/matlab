import numpy as np
from math import pi
from fillet_rule import calculate_centroid

class AngleExtrusionRule_v2:

    def __init__(self, line_length=10, turn_delay_max=5, slowdown_alpha=0.5, angle=45):
        self.n_agents = None
        self.deposition = None
        self.actions_list = None
        self.prev_x = [None, None]
        self.t = 0
        self.line_length = line_length
        self.jump_direction = 1
        self.random_jump = 0
        self.angle = angle
        self.radian = np.deg2rad(self.angle)
        self.blinders = None
        self.turn_delay = None
        self.turn_delay_actions = None
        self.turn_delay_max = turn_delay_max
        self.slowdown_alpha = slowdown_alpha
    
    def check_jump_direction(self, obs_grid):
        '''
        Parameters
        ----------
        obs_grid : TYPE - np array 
            DESCRIPTION. - sensor_readings of dimensions [row_dim, col_dim]

        Returns 
        -------
        numpy (1,2) array:
        unnormalized vector from the robot position to the centroid of the observation
        '''

        centroid = calculate_centroid(obs_grid)
        direction_to_centroid = centroid - np.floor((obs_grid.shape[0]-1)/2)*np.ones((2,))

        return -direction_to_centroid
        
        
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
            self.prev_jump = np.zeros((self.n_agents,2))
            self.blinders = np.zeros((self.n_agents,1))
            self.turn_delay = np.zeros((self.n_agents,1))
            self.turn_delay_actions = np.zeros((self.n_agents,2))
        deposition_action = []
        x = obs['x']
        threshold = 0.8
        
        
        for i in range(self.n_agents):
            obs = sensor_reading[i, :, :]
            jump_dir = self.check_jump_direction(obs)

            if self.turn_delay[i] == 0: #if we are not in turn delay state

                # if observation is all zeros or all ones
                if (jump_dir == np.array([0,0])).all():

                    #move in random direction, take blinders off, do not deposit material
                    self.actions_list[i] = (np.random.rand(2)-0.5)*100
                    self.prev_jump[i] = self.actions_list[i]*0.2
                    self.blinders[i] = 0
                    deposition_action.append(0)
                else: #else, we have some observation centroid

                    #normalize
                    jump_dir = jump_dir/np.linalg.norm(jump_dir)
                    dir_diff = (1/10)*self.prev_jump[i] - jump_dir
                    # if the difference between old mvmt dir and new movement dir is above threshold and the blinders
                    # are off
                    if np.linalg.norm(dir_diff) > threshold and self.blinders[i] == 0:
                        self.turn_delay[i] = self.turn_delay_max
                        self.turn_delay_actions[i,:] = self.prev_jump[i]
                        self.blinders[i] = 10 + self.turn_delay_max
                        # find the nearest angle to snap to
                        cand_vectors = np.array([[np.cos(self.radian), np.sin(self.radian)],
                                                [np.cos(self.radian), -np.sin(self.radian)],
                                                [-np.cos(self.radian), np.sin(self.radian)],
                                                [-np.cos(self.radian), -np.sin(self.radian)]])

                        dot_products = dir_diff @ cand_vectors.T
                        snapped_jump_dir = cand_vectors[np.argmin(dot_products),:]

                        # move in the direction opposite the new disturbance
                        self.actions_list[i] = 10*snapped_jump_dir*self.slowdown_alpha
                        deposition_action.append(1)
                        self.prev_jump[i] = 10*snapped_jump_dir

                        #import pdb; pdb.set_trace()

                    else: #else, no significant new disturbance (or blinders on)
                        if self.blinders[i] != 0: #decrement blinders if they are on
                            self.blinders[i] = self.blinders[i]-1
                        self.actions_list[i] = self.prev_jump[i]*self.slowdown_alpha
                        deposition_action.append(1)

            else:
                self.turn_delay[i] = self.turn_delay[i] - 1
                self.actions_list[i] = self.turn_delay_actions[i,:]
                deposition_action.append(1)
                #import pdb; pdb.set_trace()

        # action = np.array(self.actions_list)
        # action = action.reshape(self.n_agents, 2)
        
        #return np.concatenate((action, np.array(deposition_action).reshape(self.n_agents, 1)),
        #                      axis=1)
        return np.concatenate((np.array(self.actions_list), np.array(deposition_action).reshape(self.n_agents, 1)),
                              axis=1)
            
            
            

