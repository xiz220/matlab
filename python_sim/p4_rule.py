'''
moving avg window - 30
variance window - 15
'''

import numpy as np
from math import pi
import pdb
from fillet_rule import calculate_centroid

class P4Rule:

    def __init__(self, line_length=10, turn_delay_max=5, slowdown_alpha=0.5, angle=45, moving_avg_window_size=30, moving_var_window_size=20, var_queue_size=5, var_threshold=10):
        self.n_agents = None
        self.deposition_action = None
        self.actions_list = None
        self.prev_x = None
        self.t = 0
        self.line_length = line_length
        self.jump_direction = 1
        self.random_jump = 0

        self.robot_state = None

        #ANGLE BEAM EXTRUSION VARIABLES
        self.angle = angle
        self.radian = np.deg2rad(self.angle)
        self.blinders = None
        self.turn_delay = None
        self.turn_delay_actions = None
        self.turn_delay_max = turn_delay_max
        self.slowdown_alpha = slowdown_alpha
        self.prev_jump = None
        self.turn_counter = None

        #WALL FOLLOW / CORNER DETECTION VARIABLES
        self.z_vector = np.array([0,0,1])
        self.moving_avg = None
        self.moving_window = None
        self.variance_window = None
        self.moving_variance = None
        self.raw_angle_store = None
        self.variance_counter = None
        self.wall_follow_flag = None
        self.corner_flag = None
        self.corner_counter = None
        self.moving_var_window = None
        self.deposition_counter = None
        self.deposition_flag = None #what is this??
        self.avg_window_size = moving_avg_window_size #for eg: if defined as 10, then it will take 10 raw angle values and average it
        self.var_window_size = moving_var_window_size #function similar to moving_avg_window_size, but will take in the averaged values and then calculate variance for the particular window size
        self.var_queue_size = var_queue_size #window size used to see the consistency in varience and detect corner based on irregularity
        self.var_threshold = var_threshold #threshold on which the var_queue_size would tested
        


    
    def get_action(self, obs):
        """
        Takes in an observation and returns the next action: move randomly, deposit if adjacent to existing material.
        :param obs: observation dict {'x': robot_states (n_agents x 2 np array), 'sensor_readings': (n_agents x row_dim x col_dim) np array}
        :return: n_agents x 3 np array, where row i represents the i'th robot's: [x_action, y_action, deposition_action]
        """

        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition_action = np.zeros((self.n_agents,))
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_x = np.zeros((self.n_agents,7,2)) ##
            self.turn_delay = np.zeros((self.n_agents,1))
            self.turn_delay_actions = np.zeros((self.n_agents,2))
            self.moving_window = np.zeros((self.n_agents,self.avg_window_size))   ##
            self.variance_window = np.zeros((self.n_agents,self.var_window_size)) ##
            self.wall_follow_flag = np.zeros((self.n_agents,))
            self.variance_counter = np.zeros((self.n_agents,))
            self.corner_flag = np.zeros((self.n_agents,))
            self.corner_counter = np.zeros((self.n_agents,))
            self.moving_var_window = np.zeros((self.n_agents,self.var_queue_size))
            self.deposition_flag = np.zeros((self.n_agents,))
            self.robot_state = np.zeros((self.n_agents,))
            self.prev_jump = np.zeros((self.n_agents, 2))
            self.blinders = np.zeros((self.n_agents,1))
            self.turn_delay = np.zeros((self.n_agents,1))
            self.turn_delay_actions = np.zeros((self.n_agents,2))
            self.turn_counter = np.zeros((self.n_agents,1))

        self.x = obs['x']

        for i in range(self.n_agents):
            obs = sensor_reading[i, :, :]
            self.increment_history_vars(i)

            if self.robot_state[i] == 0:
                self.wallfollow_update(obs, i)

            if self.robot_state[i] == 1:
                self.beam_update(obs, i)

            
                
        return np.concatenate((np.array(self.actions_list), np.array(self.deposition_action).reshape(self.n_agents, 1)),
                              axis=1)

    def increment_history_vars(self, i):
        # increment position history
        self.prev_x[i][:-1] = self.prev_x[i][1:]
        self.prev_x[i][-1][0] = self.x[i, 0]
        self.prev_x[i][-1][1] = self.x[i, 1]

        degree_angle = np.rad2deg(np.arctan2(self.actions_list[i][1], self.actions_list[i][0]))

        # adding the degree_value in self.moving_window np array using queue logic, FIFO
        self.moving_window[i][:-1] = self.moving_window[i][1:]
        self.moving_window[i][-1] = degree_angle
        # take average of degree values present in the moving window to get moving average
        temp_avg = sum(self.moving_window[i]) / len(self.moving_window[i])

        # adding the temp_avg in self.variance_window np array using queue logic, FIFO
        self.variance_window[i][:-1] = self.variance_window[i][1:]
        self.variance_window[i][-1] = temp_avg
        # finding variance of the moving average values present in the variance window to get moving average
        temp_var = np.var(self.variance_window[i])

        # storing values of moving variance in a queue to check for consistency or irregularities in bot movement
        self.moving_var_window[i][:-1] = self.moving_var_window[i][1:]
        self.moving_var_window[i][-1] = temp_var

    def check_directional_and_total_density(self, obs_grid):
        t = obs_grid
        row = t.shape[0]
        col = t.shape[1]

        total_density = np.count_nonzero(t) / t.size

        # north density
        n_count = np.count_nonzero(t[0:int(np.ceil(row / 2)), :])
        n_size = float(t[0:int(np.ceil(row / 2)), :].size)
        n_density = n_count / n_size

        # south density
        s_count = np.count_nonzero(t[int(np.floor(row / 2)):row, :])
        s_size = float(t[int(np.floor(row / 2)):row, :].size)
        s_density = s_count / s_size

        # east density
        e_count = np.count_nonzero(t[:, int(np.floor(col / 2)):col])
        e_size = float(t[:, int(np.floor(col / 2)):col].size)
        e_density = e_count / e_size

        # west density
        w_count = np.count_nonzero(t[:, 0:int(np.ceil(col / 2))])
        w_size = float(t[:, 0:int(np.ceil(col / 2))].size)
        w_density = w_count / w_size

        direction_wise_density = np.array([n_density, s_density, e_density, w_density])
        direction_wise_count = np.array([n_count, s_count, e_count, w_count])
        direction_wise_size = np.array([n_size, s_size, e_size, w_size])

        return direction_wise_density, direction_wise_count, direction_wise_size, total_density

    def direction_to_centroid(self, obs_grid):
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
        direction_centroid = centroid - np.floor((obs_grid.shape[0] - 1) / 2) * np.ones((2,))

        return direction_centroid

    def wallfollow_update(self, obs, i):

        direction_centroid = self.direction_to_centroid(obs)
        direction_centroid = np.append(direction_centroid, 0)

        direction_parallel_wall = np.cross(direction_centroid, self.z_vector)

        direction_centroid_norm = direction_centroid / np.linalg.norm(direction_centroid)

        direct_parallel_wall = direction_parallel_wall / np.linalg.norm(direction_parallel_wall)

        if (np.isnan(direct_parallel_wall).any()):
            direct_parallel_wall = direction_parallel_wall

        direction_wise_density, direction_wise_count, direction_wise_size, total_density = self.check_directional_and_total_density(
            obs)

        if ((total_density > 0) and (total_density < 0.45)):
            # Bot is near some material, action is to move closer to centroid and also move parallel to the wall
            temp_movement_vector = (1 - total_density) * direction_centroid_norm[0:2] + direct_parallel_wall[0:2]
            self.actions_list[i] = ((temp_movement_vector / np.linalg.norm(
                temp_movement_vector)) * self.slowdown_alpha) * 10
            self.deposition_action[i] = 0
            self.deposition_flag[i] = 0

        elif ((total_density >= 0.45) and (total_density <= 0.55)):
            # bot is just at the wall so action is to move parallel to it
            self.actions_list[i] = direct_parallel_wall[0:2]
            self.deposition_action[i] = 1
            self.actions_list[i] = (self.actions_list[i] * self.slowdown_alpha) * 10
            self.deposition_flag[i] = 1

        elif ((total_density > 0.55) and (total_density <= 0.85)):
            # bot is inside the wall, action is to move parallel to wall but also away from centroid
            temp_movement_vector = total_density * direct_parallel_wall[0:2] + \
                                   (-1 + total_density) * direction_centroid_norm[0:2]

            self.actions_list[i] = ((temp_movement_vector / np.linalg.norm(
                temp_movement_vector)) * self.slowdown_alpha) * 10
            self.deposition_action[i] = 1
            self.deposition_flag[i] = 1

        elif ((total_density > 0.85) and (total_density < 1)):
            # bot is almost completely inside the wall so move in the opposite direction of centroid
            self.actions_list[i] = ((total_density * (-direction_centroid_norm[0:2])) +
                                    ((1 - total_density) * direct_parallel_wall[0:2])) * 10
            self.deposition_action[i] = 0
            self.deposition_flag[i] = 0

        else:
            # if inside a wall completely or in an empty area then jump randomly
            self.actions_list[i] = ((np.random.uniform(low=-1, high=1, size=(2,))) * 15) * 10
            self.deposition_action[i] = 0
            self.deposition_flag[i] = 0

        if ((self.moving_var_window[i] <= self.var_threshold).all() and (self.wall_follow_flag[i] == 0) and
                self.deposition_flag[i] == 1):
            self.wall_follow_flag[i] = 1
        if self.blinders[i] == 0:
            if ((self.wall_follow_flag[i] == 1) and (self.moving_var_window[i] > self.var_threshold).all()):
                self.wall_follow_flag[i] = 0
                self.robot_state[i] = 1
                if hasattr(self, 'env'):
                    self.env.set_flag(self.prev_x[i][0, 0], self.prev_x[i][0, 1])
                    # pdb.set_trace()
        else:
            self.blinders[i] = self.blinders[i]-1

    def beam_update(self, obs, i):


        jump_dir = -self.direction_to_centroid(obs)
        threshold = 0.8


        if self.turn_delay[i] == 0:  # if we are not in turn delay state

            # if observation is all zeros or all ones
            if (jump_dir == np.array([0, 0])).all():

                # move in random direction, take blinders off, do not deposit material
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 100
                self.prev_jump[i] = self.actions_list[i] * 0.2
                self.blinders[i] = 0
                self.deposition_action[i] = 0
            else:  # else, we have some observation centroid

                # normalize
                jump_dir = jump_dir / np.linalg.norm(jump_dir)
                dir_diff = (1 / 10) * self.prev_jump[i] - jump_dir
                # if the difference between old mvmt dir and new movement dir is above threshold and the blinders
                # are off
                if np.linalg.norm(dir_diff) > threshold and self.blinders[i] == 0:
                    self.turn_delay[i] = self.turn_delay_max
                    self.turn_delay_actions[i, :] = self.prev_jump[i]
                    self.blinders[i] = 10 + self.turn_delay_max
                    # find the nearest angle to snap to
                    cand_vectors = np.array([[np.cos(self.radian), np.sin(self.radian)],
                                             [np.cos(self.radian), -np.sin(self.radian)],
                                             [-np.cos(self.radian), np.sin(self.radian)],
                                             [-np.cos(self.radian), -np.sin(self.radian)]])

                    dot_products = dir_diff @ cand_vectors.T
                    snapped_jump_dir = cand_vectors[np.argmin(dot_products), :]

                    # move in the direction opposite the new disturbance
                    self.actions_list[i] = 10 * snapped_jump_dir * self.slowdown_alpha
                    self.deposition_action[i] = 1
                    self.prev_jump[i] = 10 * snapped_jump_dir

                    # import pdb; pdb.set_trace()

                else:  # else, no significant new disturbance (or blinders on)
                    if self.blinders[i] != 0:  # decrement blinders if they are on
                        self.blinders[i] = self.blinders[i] - 1
                    self.actions_list[i] = self.prev_jump[i] * self.slowdown_alpha
                    self.deposition_action[i] = 1

        else:
            self.turn_delay[i] = self.turn_delay[i] - 1
            self.actions_list[i] = self.turn_delay_actions[i, :] * self.slowdown_alpha
            self.deposition_action[i] = 1
            if self.turn_delay[i] == 0:
                if self.turn_counter[i] == 0:
                    self.turn_counter[i] = 1
                else:
                    self.turn_counter[i] = 0
                    if hasattr(self, 'env'):
                        self.env.set_flag(self.x[i, 0], self.x[i, 1])
                    self.robot_state[i] = 0
                    self.blinders[i] = self.avg_window_size + 10

            # import pdb; pdb.set_trace()


    def set_env(self, env):
        self.env = env
