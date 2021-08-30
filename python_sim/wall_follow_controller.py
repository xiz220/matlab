import numpy as np
from fillet_rule import calculate_centroid
from math import pi


def direction_to_centroid(obs_grid):
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


def check_directional_and_total_density(obs_grid):
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


class WallFollowController:

    def __init__(self, moving_avg_window_size=30, moving_var_window_size=20, var_queue_size=5, var_threshold=10,
                 slowdown_alpha=0.5):

        self.n_agents = None
        self.deposition_action = None
        self.actions_list = None
        self.x = None
        self.prev_x = None
        self.t = 0

        # WALL FOLLOW / CORNER DETECTION VARIABLES
        self.z_vector = np.array([0, 0, 1])
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
        self.deposition_flag = None  # tracks whether robots have started depositing material
        self.blinders = None
        self.avg_window_size = moving_avg_window_size  # for eg: if defined as 10, then it will take 10 raw angle values and average it
        self.var_window_size = moving_var_window_size  # function similar to moving_avg_window_size, but will take in the averaged values and then calculate variance for the particular window size
        self.var_queue_size = var_queue_size  # window size used to see the consistency in varience and detect corner based on irregularity
        self.var_threshold = var_threshold  # threshold on which the var_queue_size would tested
        self.slowdown_alpha = slowdown_alpha
        self.three_pt_filter_array = None  # 3 point (one-two-one) filter

    def get_action(self, obs, i):

        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition_action = np.zeros((self.n_agents,))
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.x = np.zeros((self.n_agents,2))
            self.prev_x = np.zeros((self.n_agents, 7, 2))  ##
            self.moving_window = np.zeros((self.n_agents, self.avg_window_size))  ##
            self.variance_window = np.zeros((self.n_agents, self.var_window_size))  ##
            self.wall_follow_flag = np.zeros((self.n_agents,))
            self.variance_counter = np.zeros((self.n_agents,))
            self.corner_flag = np.zeros((self.n_agents,))
            self.corner_counter = np.zeros((self.n_agents,))
            self.moving_var_window = np.zeros((self.n_agents, self.var_queue_size))
            self.deposition_flag = np.zeros((self.n_agents,))
            self.blinders = np.ones((self.n_agents, 1)) * (self.avg_window_size + 10)
            self.three_pt_filter_array = np.zeros((self.n_agents, 3))  # 3 point (one-two-one) filter

        self.increment_history_vars(i)
        self.x = obs['x']
        direction_centroid = direction_to_centroid(sensor_reading[i])
        direction_centroid = np.append(direction_centroid, 0)

        direction_parallel_wall = np.cross(direction_centroid, self.z_vector)

        direction_centroid_norm = direction_centroid / np.linalg.norm(direction_centroid)

        direct_parallel_wall = direction_parallel_wall / np.linalg.norm(direction_parallel_wall)

        if (np.isnan(direct_parallel_wall).any()):
            direct_parallel_wall = direction_parallel_wall

        direction_wise_density, direction_wise_count, direction_wise_size, total_density = check_directional_and_total_density(
            sensor_reading[i])

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

        # if not in wall follow state depositing material and NO corner is detected, enter wall follow state
        if ((self.moving_var_window[i] <= self.var_threshold).all() and (self.wall_follow_flag[i] == 0) and
                self.deposition_flag[i] == 1):
            self.wall_follow_flag[i] = 1
        # if blinders are off (i.e. we haven't JUST entered wall-follow state from p4 state
        if self.blinders[i] == 0:
            # if in wall follow state and a corner is detected, turn wall follow state off,
            # turn robot to beam extrusion state, incrememnt number of corners,
            # MOVE TO P4
            if ((self.wall_follow_flag[i] == 1) and (self.moving_var_window[i] > self.var_threshold).all()):
                self.wall_follow_flag[i] = 0
                self.corner_counter[i] += 1

        else:
            self.blinders[i] = self.blinders[i] - 1

        return self.actions_list[i], self.deposition_action[i]

    def increment_history_vars(self, i):
        # increment position history
        self.prev_x[i][:-1] = self.prev_x[i][1:]
        self.prev_x[i][-1][0] = self.x[i, 0]
        self.prev_x[i][-1][1] = self.x[i, 1]

        ######################################
        ##getting the angle of the direction in which the bot has moved
        # angle wrap around also handled

        # arctan2 usage: np.arctan2(y,x)
        temp_angle = np.arctan2(self.actions_list[i][1], self.actions_list[i][0])

        if ((temp_angle > -0.53) and (temp_angle < 0.53)):
            temp_angle = temp_angle
        else:
            temp_angle = temp_angle if (temp_angle >= 0) else (2 * pi + temp_angle)

        degree_angle = np.rad2deg(temp_angle)

        ####################################
        ##3 pt filter

        self.three_pt_filter_array[i][:-1] = self.three_pt_filter_array[i][1:]
        self.three_pt_filter_array[i][-1] = degree_angle
        temp_3pt = 0.5 * self.three_pt_filter_array[i][1] + 0.25 * (
                self.three_pt_filter_array[i][0] + self.three_pt_filter_array[i][2])

        ###################################

        ##computing moving average

        # adding the degree_value in self.moving_window np array using queue logic, FIFO
        self.moving_window[i][:-1] = self.moving_window[i][1:]
        self.moving_window[i][-1] = temp_3pt
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
