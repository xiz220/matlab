import numpy as np


class ProximityRule:

    def __init__(self, line_length=10):
        self.n_agents = None
        self.deposition = None
        self.deposition_counter = None
        self.actions_list = None
        self.directions = None
        self.prev_x = [None, None]
        self.t = 0
        self.line_length = line_length

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
            self.deposition_counter = np.zeros((self.n_agents,))
            self.directions = (np.random.rand(self.n_agents,2)-0.5)
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_x = [np.zeros((self.n_agents, 2)), np.zeros((self.n_agents, 2))]
        deposition_action = []
        x = obs['x']

        for i in range(self.n_agents):
            obs = sensor_reading[i, :, :]
            if obs[2,3] == 1 or obs[3,2] == 1 or obs[4,3] == 1 or obs[3,4] == 1: #todo generalize this to different sensor radii
                deposition_action.append(1)
                self.deposition[i] = self.line_length
                self.deposition_counter[i] += 1
            elif self.deposition[i] != 0:
                deposition_action.append(1)
                self.deposition[i] = self.deposition[i] - 1
                self.deposition_counter[i] += 1
            else:
                deposition_action.append(0)
                self.deposition_counter[i] = 0

            self.actions_list[i] = (np.random.rand(2) - 0.5) * 0.3

            if (x[i,:] == self.prev_x[0][i,:]).all() and (x[i,:] == self.prev_x[1][i]).all():
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 15
                self.deposition[i] = 0

        if self.t % 5 == 0:
            self.prev_x[0] = self.prev_x[1]
            self.prev_x[1] = x

        self.t = self.t + 1
        self.directions = self.directions + np.array(self.actions_list)

        #normalize velocities
        dir_norms = np.linalg.norm(self.directions, axis=1)
        self.directions[:,0] = self.directions[:,0] * 1/dir_norms
        self.directions[:,1] = self.directions[:,1] * 1/dir_norms
        return np.concatenate((self.directions*0.3, np.array(deposition_action).reshape(self.n_agents, 1)),
                              axis=1)
