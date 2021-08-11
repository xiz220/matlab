import numpy as np


class ProximityRule:

    def __init__(self, line_length=10, straight_line=False, slowdown_alpha=1.0):
        self.n_agents = None
        self.deposition = None
        self.actions_list = None
        self.prev_x = [None, None]
        self.prev_action = None
        self.slowdown_alpha = slowdown_alpha
        self.straight_line = straight_line
        self.state = None
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
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_x = [np.zeros((self.n_agents, 2)), np.zeros((self.n_agents, 2))]
            self.prev_action = self.actions_list
            self.state = ['looking' for _ in range(self.n_agents)]
        deposition_action = []
        x = obs['x']

        for i in range(self.n_agents):
            obs = sensor_reading[i, :, :]
            n = int(obs.shape[0]/2)
            if type(self.state[i]) == int:
                if self.state[i] == 0:
                    self.state[i] = 'looking'
                else:
                    self.state[i] -= 1
                deposition_action.append(0)
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 40 * self.slowdown_alpha
            elif self.state[i] == 'looking' and (obs[n-1,n] != 0 or obs[n,n-1] != 0 or obs[n+1,n] != 0 or obs[n,n+1] != 0):
                self.state[i] = 'building'
                deposition_action.append(1)
                self.deposition[i] = self.line_length
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 20 * self.slowdown_alpha
            elif self.state[i] == 'building' and (obs[n-1,n] == 1 or obs[n,n-1] == 1 or obs[n+1,n] == 1 or obs[n,n+1] == 1):
                deposition_action.append(1)
                self.deposition[i] = self.line_length
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 20 * self.slowdown_alpha
            elif self.deposition[i] != 0:
                deposition_action.append(1)
                self.deposition[i] = self.deposition[i] - 1
                if self.straight_line:
                    self.actions_list[i] = self.prev_action[i]
                else:
                    self.actions_list[i] = (np.random.rand(2) - 0.5) * 40 * self.slowdown_alpha
            else:
                deposition_action.append(0)
                self.state[i] = 5
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 40 * self.slowdown_alpha

            if (x[i,:] == self.prev_x[0][i,:]).all() and (x[i,:] == self.prev_x[1][i]).all():
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 150
                self.deposition[i] = 0
                self.state[i] = 'looking'

        if self.t % 5 == 0:
            self.prev_x[0] = self.prev_x[1]
            self.prev_x[1] = x

        self.t = self.t + 1
        self.prev_action = self.actions_list
        return np.concatenate((np.array(self.actions_list), np.array(deposition_action).reshape(self.n_agents, 1)),
                              axis=1)
