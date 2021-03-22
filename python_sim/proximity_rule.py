import numpy as np

class ProximityRule:

    def __init__(self):
        pass

    def get_action(self, obs):
        """
        Takes in an observation and returns the next action to take to do a wall-following sort of filleting action
        :param obs: observation dict {'x': robot_states (n_agents x 2 np array), 'sensor_readings': (n_agents x row_dim x col_dim) np array}
        :return: n_agents x 3 np array, where row i represents the i'th robot's: [x_action, y_action, deposition_action]
        """

        sensor_reading = obs['sensor_readings']

        deposition_action = []
        for i in range(len(sensor_reading)):
            obs = sensor_reading[i,:,:]
            if (obs[2, 3] == 1 or obs[3, 2] == 1 or obs[4, 3] == 1 or obs[3, 4] == 1):
                deposition_action.append(1)
            else:
                deposition_action.append(0)
        return np.concatenate(((np.random.rand(3, 2) - 0.5) * 4, np.array(deposition_action).reshape(3,1)),axis=1)

