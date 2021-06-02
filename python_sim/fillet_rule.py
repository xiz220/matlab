import numpy as np

class FilletRule:

    def __init__(self):
        self.directions = None
        self.deposit_in_two = None
        self.deposit_in_one = None
        self.n_agents = None

    def get_action(self, obs):
        """
        Takes in an observation and returns the next action to take to do a wall-following sort of filleting action
        :param obs: observation dict {'x': robot_states (n_agents x 2 np array), 'sensor_readings': (n_agents x row_dim x col_dim) np array}
        :return: n_agents x 3 np array, where row i represents the i'th robot's: [x_action, y_action, deposition_action]
        """

        sensor_reading = obs['sensor_readings']

        if self.directions is None:
            # initialize directions
            self.n_agents = sensor_reading.shape[0]
            self.directions = np.zeros((self.n_agents,2))
            self.deposit_in_two = np.zeros((self.n_agents,))
            self.deposit_in_one = np.zeros((self.n_agents,))

        deposition_action = []

        for i in range(self.n_agents):
            current_reading = sensor_reading[i,:,:]

            #xy centroid, where x is measured from left, y from bottom
            new_centroid = calculate_centroid(current_reading)
            direction_to_centroid = new_centroid - np.floor((current_reading.shape[0]-1)/2)*np.ones((2,))
            new_direction = np.matmul(np.array([[0, -1], [1, 0]]),direction_to_centroid.reshape((2,1))).flatten() #90 degree clockwise rotation

            old_direction = self.directions[i,:]
            angle_between = np.rad2deg(np.arccos(np.dot(new_direction*(1/np.linalg.norm(new_direction)),
                                                                    old_direction*(1/np.linalg.norm(old_direction)))))
            if np.isnan(angle_between).any():
                angle_between = 0
            if not self.deposit_in_one[i] and not self.deposit_in_two[i]:
                self.directions[i, :] = new_direction/np.linalg.norm(new_direction)

            if self.deposit_in_one[i]:
                self.deposit_in_two[i] = 0
                self.deposit_in_one[i] = 0
                deposition_action.append(1)
            else:
                deposition_action.append(0)

            if self.deposit_in_two[i]:
                self.deposit_in_two[i] = 0
                self.deposit_in_one[i] = 1

            if abs(angle_between)>60:
                self.deposit_in_two[i] = 1


        return np.concatenate((self.directions + 1.0*(np.random.rand(self.n_agents,2)-0.5),np.array(deposition_action).reshape((self.n_agents,1))), axis=1)


def calculate_centroid(matrix):
    """"
    Arguments:
        matrix : nxn numpy array of occupancy grid sensor readings, where n must be odd

    Returns:
        the [x,y] centroid of the matrix weighted by sensor value (binary), where x is measured from the left side of
        the matrix, and y is measured up from the bottom of the matrix
    """
    row_centroid = np.sum(np.multiply(np.sum(matrix,axis=1), np.array(range(matrix.shape[0]))))/np.sum(matrix)
    col_centroid = np.sum(np.multiply(np.sum(matrix,axis=0), np.array(range(matrix.shape[1]))))/np.sum(matrix)

    centroid = np.array([col_centroid, matrix.shape[0] - row_centroid])

    if np.isnan(centroid).any():
        centroid = np.array([0,0])
    return centroid
