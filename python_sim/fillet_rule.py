import numpy as np

class FilletRule:

    def __init__(self):
        self.directions = None

    def get_action(self, obs):
        sensor_reading = obs[1]

        if self.directions is None:
            # initialize directions
            self.directions = np.zeros((len(sensor_reading),2))

        deposition_action = []

        for i in range(len(sensor_reading)):
            current_reading = sensor_reading[i]

            #xy centroid, where x is measured from left, y from bottom
            new_centroid = self.calculate_centroid(current_reading)
            direction_to_centroid = new_centroid - np.floor(current_reading.shape[0])*np.ones((2,))
            new_direction = np.array([[0, -1], [1, 0]])*direction_to_centroid #90 degree clockwise rotation

            old_direction = self.directions[i,:]
            angle_between = np.rad2deg(np.arccos(np.dot(new_direction*(1/np.linalg.norm(new_direction),
                                                                    old_direction*(1/np.linalg.norm(old_direction))))))

            if abs(angle_between)<60:
                deposition_action.append(1)
            else:
                deposition_action.append(0)

        return np.array(deposition_action)


    def calculate_centroid(self, matrix):
        """"
        Arguments:
            matrix : nxn numpy array of occupancy grid sensor readings, where n must be odd

        Returns:
            the [x,y] centroid of the matrix weighted by sensor value (binary), where x is measured from the left side of
            the matrix, and y is measured up from the bottom of the matrix
        """
        row_centroid = np.sum(np.multiply(np.sum(matrix,axis=1), np.array(range(matrix.shape[0]))))/np.sum(matrix)
        col_centroid = np.sum(np.multiply(np.sum(matrix,axis=0), np.array(range(matrix.shape[1]))))/np.sum(matrix)
        return np.array([col_centroid, matrix.shape[0] - row_centroid])
