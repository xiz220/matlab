
import numpy as np
import pdb
from stable_baselines.common.vec_env import DummyVecEnv
from controllers.fillet_rule import calculate_centroid


class BlobRule:

    def __init__(self, blob_kwargs):
        self.n_agents = None
        self.deposition_action = None
        self.actions_list = None
        self.x = None
        self.t = 0
        self.max_t = None

        self.robot_state = None

        # STOP CONDIITON
        self.n_corners = None

        # init controller modules
        self.blob_controller = BlobController(**blob_kwargs)

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
            self.robot_state = np.zeros((self.n_agents,))
            self.n_corners = np.zeros((self.n_agents,))
            if type(self.env) is not DummyVecEnv:
                self.max_t = self.env.max_episode_length
            else:
                self.max_t = self.env.get_attr("max_episode_length",0)[0]
        self.t += 1
        if self.t % 10 == 0:
            print(self.t)
        self.x = obs['x']

        for i in range(self.n_agents):
            self.actions_list[i], self.deposition_action[i] = self.blob_controller.get_action(obs,i)
        return np.concatenate((np.array(self.actions_list), np.array(self.deposition_action).reshape(self.n_agents, 1)),
                              axis=1)

    def check_stop_position(self):
        pass

    def set_env(self, env):
        self.env = env


class BlobController:

    def __init__(self):
        self.n_agents = None
        self.t = None
        self.state = None
        self.blob_radii = None
        self.actions_list = None
        self.deposition_action = None
        self.x = None
        
    

    def get_action(self, obs, i):
        sensor_reading = obs['sensor_readings'][i,:,:]
        if self.n_agents is None:
            self.n_agents = len(obs['sensor_readings'])
            self.deposition_action = np.zeros((self.n_agents,))
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.x = np.zeros((self.n_agents, 2))
            self.state = np.zeros((self.n_agents,))
            self.blob_radii = np.zeros((self.n_agents,))
            self.deposition_counter = np.zeros((self.n_agents,))

        self.x = obs['x']
        
        #direction_centroid = direction_to_centroid(sensor_reading[i])
        total_density = np.count_nonzero(sensor_reading) / sensor_reading.size

        dist_to_center = np.linalg.norm(self.x[i,:]-[275,275])
        self.blob_radii[i] = np.max([3, 30-dist_to_center/10])
        #print('dist: ',dist_to_center, ' radius: ', self.blob_radii[i])
        if self.state[i] == 0: #random movement state
            if total_density == 0: #initialize deposition
                self.actions_list[i] = np.array([0,0])
                self.deposition_action[i] = 1
                self.deposition_counter[i] = self.get_time_from_radius(self.blob_radii[i])
                #print(["radius: ", self.blob_radii[i], " time: ", self.deposition_counter[i]])
                self.state[i] = 1 #set to DEPOSITION state
            else:
                self.actions_list[i] = (20*(np.random.rand(1,2)-0.5)).flatten()
                self.deposition_action[i] = 0

            # check if radius plus buffer is empty
            for j in range(int(sensor_reading.shape[0]/2)):
                start_ind = int(sensor_reading.shape[0]/2)-j
                end_ind = int(sensor_reading.shape[0]/2)+j
                obs_subset = sensor_reading[start_ind:end_ind+1, start_ind:end_ind+1]
                if sum(sum(obs_subset)) == 0:
                    if j > np.floor(self.blob_radii[i]*np.sqrt(2) + 3*np.max([self.blob_radii[i]/8,1])):
                        # set to DEPOSITION state
                        self.actions_list[i] = np.array([0, 0])
                        self.deposition_action[i] = 1
                        self.deposition_counter[i] = self.get_time_from_radius(self.blob_radii[i])
                        self.state[i] = 1  # set to DEPOSITION state
                else:
                    break

        elif self.state[i] == 1: #deposition state
            self.actions_list[i] = np.array([0,0])
            self.deposition_action[i] = 1
            self.deposition_counter[i] -= 1
            if self.deposition_counter[i] <= 0:
                self.state[i] = 0 #set back to random walk

        return (self.actions_list[i],self.deposition_action[i])

    def get_time_from_radius(self, radius):
        # returns estimate of time in PIXELS, TODO: adjust to take scale into account
        return int(radius**2 * np.pi)
        
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