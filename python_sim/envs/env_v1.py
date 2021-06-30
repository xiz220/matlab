import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
from utils import rgb2gray, clean_file_name
from os import path
from pathlib import Path


class OccupancyGridEnv(gym.Env):
    " A gym environment for swarm construction on a lattice using an occupancy grid representation"
    # OpenAI Gym Class Metadata
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lattice_img_path=None,
                 image_scale=1.0, # microns per grid unit
                 n_agents=3,
                 sensor_model='update_sensor_reading_laser',
                 sensor_occ_radius=3,
                 max_episode_length=500,
                 max_deposition_radius=1,
                 motion_model='unrestricted'):
        self.image_scale = image_scale
        self.n_agents = n_agents
        self.max_episode_length = max_episode_length
        self.ep_step = 0
        self.fig = None


        if lattice_img_path is not None:
            # get lattice image from file
            self.lattice_img = mpimg.imread(lattice_img_path)
            self.occupancy = self.create_occupancy_from_img(self.lattice_img)
        else:
            raise NotImplementedError('Must provide lattice image to initialize environment')

        self.motion_model = motion_model

        self.x = self.init_agents()
        self.sensor_occ_radius = int(sensor_occ_radius/self.image_scale)

        self.update_sensor_reading = getattr(self, sensor_model)  # self.update_sensor_reading_laser
        self.sensor_reading = None
        self.deposition_sorter = None #holds distance vector so as to avoid duplicating this calculation
        self.max_deposition_radius = int(max_deposition_radius/self.image_scale)
        self.deposition_rate = int(50*5/(self.image_scale**2)) #theoretically 5 um^2/s (at 1um thick) at 50x speedup
        if self.max_deposition_radius < 1 or self.sensor_occ_radius < 1 or self.deposition_rate < 1:
            print("IMG SCALE SEEMS OFF: deposition or sensor radius less than 1")

        self.update_sensor_reading() #initialize sensor reading to be of correct dimension

        self._set_observation_space()
        self.action_space = gym.spaces.Box(shape=(self.n_agents, 3), low=-np.inf, high=np.inf, dtype=np.float32)

        self.laser_angle = np.zeros(self.n_agents)
        print('deposition radius: ', self.max_deposition_radius,' pixels')
        print('deposition rate: ', self.deposition_rate,' pixels/timestep')
        print('sensor radius: ', self.sensor_occ_radius,' pixels')


    def init_agents(self):
        """ Initialize agents to be at random unoccupied positions. This initializes the self.x variable
        to be a n_agents x 2 np array"""
        num_agents = 0
        x = []
        while num_agents < self.n_agents:
            candidate_x = np.random.randint(0, self.occupancy.shape[1])
            candidate_y = np.random.randint(0, self.occupancy.shape[0])
            if not self.is_occupied(candidate_x, candidate_y):
                x.append([candidate_x, candidate_y])
                num_agents = num_agents + 1

        return np.array(x)

    def reset(self):
        """Resets the environment. Resets the agents to random positions, and resets the episode step counter.
        Returns: a new observation (dictionary: {'x': np array, 'sensor_readings': np array})
        """
        self.init_agents()
        self.ep_step = 0
        return self.get_obs()

    def step(self, action):
        """Steps the environment forward one timestep, with robots taking actions according to inputs.
        Input:
            action:  a n_agents x 3 np array, where the ith row represents
                the ith agents [x_action, y_action, deposition_action]. Deposition action is binary, whether to
                 deposit at current position
         Returns:
             obs: dictionary observation {'x': np array, 'sensor_readings': np array}
             reward: scalar, not currently implemented
             done: whether the environment episode is over
             info: not currently used
             """
        assert not np.isnan(action).any(), "Actions must not be NaN."
        self.update_state(action)
        self.update_sensor_reading()

        obs = self.get_obs()
        reward = 0

        self.ep_step = self.ep_step+1
        done = self.ep_step >= self.max_episode_length
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            plt.ion()

        # Content data limits in data units.
        data_xlim = [0, self.occupancy.shape[1]]
        data_ylim = [0, self.occupancy.shape[0]]
        data_aspect_ratio = (data_xlim[1] - data_xlim[0]) / (data_ylim[1] - data_ylim[0])

        if self.fig is None:
            # initialize figure
            # Aesthetic parameters.
            # Figure aspect ratio.
            fig_aspect_ratio = 16.0 / 9.0  # Aspect ratio of video.
            fig_pixel_height = 1080  # Height of video in pixels.
            dpi = 300  # Pixels per inch (affects fonts and apparent size of inch-scale objects).

            # Set the figure to obtain aspect ratio and pixel size.
            fig_w = fig_pixel_height / dpi * fig_aspect_ratio  # inches
            fig_h = fig_pixel_height / dpi  # inches
            self.fig, self.ax = plt.subplots(1, 1,
                                             figsize=(fig_w, fig_h),
                                             constrained_layout=True,
                                             dpi=dpi)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')

            # Set axes limits which display the workspace nicely.
            self.ax.set_xlim(data_xlim[0], data_xlim[1])
            self.ax.set_ylim(data_ylim[0], data_ylim[1])

            # Setting axis equal should be redundant given figure size and limits,
            # but gives a somewhat better interactive resizing behavior.
            self.ax.set_aspect('equal')

            # Draw robots
            self.robot_handle = self.ax.scatter(self.x[:, 0], self.x[:, 1], 10, 'red')

            # Add occupancy grid
            occ_data = (1-((self.occupancy==1).astype('float')+0.7*(self.occupancy==2).astype('float')))
            self.occ_render = self.ax.imshow(occ_data, cmap='gray', vmin=0, vmax=1)

            # Draw scale line
            self.ax.plot([50, 50+1000/self.image_scale], [25,25])
            self.ax.text(x=1,y=22,s='1 mm',fontsize=4,color='tab:blue')
            self.ax.text(x=1,y=5,s='timescale: 50x',fontsize=4,color='tab:blue')

        self.robot_handle.set_offsets(self.x)

        # update occupancy grid
        occ_data = (1-((self.occupancy==1).astype('float')+0.7*(self.occupancy==2).astype('float')))
        self.occ_render.set_data(occ_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


        # plt.show(block=True)

        if mode == 'rgb_array':
            s, (width, height) = self.fig.canvas.print_to_buffer()
            rgb = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            return rgb

    def get_obs(self):
        return {'x': self.x, 'sensor_readings': self.sensor_reading}

    def update_state(self, action):
        deposition_action = action[:, 2:]
        # robot deposition of material
        self.perform_deposition(deposition_action)

        movement_action = action[:, 0:2]
        candidate_state = self.x + (1/self.image_scale)*movement_action
        for i in range(self.n_agents):
            if self.motion_model=='restricted':
                if self.is_occupied(candidate_state[i, 0], candidate_state[i, 1]):
                    candidate_state[i, :] = self.x[i, :]
            if self.motion_model=='unrestricted':
                if self.is_ob(candidate_state[i,0], candidate_state[i,1]):
                    candidate_state[i, :] = self.x[i, :]

        self.x = candidate_state



    def perform_deposition(self, deposition_action):
        # when self.max_deposition_radius is 1, this reduces to "only deposit in your current cell"

        #generate list of distances to the other indices in the local occupancy grid area
        if self.deposition_sorter is None:
            dist = []
            inds = []
            for i in range(2*self.max_deposition_radius + 1):
                for j in range(2*self.max_deposition_radius + 1):
                    dist.append(np.linalg.norm(np.array([i-self.max_deposition_radius,j-self.max_deposition_radius])))
                    inds.append([i-self.max_deposition_radius, j-self.max_deposition_radius])
            self.deposition_sorter = np.argsort(np.array(dist))
            self.deposition_inds = np.array(inds)

        deposition_action = np.concatenate((deposition_action, deposition_action), axis=1)
        deposition_centers_array = self.x[deposition_action == 1].reshape((-1, 2))
        for i in range(deposition_centers_array.shape[0]):
            row_ind, col_ind = self.xy_to_occ_ind(deposition_centers_array[i, 0], deposition_centers_array[i, 1])
            #local_occ = self.get_window(self.x[i,0], self.x[i,1], self.max_deposition_radius)
            for _ in range(self.deposition_rate):
                for k in range(self.deposition_sorter.shape[0]):
                    cand_row = self.deposition_inds[self.deposition_sorter[k],0] + row_ind
                    cand_col = self.deposition_inds[self.deposition_sorter[k],1] + col_ind
                    cand_x, cand_y = self.occ_ind_to_xy(cand_row, cand_col)
                    if (not self.is_ob(cand_x,cand_y)) and np.linalg.norm(np.array([row_ind-cand_row, col_ind-cand_col])) < self.max_deposition_radius:
                        if self.occupancy[cand_row,cand_col] == 0:
                            self.occupancy[cand_row,cand_col] = 2
                            break
            # print('deposited at: %d, %d' % (deposition_list[i,0], deposition_list[i,1]))

    def update_sensor_reading_laser(self, laser_resolution=0.1):
        """ Updates self.sensor_reading with the new sensor readings.
        self.sensor_reading[i] holds a numpy array with the ith robot's sensor reading.
        self

        Laser scanner should return n readings parameterized by an angle theta. The readings should be the distance
        along the line at that angle to the nearest obstacle, saturated at some sensor radius value."""

        for i in range(self.n_agents):
            cur_dist = 0
            cur_x = self.x[i]
            cur_angle = self.laser_angle[i]
            while not self.is_occupied(cur_x[0], cur_x[1]):
                cur_x[0] += laser_resolution * np.cos(cur_angle)
                cur_x[1] += laser_resolution * np.sin(cur_angle)
                cur_dist += laser_resolution
                if cur_dist > self.sensor_occ_radius:
                    cur_dist = self.sensor_occ_radius
                    break
            self.sensor_reading[i].append(cur_dist)

        print(self.sensor_reading)

    def update_sensor_reading_occupancy(self):
        """
         Updates self.sensor_reading, an n_agents x sensor_reading_rows x sensor_reading_cols numpy array
        """
        sensor_reading = [np.zeros((2 * self.sensor_occ_radius + 1, 2 * self.sensor_occ_radius + 1)) for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            sensor_reading[i] = self.get_window(self.x[i,0], self.x[i,1],self.sensor_occ_radius)

        self.sensor_reading = np.array(sensor_reading)

    def create_occupancy_from_img(self, img):
        occupancy_map = rgb2gray(img)
        occupancy_map = np.where(occupancy_map < 0.5, 1, 0)
        return occupancy_map

    def xy_to_occ_ind(self, x, y):
        return int(y), int(x)

    def occ_ind_to_xy(self, row, col):
        return col, row

    def is_occupied(self, x, y):

        if self.is_ob(x, y):
            return 1
            #raise ValueError("(%d,%d) is out of bounds" % (x, y))

        row_ind, col_ind = self.xy_to_occ_ind(x, y)
        return self.occupancy[row_ind, col_ind] >= 1

    def is_ob(self, x, y):
        x = int(x)
        y = int(y)
        return not (0 <= x < self.occupancy.shape[1] and 0 <= y < self.occupancy.shape[0])

    def _set_observation_space(self):
        """ Set the fixed observation space based on the observation function. """
        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(shape=(self.n_agents,2), low=-np.inf, high=np.inf, dtype=np.float32),
            'sensor_readings': gym.spaces.Box(shape=self.sensor_reading.shape,low=-np.inf, high=np.inf, dtype=np.float32)
        })

    def get_window(self, x, y, radius=2):
        window = np.zeros((radius*2+1,radius*2+1))
        for i_x in range(2 * radius + 1):
                for i_y in range(2 * radius + 1):
                    if not self.is_ob(x - radius + i_x,
                                      y -  radius + i_y):
                        window[2 * radius - i_y, i_x] = self.is_occupied(
                            x - radius + i_x, y - radius + i_y)
                    else:
                        window[2 * radius - i_y, i_x] = 1

        return window
