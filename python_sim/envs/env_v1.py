import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import rgb2gray


class OccupancyGridEnv(gym.Env):
    " A gym environment for swarm construction on a lattice using an occupancy grid representation"
    # OpenAI Gym Class Metadata
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lattice_img_path=None, n_agents=3, sensor_model='update_sensor_reading_laser'):
        self.n_agents = n_agents
        self.max_episode_length=10
        self.ep_step = 0
        self.fig = None


        if lattice_img_path is not None:
            # get lattice image from file
            self.lattice_img = mpimg.imread(lattice_img_path)
            self.occupancy = self.create_occupancy_from_img(self.lattice_img)
        else:
            raise NotImplementedError('Must provide lattice image to initialize environment')

        self.x = self.init_agents()
        self.sensor_occ_radius = 3
        self.sensor_reading = np.array([np.zeros((2 * self.sensor_occ_radius + 1, 2 * self.sensor_occ_radius + 1)) for _ in range(self.n_agents)])
        self.update_sensor_reading = getattr(self, sensor_model)  # self.update_sensor_reading_laser

        self._set_observation_space()
        self.action_space = gym.spaces.Box(shape=(self.n_agents, 3), low=-np.inf, high=np.inf, dtype=np.float32)

        self.laser_angle = np.zeros(self.n_agents)

    def init_agents(self):
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
        self.init_agents()
        self.ep_step = 0
        return self.get_obs()

    def step(self, action):
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

        self.robot_handle.set_offsets(self.x)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Add occupancy grid
        self.ax.imshow(1 - self.occupancy, cmap='gray', vmin=0, vmax=1)
        # plt.show(block=True)

        if mode == 'rgb_array':
            s, (width, height) = self.fig.canvas.print_to_buffer()
            rgb = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            return rgb

    def get_obs(self):
        return {'x': self.x, 'sensor_readings': self.sensor_reading}

    def update_state(self, action):
        movement_action = action[:, 0:2]
        deposition_action = action[:, 2:]
        candidate_state = self.x + movement_action
        for i in range(self.n_agents):
            if self.is_occupied(candidate_state[i, 0], candidate_state[i, 1]):
                candidate_state[i, :] = self.x[i, :]

        self.x = candidate_state

        # robot deposition of material
        self.perform_deposition(deposition_action)

    def perform_deposition(self, deposition_action):
        deposition_action = np.concatenate((deposition_action, deposition_action), axis=1)
        deposition_list = self.x[deposition_action == 1].reshape((-1, 2))
        for i in range(deposition_list.shape[0]):
            row_ind, col_ind = self.xy_to_occ_ind(deposition_list[i, 0], deposition_list[i, 1])
            self.occupancy[row_ind, col_ind] = 1
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
        # TODO fix this
        """
         Updates self.sensor_reading, an n_agents x sensor_reading_rows x sensor_reading_cols numpy array
        """
        sensor_reading = [np.zeros((2 * self.sensor_occ_radius + 1, 2 * self.sensor_occ_radius + 1)) for _ in range(self.n_agents)]
        for i in range(self.n_agents):

            for i_x in range(2 * self.sensor_occ_radius + 1):
                for i_y in range(2 * self.sensor_occ_radius + 1):
                    if not self.is_ob(self.x[i, 0] - self.sensor_occ_radius + i_x,
                                      self.x[i, 1] - self.sensor_occ_radius + i_y):
                        self.sensor_reading[i][2 * self.sensor_occ_radius - i_y, i_x] = self.is_occupied(
                            self.x[i, 0] - self.sensor_occ_radius + i_x, self.x[i, 1] - self.sensor_occ_radius + i_y)
                    else:
                        self.sensor_reading[i][2 * self.sensor_occ_radius - i_y, i_x] = 1

        self.sensor_reading = np.array(sensor_reading)

    def create_occupancy_from_img(self, img):
        occupancy_map = rgb2gray(img)
        occupancy_map = np.where(occupancy_map < 0.5, 1, 0)
        return occupancy_map

    def xy_to_occ_ind(self, x, y):
        return int(y), int(x)

    def is_occupied(self, x, y):

        if self.is_ob(x, y):
            raise ValueError("(%d,%d) is out of bounds" % (x, y))

        row_ind, col_ind = self.xy_to_occ_ind(x, y)
        return self.occupancy[row_ind, col_ind] == 1

    def is_ob(self, x, y):
        x = int(x)
        y = int(y)
        return not (0 <= x < self.occupancy.shape[1] and 0 <= y < self.occupancy.shape[0])

    def _set_observation_space(self):
        """ Set the fixed observation space based on the observation function. """
        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(shape=(self.n_agents,2), low=-np.inf, high=np.inf, dtype=np.float32),
            'sensor_readings': gym.spaces.Box(shape=(self.n_agents, 2 * self.sensor_occ_radius + 1, 2 * self.sensor_occ_radius + 1),low=-np.inf, high=np.inf, dtype=np.float32)
        })
if __name__ == "__main__":
    lattice_img_path = "images/candidate_1.png"
    env = OccupancyGridEnv(lattice_img_path=lattice_img_path, n_agents=3)
    env.update_sensor_reading_laser()
    # print(env.occupancy)
    # plt.imshow(env.occupancy, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()