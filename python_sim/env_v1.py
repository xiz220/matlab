import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import rgb2gray


class OccupancyGridEnv(gym.Env):
    " A gym environment for swarm construction on a lattice using an occupancy grid representation"
    # OpenAI Gym Class Metadata
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lattice_img_path=None, n_agents=3):
        self.n_agents = n_agents
        self.fig = None

        if lattice_img_path is not None:
            # get lattice image from file
            self.lattice_img = mpimg.imread(lattice_img_path)
            self.occupancy = self.create_occupancy_from_img(self.lattice_img)
        else:
            raise NotImplementedError('Must provide lattice image to initialize environment')

        self.x = self.init_agents()
        self.sensor_occ_radius = 3
        self.update_sensor_reading = self.update_sensor_reading_occupancy #self.update_sensor_reading_laser
        self.sensor_reading = [[] for _ in range(self.n_agents)] # a list of length n_agents

    def init_agents(self):
        num_agents = 0
        x = []
        while num_agents < self.n_agents:
            candidate_x = np.random.randint(0,self.occupancy.shape[1])
            candidate_y = np.random.randint(0,self.occupancy.shape[0])
            if not self.is_occupied(candidate_x,candidate_y):
                x.append([candidate_x,candidate_y])
                num_agents = num_agents + 1

        return np.array(x)

    def reset(self):
        self.init_agents()

    def step(self, action):
        assert not np.isnan(action).any(), "Actions must not be NaN."
        self.update_state(action)
        self.update_sensor_reading()

        obs = self.get_obs()
        reward = 0
        return obs, reward

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
            self.robot_handle = self.ax.scatter(self.x[:, 0], self.x[:, 1], 20, 'red')

        self.robot_handle.set_offsets(self.x)


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


        # Add occupancy grid
        self.ax.imshow(1-self.occupancy, cmap='gray', vmin=0, vmax=1)
        #plt.show(block=True)

    def get_obs(self):
        return [self.x, self.sensor_reading]

    def update_state(self, action):
        candidate_state = self.x + action
        for i in range(self.n_agents):
            if self.is_occupied(candidate_state[i,0], candidate_state[i,1]):
                candidate_state[i,:] = self.x[i,:]

        self.x = candidate_state

    def update_sensor_reading_laser(self):
        """ Updates self.sensor_reading with the new sensor readings.
        self.sensor_reading[i] holds a numpy array with the ith robot's sensor reading.
        self

        Laser scanner should return n readings parameterized by an angle theta. The readings should be the distance
        along the line at that angle to the nearest obstacle, saturated at some sensor radius value."""

        pass

    def update_sensor_reading_occupancy(self):
        for i in range(self.n_agents):
            self.sensor_reading[i] = np.zeros((2*self.sensor_occ_radius+1, 2*self.sensor_occ_radius+1))
            for i_x in range(2*self.sensor_occ_radius+1):
                for i_y in range(2*self.sensor_occ_radius+1):
                    if not self.is_ob(self.x[i,0]-self.sensor_occ_radius+i_x,self.x[i,1]-self.sensor_occ_radius+i_y):
                        self.sensor_reading[i][2*self.sensor_occ_radius - i_y, i_x] = self.is_occupied(self.x[i,0]-self.sensor_occ_radius+i_x,self.x[i,1]-self.sensor_occ_radius+i_y)
                    else:
                        self.sensor_reading[i][2 * self.sensor_occ_radius - i_y, i_x] = 1


    def create_occupancy_from_img(self, img):
        occupancy_map = rgb2gray(img)
        occupancy_map = np.where(occupancy_map < 0.5, 1, 0)
        return occupancy_map

    def is_occupied(self,x,y):
        y_floor = int(y)
        x_floor = int(x)

        if self.is_ob(x_floor, y_floor):
            raise ValueError("(%d,%d) is out of bounds" % (x_floor,y_floor) )

        return self.occupancy[(self.occupancy.shape[0] - 1) - y_floor, x_floor] == 1

    def is_ob(self,x,y):
        x = int(x)
        y = int(y)
        return not (x>=0 and x<self.occupancy.shape[1] and y>=0 and y<self.occupancy.shape[0])

if __name__ == "__main__":
    lattice_img_path = "images/candidate_1.png"
    env = OccupancyGridEnv(lattice_img_path=lattice_img_path, n_agents=3)
    # print(env.occupancy)
    # plt.imshow(env.occupancy, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()

