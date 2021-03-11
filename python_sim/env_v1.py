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

    def init_agents(self):
        num_agents = 0
        x = []
        while num_agents < self.n_agents:
            candidate_x = np.random.randint(0,self.occupancy.shape[1])
            candidate_y = np.random.randint(0,self.occupancy.shape[0])
            if self.occupancy[(self.occupancy.shape[0]-1)-candidate_y, candidate_x] == 0:
                x.append([candidate_x,candidate_y])
                num_agents = num_agents + 1

        return np.array(x)

    def reset(self):
        self.init_agents()

    def step(self, action):
        assert not np.isnan(action).any(), "Actions must not be NaN."
        self.update_state(action)

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
        plt.show(block=True)

    def get_obs(self):
        return self.x

    def update_state(self, action):
        self.x = self.x + action

    def create_occupancy_from_img(self, img):
        occupancy_map = rgb2gray(img)
        occupancy_map = np.where(occupancy_map < 0.5, 1, 0)
        return occupancy_map


if __name__ == "__main__":
    lattice_img_path = "images/candidate_1.png"
    env = OccupancyGridEnv(lattice_img_path=lattice_img_path, n_agents=3)
    # print(env.occupancy)
    # plt.imshow(env.occupancy, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()

