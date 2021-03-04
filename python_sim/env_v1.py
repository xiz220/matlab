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
            lattice_img = mpimg.imread(lattice_img_path)
            self.occupancy = self.create_occupancy_from_img(lattice_img)
        else:
            raise NotImplementedError('Must provide lattice image to initialize environment')

        self.init_agents()

    def init_agents(self):
        pass

    def reset(self):
        self.init_agents()

    def step(self, action):
        assert not np.isnan(action).any(), "Actions must not be NaN."
        self.update_state(action)

    def render(self, mode='human'):
        if mode == 'human':
            plt.ion()

        if self.fig is None:
            # initialize figure
            pass

    def get_obs(self):
        return self.x

    def update_state(self, action):
        self.x = self.x + action

    def create_occupancy_from_img(self, img):
        occupancy_map = rgb2gray(img)
        occupancy_map = np.where(occupancy_map > 0.5, 1, 0)
        return occupancy_map


if __name__ == "__main__":
    lattice_img_path = "images/candidate_1.png"
    env = OccupancyGridEnv(lattice_img_path=lattice_img_path, n_agents=3)
    # print(env.occupancy)
    # plt.imshow(env.occupancy, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()

