import gym
import numpy as np
import matplotlib.pyplot as plt


class OccupancyGridEnv(gym.Env):
    " A gym environment for swarm construction on a lattice using an occupancy grid representation"
    # OpenAI Gym Class Metadata
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lattice_img_path=None, n_agents=3):
        self.n_agents = n_agents
        self.fig = None

        if lattice_img_path is not None:
            #get lattice image from file
            lattice_img = None
            self.occupancy = self.create_occupancy_from_img(lattice_img)
        else:
            raise NotImplementedError('Must provide lattice image to initialize environment')

        self.init_agents()


    def reset(self):
        self.init_agents()

    def step(self, action):
        assert not np.isnan(action).any(), "Actions must not be NaN."
        self.update_state(action)

    def render(self, mode='human'):
        if mode== 'human':
            plt.ion()

        if self.fig is None:
            #initialize figure
            pass

    def get_obs(self):
        return self.x


    def update_state(self, action):
        self.x = self.x+action

    def create_occupancy_from_img(self,img):
