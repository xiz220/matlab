from env_v1 import OccupancyGridEnv
import numpy as np

env = OccupancyGridEnv(lattice_img_path='images/candidate_3.png')

for i in range(100):
    obs, r = env.step((np.random.rand(3,2)-0.5)*4)
    print(obs)
    env.render()