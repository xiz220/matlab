from env_v1 import OccupancyGridEnv
import numpy as np

env = OccupancyGridEnv(lattice_img_path='images/candidate_3.png')

for i in range(100):
    deposition_action = i%5==0
    obs, r = env.step(np.concatenate(((np.random.rand(3,2)-0.5)*4, deposition_action*np.ones((3,1))),axis=1))
    print(obs)
    env.render()