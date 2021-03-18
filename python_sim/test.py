from env_v1 import OccupancyGridEnv
import numpy as np

env = OccupancyGridEnv(lattice_img_path='images/candidate_3.png')

def proximity_rule(obs):
    sensor_reading = obs[1]
    deposition_action = []
    for i in range(len(sensor_reading)):
        obs = sensor_reading[i]
        if (obs[2,3]==1 or obs[3,2]==1 or obs[4,3]==1 or obs[3,4]==1):
            deposition_action.append(1)
        else:
            deposition_action.append(0)
    return np.array(deposition_action)

action = np.concatenate(((np.random.rand(3, 2) - 0.5) * 4, np.zeros((3, 1))), axis=1)

for i in range(100):
    deposition_action = i % 5 == 0
    obs, r = env.step(action)


    print(obs)
    deposition_action = proximity_rule(obs)
    action = np.concatenate(((np.random.rand(3, 2) - 0.5) * 4,deposition_action.reshape((-1,1))), axis=1)
    env.render()
