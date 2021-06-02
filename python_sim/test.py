from envs.env_v1 import OccupancyGridEnv
import numpy as np
from proximity_rule import ProximityRule
from fillet_rule import FilletRule

n_agents = 3
render_interval = 1
env = OccupancyGridEnv(lattice_img_path='images/candidate_3_clean.png', sensor_model="update_sensor_reading_occupancy",
                       n_agents=n_agents,
                       max_deposition_radius=15)
rule = ProximityRule()

action = np.concatenate(((np.random.rand(n_agents, 2) - 0.5) * 4, np.zeros((n_agents, 1))), axis=1)

for i in range(1000):
    #deposition_action = i % 5 == 0
    #action = np.concatenate((np.zeros((n_agents,2)),np.ones((n_agents,1))),axis=1)
    obs, r, done, info = env.step(action)

    action = rule.get_action(obs)
    if i % render_interval == 0:
        env.render()
