from envs.env_v1 import OccupancyGridEnv
from render_utils import render_model
from fillet_rule import FilletRule
from proximity_rule import ProximityRule
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import cProfile
import numpy as np

rule = ProximityRule()

def env_callable():
    return gym.make('OccupancyGrid-v0', lattice_img_path='images/candidate_3_clean.png',
             sensor_model="update_sensor_reading_occupancy", n_agents = 10)


env = DummyVecEnv([env_callable])

#cProfile.run('render_model(rule, env,video_folder=\'videos\',name_prefix=\'video\',n_episodes=1)')

render_model(rule, env,
            video_folder='videos',
            name_prefix='video',
            n_episodes=1)