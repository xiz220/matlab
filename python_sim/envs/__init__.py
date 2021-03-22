from gym.envs.registration import register

register(
    id='OccupancyGrid-v0',
    entry_point='envs.env_v1:OccupancyGridEnv',
    max_episode_steps=1000,
)