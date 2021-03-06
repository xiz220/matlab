import argparse
import toml
import importlib
from envs.env_v1 import OccupancyGridEnv
from utils import clean_dir_name
from render_utils import render_model
import gym
from pathlib import Path
from stable_baselines.common.vec_env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-exp_name', default=None, help='Experiment name')
    parser.add_argument('-cfg', default=None, nargs='+', help='Configuration file path')
    parser.add_argument('-test', action="store_true", default=False, help='Live render instead of saving video')
    args = parser.parse_args()

    exp_cfg = toml.load(args.cfg)
    env_cfg = exp_cfg['env']
    rule_cfg = exp_cfg['rule']
    if 'rule_args' in exp_cfg.keys():
        rule_args = exp_cfg['rule_args']
    else:
        rule_args = {}

    # import and create rule
    rule_class = getattr(importlib.import_module(rule_cfg['rule_file']), rule_cfg['rule_name'])
    rule = rule_class(**rule_args)

    if args.test:  # live rendering
        # create environment
        env = OccupancyGridEnv(**env_cfg)
        done = False
        if rule_cfg['include_env']:
            rule.set_env(env)
        obs = env.reset()
        while not done:
            action = rule.get_action(obs)
            obs, r, done, info = env.step(action)
            env.render()
    else:  # recording a video

        experiment_root_dir = Path(__file__).resolve().parents[0] / 'experiments'

        # Create new experiment directory
        experiment_dir, exp_name = clean_dir_name(args.exp_name, experiment_root_dir)
        experiment_dir.mkdir(parents=True)

        # Write training parameters to file.
        with open(experiment_dir / 'experiment.toml', 'w') as f:
            toml.dump(exp_cfg, f)

        # create environment callable for render env
        def env_callable():
            return gym.make('OccupancyGrid-v0', **env_cfg)

        env = DummyVecEnv([env_callable])

        render_model(rule, env,
                     video_folder=str(experiment_dir),
                     name_prefix='video',
                     n_episodes=1,
                     save_image=True)


if __name__ == '__main__':
    main()  # Don't pollute namespace
