import argparse
import toml
import importlib
from envs.env_v1 import OccupancyGridEnv
from utils import clean_dir_name, clean_file_name
from render_utils import render_model
import gym
from pathlib import Path
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib.image as mpimg


def main():
    parser = argparse.ArgumentParser(description='Run stiffness analysis on an image.')
    parser.add_argument('-img_filepath', default=None, help='Filepath (from python_sim) of the image')
    args = parser.parse_args()


    #save image on last frame
    filepath = Path(args.img_filepath)

    import matlab.engine
    eng = matlab.engine.start_matlab()
    result = eng.calc_stiffness(str(filepath))
    stiffness = result[0][0]
    vol_frac = result[0][1]
    file = open(str(filepath.parents[0] / 'stiffness.txt'), 'w')
    file.write("stiffness: " + str(stiffness) + '\n')
    file.write("volume fraction: " + str(vol_frac))
    file.close()



if __name__ == '__main__':
    main()  # Don't pollute namespace
