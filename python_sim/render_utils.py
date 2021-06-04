"""
Tools for visualizing model behavior.
- A function render_model(model, env, ...) which saves a video.
- A function display_model(model, env) which displays an infinite loop of episodes.
- A command line tool to access these functions.
For help with the command line tool, run
    python render_model.py --help
"""

import itertools
import pdb
from pathlib import Path
from os import path

import numpy as np
import matplotlib.image as mpimg
from utils import clean_file_name
from gym.wrappers.monitoring import video_recorder

from stable_baselines.common.vec_env import VecVideoRecorder


def render_model(rule, env, video_folder, name_prefix, n_episodes, save_image=True):
    """
    Save a single video of the model in the environment over n episodes. If a
    video of the same name already exists in the target folder, append a unique
    suffix instead of overwriting.
    Inputs:
        rule,        class with a get_action() member function that gives the next action. TODO: this should maybe
                        be compatible with a tf model as well
        env,          vectorized environment, typically of size 1
        video_folder, directory for output
        name_prefix,  file name with no extension
        n_episodes,   number of episodes in the video
    Typically the vectorized environment will have just one environment. If it
    has multiple, the first entry will control the duration of the video.
    """
    if n_episodes > 0:
        #NOTE: only valid with a vector environment of length one
        video_env = CustomVecVideoRecoder(env,
                                          video_folder=video_folder,
                                          name_prefix=name_prefix)
        count = 0
        obs = video_env.reset()
        while count < n_episodes:
            #remove redundant obs dimensions from vectorized env (??)
            for key in obs:
                obs[key] = obs[key][0]
            action = rule.get_action(obs) # can replace with model.predict(obs) if using tf model
            obs, rewards, dones, info = video_env.step([action])
            if dones[0] and save_image:
                #import pdb; pdb.set_trace()
                ws_dir = Path(__file__).resolve().parents[0]
                im_dir = Path(video_folder)
                filepath, _ = clean_file_name('result.png', im_dir)
                mpimg.imsave(str(filepath), 1-video_env.get_attr('occupancy',0)[0], cmap="gray",origin='lower')
                count = count + 1
                print(f'Finished episode {count}.')
        video_env.close_video_recorder()

class CustomVecVideoRecoder(VecVideoRecorder):
    def __init__(self, venv, video_folder='/video', name_prefix='video'):
        """
        Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
        It requires ffmpeg or avconv to be installed on the machine.
        :param venv: (VecEnv or VecEnvWrapper)
        :param video_folder: (str) Where to save videos
        :param name_prefix: (str)  Video name without extension (will be made unique with incrementing suffix).
        Customizes StableBaselines VecVideoRecorder through inheritance for more
        granular control of output filenames.
        """
        super().__init__(venv,
                         video_folder=video_folder,
                         record_video_trigger=lambda x: x == 0,
                         video_length=np.inf,  # not used
                         name_prefix=name_prefix
                         )

        self.video_folder = Path(video_folder)

    def start_video_recorder(self):
        self.close_video_recorder()

        # If video name already exists, apply unique numbered suffix.
        path = self.video_folder / (self.name_prefix + '.mp4')
        if path.exists():
            for i in itertools.count():
                path = self.video_folder / (self.name_prefix + f'_{i}.mp4')
                if not path.exists():
                    break

        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            path=str(path),
            metadata={'step_id': self.step_id})

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True
