# Simulator for occupancy-grid based lattice construction environment

### Python Environment

It is recommended that you install via `conda`. From this directory, type:

```
conda env create --name lattice-robots -f environment.yml
conda activate lattice-robots
```

You can also install manually. This repository requires python 3 and the following packages:
* numpy
* scipy
* stable_baselines version 2.10.1
* toml
* gym

### Running Experiments

To start a live rendering of the environment, run `run_experiment.py` in testing mode:
```run_experiment.py -exp_name your_exp_name_here -cfg path_to_cfg_file -test```

To record a video of the environment, run `run_experiment.py` without the testing flag:
```run_experiment.py -exp_name your_exp_name_here -cfg path_to_cfg_file.toml ```

This will create a new folder in the `experiments` directory with the name `your_exp_name_here_#` with an integer added 
to the end to differentiate it. It will run the experiment specified in the parameters toml at `path_to_cfg_file.toml`,
save a video recording and a final image to the experiment folder.

Example:
```run_experiment.py -exp_name test_experiment -cfg cfg/test.toml```


### Environment

The environment is an occupancy-grid-based representation of a lattice, which must be loaded from an image (sample images included in the `images` directory). The environment is structured in an OpenAI Gym style, with a `step` function that applies a set of actions to the robots in the environment, and updates the environment, and a `render` function that paints the environment and the robots in it. 
