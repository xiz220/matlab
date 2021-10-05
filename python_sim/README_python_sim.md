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
For mechanical property calculation (stiffness via direct stiffness method) matlab is also required, via the python API. See a later section for installation instructions.

### Running Experiments

To start a live rendering of the environment, run `run_experiment.py` in testing mode:

```python run_experiment.py -exp_name your_exp_name_here -cfg path_to_cfg_file -test```

To record a video of the environment, run `run_experiment.py` without the testing flag:

```python run_experiment.py -exp_name your_exp_name_here -cfg path_to_cfg_file.toml ```

This will create a new folder in the `experiments` directory with the name `your_exp_name_here_#` with an integer added 
to the end to differentiate it. It will run the experiment specified in the parameters toml at `path_to_cfg_file.toml`,
save a video recording and a final image to the experiment folder.

Example:
```python run_experiment.py -exp_name test_experiment -cfg cfg/test.toml```


### Environment

The environment is an occupancy-grid-based representation of a lattice, which must be loaded from an image (sample images included in the `images` directory). The environment is structured in an OpenAI Gym style, with a `step` function that applies a set of actions to the robots in the environment, and updates the environment, and a `render` function that paints the environment and the robots in it. 


### Installing matlab via python API

Installation instructions can be found [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
With further details about installing in non-default folders [here](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html)

This is a little hacky, but what worked for me was to navigate to the root matlab directory, cd into `extern/engines/python`, and run

```python setup.py install --prefix="lattice-venv-dir"```
Where `lattice-venv-dir` is the directory for the conda environment. Navigate to `lattice-venv-dir` and movedpy the matlab engine installation from a subfolder called `lib.linux-x86_64-2.7` (it will probably be different on your machine) into the main `lattice-venv-dir` by running the following from the main `lattice-venv-dir`:
```cp -r lib.linux-x86_64-2.7/matlab .```