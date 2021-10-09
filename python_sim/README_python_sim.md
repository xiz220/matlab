# Simulator for occupancy-grid based lattice construction environment

### Python Environment

It is recommended that you install via `conda`. If you are on Windows, from this directory, type:

```
conda env create --name lattice-robots -f environment_windows.yml
conda activate lattice-robots
```

If you are on Linux, use `environment.yml` instead of `environment_windows.yml`.

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
save a video recording, and save a final image to the experiment folder.

To save time, you can skip recording the video -- this results in roughly 30x speedup of the experiment. To do this, use the `-no_record` flag:

```python run_experiment.py -exp_name your_exp_name_here -cfg path_to_cfg_file.toml -no_record```

Example:
```python run_experiment.py -exp_name test_experiment -cfg cfg/test.toml```

The parameters of the experiment are defined completely in the configuration file. You can create your own, or modify an existing one in the `cfg` folder. These files specify the general environment parameters, such as how many agents, how large the environment, and which image to use as the base structure on top of which the robots build. They also specify the local rules the robots use, and the parameters of those rules, such as beam angle, beam thickness, and many other parameters, specific to each rule. 


### Environment

The environment is an occupancy-grid-based representation of a lattice, which must be loaded from an image (sample images included in the `images` directory). The environment is structured in an OpenAI Gym style, with a `step` function that applies a set of actions to the robots in the environment, and updates the environment, and a `render` function that paints the environment and the robots in it. 


### Rule Sets
Sets of local rules that control the robots are stored in the `controllers` folder. There are two types of rule files: "rules," which end in `_rule.py`, and combine multiple motion primitives into a complex rule that is executed through an episode; and "controllers," which are the motion primitives (such as "wall follow" or "extrude beam at angle") that constitute the rules themselves. "Controllers" end in `_controller.py` and cannot be run on their own; they must be wrapped into a `rule`.

The `rules` are essentially a big finite state machine that tells agents when to follow which set of instructions -- e.g. when to wall follow, and when to transition from wall following to beam extrusion. This makes the `rule` files a good higher-level entry point for messing with the local rules! They also tell you exactly which parameters can be specified for the overarching rule, and you can look at the `controller` files to see which parameters can be further specified for those, if the defaults are not sufficient.  

### Installing matlab via python API

Installation instructions can be found [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
With further details about installing in non-default folders [here](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html)

This is a little hacky, but what worked for me was to navigate to the root matlab directory, cd into `extern/engines/python`, and run

```python setup.py install --prefix="lattice-venv-dir"```
Where `lattice-venv-dir` is the directory for the conda environment. Navigate to `lattice-venv-dir` and movedpy the matlab engine installation from a subfolder called `lib.linux-x86_64-2.7` (it will probably be different on your machine) into the main `lattice-venv-dir` by running the following from the main `lattice-venv-dir`:
```cp -r lib.linux-x86_64-2.7/matlab .```
