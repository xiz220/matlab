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

### Testing

To start a live rendering of the environment, run the `test.py` script. To record a video of the environment, run the `record_video.py` script.

### Environment

The environment is an occupancy-grid-based representation of a lattice, which must be loaded from an image (sample images included in the `images` directory). The environment is structured in an OpenAI Gym style, with a `step` function that applies a set of actions to the robots in the environment, and updates the environment, and a `render` function that paints the environment and the robots in it. 
