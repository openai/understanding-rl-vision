**Status:** Archive (code is provided as-is, no updates expected)

# Understanding RL Vision

#### [ [Paper] ](https://distill.pub/2020/understanding-rl-vision) [ [Demo] ](https://openaipublic.blob.core.windows.net/rl-clarity/attribution/demo/interface.html)

Generate interfaces for interpreting vision models trained using RL.

The core utilities used to compute feature visualization, attribution and dimensionality reduction can be found in `lucid.scratch.rl_util`, a submodule of [Lucid](https://github.com/tensorflow/lucid/). These are demonstrated in [this notebook](https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/misc/rl_util.ipynb). The code here leverages these utilities to build HTML interfaces similar to the above demo.

![](https://openaipublic.blob.core.windows.net/rl-clarity/attribution/demo.gif)

## Installation

Supported platforms: MacOS and Ubuntu, Python 3.7, TensorFlow <= 1.14

- Install [Baselines](https://github.com/openai/baselines) and its dependencies, including TensorFlow 1.
- Clone the repo:
    ```
    git clone https://github.com/openai/understanding-rl-vision.git
    ```
- Install the repo and its dependencies, among which is a pinned version of [Lucid](https://github.com/tensorflow/lucid):
    ```
    pip install -e understanding-rl-vision
    ```
- Install an RL environment of your choice. Supported environments:
    - [CoinRun](https://github.com/openai/coinrun) (the original version used in the paper): follow the instructions. Note: due to CoinRun's requirements, you should re-install Baselines after installing CoinRun.
    - [Procgen](https://github.com/openai/procgen): `pip install procgen`
    - [Atari](https://github.com/openai/atari-py): `pip install atari-py`

## Generating interfaces

The main script processes checkpoint files saved by RL code:
```
from understanding_rl_vision import rl_clarity

rl_clarity.run('path/to/checkpoint/file', output_dir='path/to/directory')
```

An example checkpoint file can be downloaded [here](https://openaipublic.blob.core.windows.net/rl-clarity/attribution/models/coinrun.jd), or can be generated using the [example script](understanding_rl_vision/rl_clarity/example.py). Checkpoint files for a number of pre-trained models are indexed [here](https://openaipublic.blob.core.windows.net/rl-clarity/attribution/models/index.html).

The precise format required of the checkpoint file, along with a full list of keyword arguments, can be found in the function's [docstring](understanding_rl_vision/rl_clarity/__init__.py).

The script will create an `interface.html` file, along with directories containing images (which can take up several GB), at the location specified by `output_dir`.

By default, the script will also create some files in the directory of the checkpoint file, in an `rl-clarity` subdirectory. These contain all the necessary information extracted from the model and environment for re-creating the same interface. To create these files in a temporary location instead, set `load_kwargs={'temp_files': True}`. To re-create an interface using existing files, set `load_kwargs={'resample': False}`.

### Speed issues

The slowest part of the script is computing the attribution in all the required combinations. If you set `trajectories_kwargs={'num_envs': num_envs, 'num_steps': num_steps}`, then `num_envs` trajectories will be collected, each of length `num_steps`, and the script will distribute the trajectories among the MPI workers for computing the attribution. The memory requirements of each worker scales with `num_steps`, which defaults to 512 (about as large as a machine with 34 GB of memory can typically handle). The default `num_envs` is 8, so it is best to use 8 MPI workers by default to save time, if you have 8 GPUs available.

The script should take a few hours to run, but if it is taking too long, then you can tell the script to ignore the first couple of non-input layers by setting `layer_kwargs={'discard_first_n': 2}`, for example. These layers take the longest to compute attribution for since they have the highest spatial resolution, and are usually not that informative anyway.

By default, attribution is only computed for the value function, since computing attribution for every logit of the policy amounts to a large multiplier on the time taken by the script to run. To compute attribution for the policy, set `attr_policy=True`. To offset the increased computational load when doing this, you may wish to choose a single layer to compute attribution for by setting `layer_kwargs={'name_contains_one_of': ['2b']}`, for example.

To save disk space, the hover effect for isolating single attribution channels can be disabled by setting `attr_single_channels=False`, though this will not have much effect on speed.

## Guide to interfaces

As shown in [this demo](https://openaipublic.blob.core.windows.net/rl-clarity/attribution/demo/interface.html), interfaces are divided into a number of sections:

- **Trajectories** - Each trajectory is a separate rollout of the agent interacting with the environment. Here you can select one of them.
- **Bookmarks** - Advantages have been computed using [generalized advantage estimation](https://arxiv.org/abs/1506.02438) (GAE). These provide a measure of how successful each choice made by the agent turned out relative to its expectations, and would usually be used to improve the agent's policy during training. The links here allow you to skip to specific frames from the trajectories with the highest and lowest advantages (with at most one link per episode).
- **Layers** - Here you can select a layer for which attribution (explained below) has been computed. For the input layer, if included, attribution makes less sense, so gradients have been computed instead.
- **Timeline** - Here you can navigate through the frames in each trajectory, either using the buttons or by scrubbing. At the top, information about the current frame is displayed, including the last reward received, the agent's policy, and the action that was chosen next. There are graphs of advantages (as used by the Bookmarks section) and of each network output that has been selected in the Attribution section.
- **Attribution** - Here you can view the observations processed by the agent, and attribution from network outputs (just the value function by default) to the selected layer. Below the observation is chart of the attribution summed over spatial positions. If attribution has been computed for the policy, you can add and remove rows from this section, and select a different network output for each row, such as the value function, or the policy's logit for a particular action. Attribution has been computed using the method of [integrated gradients](https://arxiv.org/abs/1703.01365): the gradient of the network output with respect to selected layer has been numerically integrated along the straight line from zero to the layer's output given the current observation. This effectively decomposes (or "attributes") the network output across the spatial positions and channels of the selected layer. Dimensionality reduction ([non-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)) has been applied to the channels using a large batch of varied observations, and the resulting channels are represented using different colors. Additional normalization and smoothing has been applied, with strong attribution bleeding into nearby spatial positions.
- **Attribution legend** - For each of the channels produced by dimensionality reduction (explained above), there are small visualizations here of the feature picked out by that channel. These consist of patches taken from observations at the spatial positions where the selected layer was most strongly activated in the direction of the channel. Hovering over these isolates the channel for the displayed attribution, and clicking opens a the Feature visualization popup, where the feature can be further analyzed.
- **Feature visualization** (in popup) - This is displayed after a feature from the Attribution legend section has been selected, and shows a larger visualization of the feature. This also consists of patches taken from observations where the selected layer was most strongly activated in the appropriate direction, but here the location of a patch determines a specific spatial position that must be activated. This means that there is a spatial correspondence between the visualization and observations. Patches with weaker activations are displayed with greater transparency, except when hovering over the image. There are sliders that can be used to set the zoom level of the patches (which can also be controlled by scrolling over the image) and the number of patches (which initially equals the number of spatial positions of the selected layer). Clicking on a patch reveals the full observation from which the patch was extracted.
- **Hotkeys** - Here is a list of available keyboard shortcuts. Toggling between play and pause also toggles between whether the arrow keys change the play direction or take a single step in one direction.

## Training models

There is also a script for training a model using [PPO2](https://github.com/openai/baselines/tree/master/baselines/ppo2) from [Baselines](https://github.com/openai/baselines), and saving a checkpoint file in the required format:
```
from understanding_rl_vision import rl_clarity

rl_clarity.train(env_name='coinrun_old', save_dir='path/to/directory')
```

This script is intended to explain checkpoint files, and has not been well-tested. The [example script](understanding_rl_vision/rl_clarity/example.py) demonstrates how to train a model and then generate an interface for it.

## Svelte compilation

To generate interfaces, the Svelte source must be compiled to JavaScript. At installation, the module will automatically attempt to download the pre-compiled JavaScript from a remote copy, though this copy is not guaranteed to be kept up-to-date.

To obtain an up-to-date copy, or for development, you may wish to re-compile the JavaScript locally. To do this, first install [Node.js](https://nodejs.org/) if you have not already. On Mac:
```
brew install node
```
You will then be able to re-compile the JavaScript:
```
python -c 'from understanding_rl_vision import rl_clarity; rl_clarity.recompile_js()'
```

### Standalone compiler

The `svelte3` package provides generic functions for compiling version 3 of Svelte to JavaScript or HTML. These can be used to create an easy-to-use command-line tool:
```
python -c 'from understanding_rl_vision import svelte3; svelte3.compile_html("path/to/svelte/file", "path/to/html/file")'
```

Detailed usage instructions can be found in the functions' [docstrings](svelte3/compiling.py).

## Citation

Please cite using the following BibTeX entry:
```
@article{hilton2020understanding,
  author = {Hilton, Jacob and Cammarata, Nick and Carter, Shan and Goh, Gabriel and Olah, Chris},
  title = {Understanding RL Vision},
  journal = {Distill},
  year = {2020},
  note = {https://distill.pub/2020/understanding-rl-vision},
  doi = {10.23915/distill.00029}
}
```