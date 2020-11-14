from .interface import generate
from .loading import load
from .compiling import recompile_js
from .training import train


def run(
    checkpoint_path,
    *,
    output_dir,
    load_kwargs={},
    trajectories_kwargs={},
    observations_kwargs={},
    **generate_kwargs
):
    """Generate an interface from a checkpoint file.

    Arguments:
        checkpoint_path: path to checkpoint file, a joblib file containing a
                         dictionary with these keys
          - params: saved model parameters as a dictionary mapping tensor names
                    to numpy arrays
          - args: dictionary of metadata with these keys
              - env_name:     name of the Procgen environment
                              required if env_kind is 'procgen'
              - env_id:       lowercase id of the Atari environment
                              required if env_kind is 'atari'
              - env_kind:     either 'procgen' or 'atari'
                              defaults to 'procgen'
              - gamma:        GAE hyperparameter gamma used to train the model
                              defaults to None
              - lambda:       GAE hyperparameter lambda used to train the model
                              defaults to None
              - cnn:          model architecture, one of 'clear', 'impala' or
                              'nature'
                              defaults to 'clear'
              - any other optional arguments used to create the environment or
                get the architecture
        output_dir:            path to directory where interface is to be saved
                               required
        load_kwargs: dictionary with keys for any of the following
          - resample:          whether to process the checkpoint file from
                               scratch, rather than reusing samples previously
                               saved to a non-temporary location
                               defaults to True
          - model_path:        lucid model save location
          - metadata_path:     metadata dictionary save location
          - trajectories_path: trajectories save location
          - observations_path: additional observations save location
          - full_resolution:   whether to also save observations in human-scale
                               resolution (significant performance cost)
                               defaults to False
          - temp_files:        if any of the above paths is not specified,
                               whether to default to a temporary location
                               rather than a sudirectory of the checkpoint
                               file's directory
                               defaults to False
        trajectories_kwargs: dictionary with keys for any of the following
                             only used if resampling
          - num_envs:  number of trajectories to collect
                       defaults to 8
          - num_steps: length of each trajectory
                       defaults to 512
        observations_kwargs: dictionary with keys for any of the following
                             only used if resampling
          - num_envs:  number of environments to collect additional
                       observations from in parallel
                       defaults to 32
          - num_obs:   number of additional observations to collect from
                       each parallel environment
                       defaults to 128
          - obs_every: number of steps to wait between each observation
                       defaults to 128
        model_bytes:          lucid model, represented as a save file's bytes
                              defaults to being extracted automatically
        observations:         numpy array of additional observations used for
                              feature visualization
                              defaults to being extracted automatically
        observations_full:    numpy array of the additional observations in
                              human-scale resolution, or None to only use
                              observations at the resolution seen by the model
                              defaults to being extracted automatically, or None
                              if human-scale resolution observations were not
                              saved
        trajectories:         dictionary of trajectories with keys
                              'observations', 'actions', 'rewards', either
                              'firsts' or 'dones', and optionally
                              'observations_full', each value being a numpy
                              array with first two dimensions batch and timestep
                              defaults to being extracted automatically
        policy_logits_name:   name of tensor of policy logits
                              defaults to being extracted automatically
        value_function_name:  name of tensor of value function
                              defaults to being extracted automatically
        env_name:             Procgen environment name, used to help infer
                              action_combos if that is not provided
                              defaults to being extracted automatically, or
                              'unknown' if that fails
        numpy_precision:      number of significant figures to round numpy
                              arrays in the HTML file to
                              defaults to 6
        inline_js:            whether to include the JavaScript in the HTML file
                              inline, rather than referencing a separate file
                              defaults to True (to avoid ad-blocker issues)
        inline_large_json:    whether to include large amounts of JSON data in
                              the HTML file inline, rather than referencing
                              separate files
                              defaults to whether output_dir does not contain
                              '://'
        batch_size:           size of minibatch of observations to pass through
                              model
                              defaults to 512
        action_combos:        list of tuples of strings describing the
                              combinations of buttons triggered by each action
                              defaults to being extracted automatically, or
                              [('0',), ..., ('<num_actions - 1>',)] if that fails
        action_group_fns:     list of function filters for grouping the action
                              combos in different ways
                              defaults to [
                                  lambda combo: 'RIGHT' in combo,
                                  lambda combo: 'LEFT' in combo,
                                  lambda combo: 'UP' in combo,
                                  lambda combo: 'DOWN' in combo,
                                  lambda combo: 'RIGHT' not in combo
                                                 and 'LEFT' not in combo
                                                 and 'UP' not in combo
                                                 and 'DOWN' not in combo
                              ]
        layer_kwargs: dictionary of options for choosing layers, with keys for
                      any of the following
          - name_contains_one_of: list of strings each layer name must contain
                                  one of, or None to not filter by name
                                  defaults to None
          - op_is_one_of:         list of strings each layer op must be one of
                                  defaults to ['relu']
          - bottleneck_only:      whether to only include layers such that every
                                  path to an earlier convolutional layer passes
                                  through a bottleneck of the network
                                  defaults to True
          - discard_first_n:      number of first layers to discard
                                  defaults to 0
        input_layer_include:  whether to additionally calcuate gradients with
                              respect to the input layer
                              defaults to False
        input_layer_name:     display name of the input layer
                              defaults to 'input'
        gae_gamma:            gamma for computing advantages using GAE
                              defaults to being extracted automatically, or
                              0.999 if that fails
        gae_lambda:           lambda for computing advantages using GAE
                              defaults to being extracted automatically, or
                              0.95 if that fails
        trajectory_bookmarks: number of links to display to highest advantage
                              episodes and to lowest advantage episodes
                              defaults to 16
        nmf_features:         number of dimensions for NMF dimensionality
                              reduction
                              defaults to 8
        nmf_attr_opts:        dictionary of options for computing attribution
                              for NMF dimensionality reduction, the main one
                              being integrate_steps (explained below, see
                              attr_integrate_steps)
                              defaults to {'integrate_steps': 10}, though if a
                              dictionary is provided without an
                              'integrate_steps' key, then integrate_steps
                              defaults to 1
        vis_subdiv_mults:     list of values of subdiv_mult, the spatial
                              resolution of the grid of dataset examples used
                              for feature visualization, as a mulitple of the
                              resolution of the layer's activations
                              defaults to [0.25, 0.5, 1, 2]
        vis_subdiv_mult_default: default value of subdiv_mult (explained above)
                              defaults to 1
        vis_expand_mults:     list of values of expand_mult, the height and
                              width of each patch used for feature
                              visualization, as a multiple of the number of
                              pixels if the layer were overlaid on the
                              observation
                              defaults to [1, 2, 4, 8]
        vis_expand_mult_default: default value of expand_mult (explained above)
                              defaults to 4
        vis_thumbnail_num_mult: spatial resolution of the grid of dataset
                              examples used for feature visualization thumbnails
                              defaults to 4
        vis_thumbnail_expand_mult: the height and width of each patch used for
                              feature visualization thumbnails, as a multiple of
                              the number of pixels if the layer were overlaid on
                              the observation
                              defaults to 4
        scrub_range:          horizonal interval of observations and attribution
                              used to construct scrubs
                              defaults to (42 / 64, 44 / 64)
        attr_integrate_steps: number of points on the path used for numerical
                              integration for computing attribution
                              defaults to 10
        attr_max_paths:       maximum number of paths for multi-path
                              attribution, or None to use single-path
                              attribution
                              defaults to None
        attr_policy:          whether to compute attribution for the policy
                              defaults to False
        attr_single_channels: whether to allow attribution for single channels
                              to be displayed
                              defaults to True
        observations_subdir:  name of subdirectory containing additional
                              observations
                              defaults to 'observations/'
        trajectories_subdir:  name of subdirectory containing trajectory
                              observations
                              defaults to 'trajectories/'
        trajectories_scrub_subdir: name of subdirectory containing scrubs of
                              trajectory observations
                              defaults to 'trajectories_scrub/'
        features_subdir:      name of subdirectory containing feature
                              visualizations
                              defaults to 'features/'
        thumbnails_subdir:    name of subdirectory containing feature thumbnails
                              defaults to 'thumbnails/'
        attribution_subdir:   name of subdirectory containing attribution
                              defaults to 'attribution/'
        attribution_scrub_subdir: name of subdirectory containing scrubs of
                              attribution
                              defaults to 'attribution_scrub/'
        video_height:         css height of each video screen
                              defaults to '16em'
        video_width:          css width of each video screen
                              defaults to '16em'
        video_speed:          speed of vidoes in frames per second
                              defaults to 12
        policy_display_height: css height of bar displaying policy
                              defaults to '2em'
        policy_display_width: css width of bar displaying policy
                              defaults to '40em'
        navigator_width:      css width of navigator bar
                              defaults to '24em'
        scrubber_height:      css height of each scrubber
                              defaults to '4em'
        scrubber_width:       css width of each scrubber
                              defaults to '48em'
        scrubber_visible_duration: number of frames visible in each scrubber
                              defaults to 256
        legend_item_height:   css height of each legend item
                              defaults to '6em'
        legend_item_width:    css width of each legend item
                              defaults to '6em'
        feature_viewer_height: css height of feature visualizations in the popup
                              defaults to '40em'
        feature_viewer_width: css width of feature visualizations in the popup
                              defaults to '40em'
        attribution_weight:   css opacity of attribution when overlaid on
                              observations (taking into account the fact that
                              attribution is mostly transparent)
                              defaults to 0.9
        graph_colors:         dictionary specifying css colors of graphs of each
                              type
                              defaults to {
                                  'v': 'green',
                                  'action': 'red',
                                  'action_group': 'orange',
                                  'advantage': 'blue'
                              }
        trajectory_color:     css color of text displaying trajectory
                              information such as actions and rewards
                              defaults to 'blue'
    """
    import tensorflow as tf
    from mpi4py import MPI
    from baselines.common.mpi_util import setup_mpi_gpus

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    setup_mpi_gpus()

    exn = None
    if rank == 0 and load_kwargs.get("resample", True):
        kwargs = load(
            checkpoint_path,
            trajectories_kwargs=trajectories_kwargs,
            observations_kwargs=observations_kwargs,
            **load_kwargs
        )
        comm.barrier()
    else:
        comm.barrier()
        load_kwargs["resample"] = False
        try:
            kwargs = load(
                checkpoint_path,
                trajectories_kwargs=trajectories_kwargs,
                observations_kwargs=observations_kwargs,
                **load_kwargs
            )
        except tf.errors.NotFoundError as e:
            exn = e
            kwargs = None
    errors = comm.allreduce(0 if exn is None else 1, op=MPI.SUM)
    if errors == size:
        raise FileNotFoundError from exn
    elif errors > 0:
        kwargs = comm.bcast(kwargs, root=0)
    kwargs["output_dir"] = output_dir
    kwargs.update(generate_kwargs)

    generate(**kwargs)
