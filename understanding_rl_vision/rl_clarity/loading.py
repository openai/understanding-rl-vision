import numpy as np
import tensorflow as tf
from contextlib import contextmanager
import os
import re
import tempfile
from lucid.modelzoo.vision_base import Model
from lucid.misc.io.reading import read
from lucid.scratch.rl_util.joblib_wrapper import load_joblib, save_joblib
from .training import create_env, get_arch


def load_params(params, *, sess):
    var_list = tf.global_variables()
    for name, var_value in params.items():
        matching_vars = [var for var in var_list if var.name == name]
        if matching_vars:
            matching_vars[0].load(var_value, sess)


def save_lucid_model(config, params, *, model_path, metadata_path):
    config = config.copy()
    config.pop("num_envs")
    library = config.get("library", "baselines")
    venv = create_env(1, **config)
    arch = get_arch(**config)

    with tf.Graph().as_default(), tf.Session() as sess:
        observation_space = venv.observation_space
        observations_placeholder = tf.placeholder(
            shape=(None,) + observation_space.shape, dtype=tf.float32
        )

        if library == "baselines":
            from baselines.common.policies import build_policy

            with tf.variable_scope("ppo2_model", reuse=tf.AUTO_REUSE):
                policy_fn = build_policy(venv, arch)
                policy = policy_fn(
                    nbatch=None,
                    nsteps=1,
                    sess=sess,
                    observ_placeholder=(observations_placeholder * 255),
                )
                pd = policy.pd
                vf = policy.vf

        else:
            raise ValueError(f"Unsupported library: {library}")

        load_params(params, sess=sess)

        Model.save(
            model_path,
            input_name=observations_placeholder.op.name,
            output_names=[pd.logits.op.name, vf.op.name],
            image_shape=observation_space.shape,
            image_value_range=[0.0, 1.0],
        )

    metadata = {
        "policy_logits_name": pd.logits.op.name,
        "value_function_name": vf.op.name,
        "env_name": config.get("env_name"),
        "gae_gamma": config.get("gamma"),
        "gae_lambda": config.get("lambda"),
    }
    env = venv
    while hasattr(env, "env") and (not hasattr(env, "combos")):
        env = env.env
    if hasattr(env, "combos"):
        metadata["action_combos"] = env.combos
    else:
        metadata["action_combos"] = None

    save_joblib(metadata, metadata_path)
    return {"model_bytes": read(model_path, cache=False, mode="rb"), **metadata}


@contextmanager
def get_step_fn(config, params, *, num_envs, full_resolution):
    config = config.copy()
    config.pop("num_envs")
    library = config.get("library", "baselines")
    venv = create_env(num_envs, **config)
    arch = get_arch(**config)

    with tf.Graph().as_default(), tf.Session() as sess:
        if library == "baselines":
            from baselines.common.policies import build_policy

            with tf.variable_scope("ppo2_model", reuse=tf.AUTO_REUSE):
                policy_fn = build_policy(venv, arch)
                policy = policy_fn(nbatch=venv.num_envs, nsteps=1, sess=sess)

            stepdata = {
                "ob": venv.reset(),
                "state": policy.initial_state,
                "first": np.ones((venv.num_envs,), bool),
            }
            if full_resolution:
                stepdata["ob_full"] = np.stack(
                    [info["rgb"] for info in venv.env.get_info()], axis=0
                )

            def step_fn():
                result = {"ob": stepdata["ob"], "first": stepdata["first"].astype(bool)}
                if full_resolution:
                    result["ob_full"] = stepdata["ob_full"]
                result["ac"], _, stepdata["state"], _ = policy.step(
                    stepdata["ob"],
                    S=stepdata["state"],
                    M=stepdata["first"].astype(float),
                )
                (
                    stepdata["ob"],
                    result["reward"],
                    stepdata["first"],
                    result["info"],
                ) = venv.step(result["ac"])
                if full_resolution:
                    stepdata["ob_full"] = np.stack(
                        [info["rgb"] for info in result["info"]], axis=0
                    )
                return result

        else:
            raise ValueError(f"Unsupported library: {library}")

        load_params(params, sess=sess)

        yield step_fn


def save_observations(
    config, params, *, observations_path, num_envs, num_obs, obs_every, full_resolution
):
    with get_step_fn(
        config, params, num_envs=num_envs, full_resolution=full_resolution
    ) as step_fn:
        observations = []
        if full_resolution:
            observations_full = []
        for _ in range(num_obs):
            for _ in range(obs_every):
                step_result = step_fn()
            observations.append(step_result["ob"])
            if full_resolution:
                observations_full.append(step_result["ob_full"])
        observations = np.concatenate(observations, axis=0)
        if full_resolution:
            observations_full = np.concatenate(observations_full, axis=0)

    result = {"observations": observations}
    if full_resolution:
        result["observations_full"] = observations_full
    save_joblib(result, observations_path)
    return result


def save_trajectories(
    config, params, *, trajectories_path, num_envs, num_steps, full_resolution
):
    with get_step_fn(
        config, params, num_envs=num_envs, full_resolution=full_resolution
    ) as step_fn:
        step_fn()
        trajectories = [step_fn() for _ in range(num_steps)]
        get_and_stack = lambda ds, key, axis=1: np.stack(
            [d[key] for d in ds], axis=axis
        )
        result = {
            "observations": get_and_stack(trajectories, "ob"),
            "actions": get_and_stack(trajectories, "ac"),
            "rewards": get_and_stack(trajectories, "reward"),
            "firsts": get_and_stack(trajectories, "first"),
        }
        if full_resolution:
            result["observations_full"] = get_and_stack(trajectories, "ob_full")

    save_joblib(result, trajectories_path)
    return {"trajectories": result}


def load(
    checkpoint_path,
    *,
    resample=True,
    model_path=None,
    metadata_path=None,
    trajectories_path=None,
    observations_path=None,
    trajectories_kwargs={},
    observations_kwargs={},
    full_resolution=False,
    temp_files=False,
):
    if temp_files:
        default_path = lambda suffix: tempfile.mkstemp(suffix=suffix)[1]
    else:
        path_stem = re.split(r"(?<=[^/])\.[^/\.]*$", checkpoint_path)[0]
        path_stem = os.path.join(
            os.path.dirname(path_stem), "rl-clarity", os.path.basename(path_stem)
        )
        default_path = lambda suffix: path_stem + suffix
    if model_path is None:
        model_path = default_path(".model.pb")
    if metadata_path is None:
        metadata_path = default_path(".metadata.jd")
    if trajectories_path is None:
        trajectories_path = default_path(".trajectories.jd")
    if observations_path is None:
        observations_path = default_path(".observations.jd")

    if resample:
        trajectories_kwargs.setdefault("num_envs", 8)
        trajectories_kwargs.setdefault("num_steps", 512)
        observations_kwargs.setdefault("num_envs", 32)
        observations_kwargs.setdefault("num_obs", 128)
        observations_kwargs.setdefault("obs_every", 128)

        checkpoint_dict = load_joblib(checkpoint_path, cache=False)
        config = checkpoint_dict["args"]
        if full_resolution:
            config["render_human"] = True
        if config.get("use_lstm", 0):
            raise ValueError("Recurrent networks not yet supported by this interface.")
        params = checkpoint_dict["params"]
        config["coinrun_old_extra_actions"] = 0
        if config.get("env_name") == "coinrun_old":
            # we may need to add extra actions depending on the size of the policy head
            policy_bias_keys = [
                k for k in checkpoint_dict["params"] if k.endswith("pi/b:0")
            ]
            if policy_bias_keys:
                [policy_bias_key] = policy_bias_keys
                (num_actions,) = checkpoint_dict["params"][policy_bias_key].shape
                if num_actions == 9:
                    config["coinrun_old_extra_actions"] = 2

        return {
            **save_lucid_model(
                config, params, model_path=model_path, metadata_path=metadata_path
            ),
            **save_observations(
                config,
                params,
                observations_path=observations_path,
                num_envs=observations_kwargs["num_envs"],
                num_obs=observations_kwargs["num_obs"],
                obs_every=observations_kwargs["obs_every"],
                full_resolution=full_resolution,
            ),
            **save_trajectories(
                config,
                params,
                trajectories_path=trajectories_path,
                num_envs=trajectories_kwargs["num_envs"],
                num_steps=trajectories_kwargs["num_steps"],
                full_resolution=full_resolution,
            ),
        }

    else:
        observations = load_joblib(observations_path, cache=False)
        if not isinstance(observations, dict):
            observations = {"observations": observations}
        return {
            "model_bytes": read(model_path, cache=False, mode="rb"),
            **observations,
            "trajectories": load_joblib(trajectories_path, cache=False),
            **load_joblib(metadata_path, cache=False),
        }
