import os
import tempfile
import argparse
from understanding_rl_vision import rl_clarity
from understanding_rl_vision.rl_clarity.training import (
    PROCGEN_ENV_NAMES,
    ATARI_ENV_DICT,
)


def train_and_run(env_name_or_id, *, base_path=None):
    if base_path is None:
        base_path = tempfile.mkdtemp(prefix="rl_clarity_example_")
    training_dir = os.path.join(base_path, "training")
    interface_dir = os.path.join(base_path, "interface")
    if "://" not in base_path:
        os.makedirs(training_dir, exist_ok=True)
        os.makedirs(interface_dir, exist_ok=True)

    if env_name_or_id in PROCGEN_ENV_NAMES + ["coinrun_old"]:
        env_kwargs = {"env_name": env_name_or_id}
    elif env_name_or_id in ATARI_ENV_DICT:
        env_kwargs = {"env_id": env_name_or_id, "env_kind": "atari"}
    else:
        raise ValueError(f"Unsupported env {env_name_or_id}")

    # train for very few timesteps, to demonstrate
    # note: training code has not been well-tested
    rl_clarity.train(
        num_envs=8,
        num_steps=16,
        timesteps_per_proc=8 * 16 * 2,
        save_interval=2,
        save_dir=training_dir,
        **env_kwargs,
    )
    checkpoint_path = os.path.join(training_dir, "checkpoint.jd")
    print(f"Checkpoint saved to: {checkpoint_path}")

    print("Generating interface...")
    # generate a small interface, to demonstrate
    rl_clarity.run(
        checkpoint_path,
        output_dir=interface_dir,
        trajectories_kwargs={"num_envs": 8, "num_steps": 16},
        observations_kwargs={"num_envs": 8, "num_obs": 4, "obs_every": 4},
        layer_kwargs={"name_contains_one_of": ["2b"]},
    )

    interface_path = os.path.join(interface_dir, "interface.html")
    interface_url = ("" if "://" in interface_path else "file://") + interface_path
    print(f"Interface URL: {interface_url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", nargs='?', default="coinrun_old")
    parser.add_argument("-p", "--path")
    args = parser.parse_args()
    train_and_run(args.env, base_path=args.path)


if __name__ == "__main__":
    main()
