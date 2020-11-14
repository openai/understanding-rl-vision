import numpy as np
import os
import urllib
import subprocess
import tempfile
from collections import OrderedDict
from lucid.modelzoo.vision_base import Model
from lucid.misc.io import save
from lucid.misc.io.writing import write_handle
from lucid.scratch.rl_util.nmf import LayerNMF, rescale_opacity
from lucid.scratch.rl_util.attribution import (
    get_acts,
    default_score_fn,
    get_grad,
    get_attr,
    get_multi_path_attr,
)
from lucid.scratch.rl_util.util import (
    get_shape,
    concatenate_horizontally,
    channels_to_rgb,
    conv2d,
    norm_filter,
    brightness_to_opacity,
)
from .compiling import get_compiled_js
from ..svelte3 import compile_html


def get_model(model_bytes):
    model_fd, model_path = tempfile.mkstemp(suffix=".model.pb")
    with open(model_fd, "wb") as model_file:
        model_file.write(model_bytes)
    return Model.load(model_path)


def exists_path(model, *, from_names, to_names, without_names=[]):
    for from_name in from_names:
        if from_name not in without_names:
            if from_name in to_names:
                return True
            next_names = [
                name.rsplit(":", 1)[0]
                for node in model.graph_def.node
                if node.name == from_name
                for name in node.input
            ]
            if exists_path(
                model,
                from_names=next_names,
                to_names=to_names,
                without_names=without_names,
            ):
                return True
    return False


def longest_common_prefix(l):
    l = set([s[: min(map(len, l))] for s in l])
    while len(l) > 1:
        l = set([s[:-1] for s in l])
    return list(l)[0]


def longest_common_suffix(l):
    l = set([s[-min(map(len, l)) :] for s in l])
    while len(l) > 1:
        l = set([s[1:] for s in l])
    return list(l)[0]


def get_abbreviator(names):
    if len(names) <= 1:
        return slice(None, None)
    prefix = longest_common_prefix(names)
    prefix = prefix.rsplit("/", 1)[0] + "/" if "/" in prefix else ""
    suffix = longest_common_suffix(names)
    suffix = "/" + suffix.split("/", 1)[-1] if "/" in suffix else ""
    return slice(len(prefix), None if len(suffix) == 0 else -len(suffix))


def get_layer_names(
    model,
    output_names,
    *,
    name_contains_one_of,
    op_is_one_of,
    bottleneck_only,
    discard_first_n,
):
    if isinstance(name_contains_one_of, str):
        name_contains_one_of = [name_contains_one_of]
    if isinstance(op_is_one_of, str):
        name_contains_one_of = [op_is_one_of]

    nodes = model.graph_def.node

    shape_condition = lambda node: len(get_shape(model, node.name)) >= 4
    op_condition = lambda node: any(node.op.lower() == s.lower() for s in op_is_one_of)
    if bottleneck_only:
        bottleneck_names = [
            node.name
            for node in nodes
            if not exists_path(
                model,
                from_names=output_names,
                to_names=[model.input_name],
                without_names=[node.name],
            )
        ]
        conv_names = [node.name for node in nodes if node.op.lower()[:4] == "conv"]
        bottleneck_condition = lambda node: not exists_path(
            model,
            from_names=[node.name],
            to_names=conv_names,
            without_names=bottleneck_names,
        )
    else:
        bottleneck_condition = lambda node: True

    layer_names = [
        node.name
        for node in nodes
        if shape_condition(node) and op_condition(node) and bottleneck_condition(node)
    ]
    abbreviator = get_abbreviator(layer_names)

    if name_contains_one_of is None:
        name_condition = lambda name: True
    else:
        name_condition = lambda name: any(s in name for s in name_contains_one_of)

    return OrderedDict(
        [(name[abbreviator], name) for name in layer_names if name_condition(name)][
            discard_first_n:
        ]
    )


def batched_get(data, batch_size, process_minibatch):
    n_points = data.shape[0]
    n_minibatches = -((-n_points) // batch_size)
    return np.concatenate(
        [
            process_minibatch(data[i * batch_size : (i + 1) * batch_size])
            for i in range(n_minibatches)
        ],
        axis=0,
    )


def compute_gae(trajectories, *, gae_gamma, gae_lambda):
    values = trajectories["values"]
    next_values = values[:, 1:]
    rewards = trajectories["rewards"][:, :-1]
    try:
        dones = trajectories["dones"][:, :-1]
    except KeyError:
        dones = trajectories["firsts"][:, 1:]
    assert next_values.shape == rewards.shape == dones.shape
    deltas = rewards + (1 - dones) * gae_gamma * next_values - values[:, :-1]
    result = np.zeros(values.shape, values.dtype)
    for step in reversed(range(values.shape[1] - 1)):
        result[:, step] = (
            deltas[:, step]
            + (1 - dones[:, step]) * gae_gamma * gae_lambda * result[:, step + 1]
        )
    return result


def get_bookmarks(trajectories, *, sign, num):
    advantages = trajectories["advantages"]
    dones = trajectories["dones"].copy()
    dones[:, -1] = np.ones_like(dones[:, -1])
    high_scores_and_coords = []
    for trajectory in range(advantages.shape[0]):
        high_score = 0
        high_score_coords = None
        for step in range(advantages.shape[1]):
            score = advantages[trajectory][step] * sign
            if score > high_score:
                high_score = score
                high_score_coords = (trajectory, step)
            if dones[trajectory][step] and high_score_coords is not None:
                high_scores_and_coords.append((high_score, high_score_coords))
                high_score = 0
                high_score_coords = None
    high_scores_and_coords.sort(key=lambda x: -x[0])
    return list(map(lambda x: x[1], high_scores_and_coords[:num]))


def number_to_string(x):
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return "".join([c for c in s if c.isdigit() or c == "e"])


def get_html_colors(n, grayscale=False, mix_with=None, mix_weight=0.5, **kwargs):
    if grayscale:
        colors = np.linspace(0, 1, n)[..., None].repeat(3, axis=1)
    else:
        colors = channels_to_rgb(np.eye(n), **kwargs)
        colors = colors / colors.max(axis=-1, keepdims=True)
    if mix_with is not None:
        colors = colors * (1 - mix_weight) + mix_with[None] * mix_weight
    colors = np.round(colors * 255)
    colors = np.vectorize(lambda x: hex(x)[2:].zfill(2))(colors.astype(int))
    return ["#" + "".join(color) for color in colors]


def removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def generate(
    *,
    output_dir,
    model_bytes,
    observations,
    observations_full=None,
    trajectories,
    policy_logits_name,
    value_function_name,
    env_name=None,
    numpy_precision=6,
    inline_js=True,
    inline_large_json=None,
    batch_size=512,
    action_combos=None,
    action_group_fns=[
        lambda combo: "RIGHT" in combo,
        lambda combo: "LEFT" in combo,
        lambda combo: "UP" in combo,
        lambda combo: "DOWN" in combo,
        lambda combo: "RIGHT" not in combo
        and "LEFT" not in combo
        and "UP" not in combo
        and "DOWN" not in combo,
    ],
    layer_kwargs={},
    input_layer_include=False,
    input_layer_name="input",
    gae_gamma=None,
    gae_lambda=None,
    trajectory_bookmarks=16,
    nmf_features=8,
    nmf_attr_opts=None,
    vis_subdiv_mults=[0.25, 0.5, 1, 2],
    vis_subdiv_mult_default=1,
    vis_expand_mults=[1, 2, 4, 8],
    vis_expand_mult_default=4,
    vis_thumbnail_num_mult=4,
    vis_thumbnail_expand_mult=4,
    scrub_range=(42 / 64, 44 / 64),
    attr_integrate_steps=10,
    attr_max_paths=None,
    attr_policy=False,
    attr_single_channels=True,
    observations_subdir="observations/",
    trajectories_subdir="trajectories/",
    trajectories_scrub_subdir="trajectories_scrub/",
    features_subdir="features/",
    thumbnails_subdir="thumbnails/",
    attribution_subdir="attribution/",
    attribution_scrub_subdir="attribution_scrub/",
    features_grids_subdir="features_grids/",
    attribution_totals_subdir="attribution_totals/",
    video_height="16em",
    video_width="16em",
    video_speed=12,
    policy_display_height="2em",
    policy_display_width="40em",
    navigator_width="24em",
    scrubber_height="4em",
    scrubber_width="48em",
    scrubber_visible_duration=256,
    legend_item_height="6em",
    legend_item_width="6em",
    feature_viewer_height="40em",
    feature_viewer_width="40em",
    attribution_weight=0.9,
    graph_colors={
        "v": "green",
        "action": "red",
        "action_group": "orange",
        "advantage": "blue",
    },
    trajectory_color="blue",
):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    model = get_model(model_bytes)
    if rank == 0:
        js_source_path = get_compiled_js()

    if env_name is None:
        env_name = "unknown"
    if inline_large_json is None:
        inline_large_json = "://" not in output_dir
    layer_kwargs.setdefault("name_contains_one_of", None)
    layer_kwargs.setdefault("op_is_one_of", ["relu"])
    layer_kwargs.setdefault("bottleneck_only", True)
    layer_kwargs.setdefault("discard_first_n", 0)
    if observations_full is None:
        observations_full = observations
    if "observations_full" not in trajectories:
        trajectories["observations_full"] = trajectories["observations"]
    if np.issubdtype(observations.dtype, np.integer):
        observations = observations / np.float32(255)
    if np.issubdtype(observations_full.dtype, np.integer):
        observations_full = observations_full / np.float32(255)
    if np.issubdtype(trajectories["observations"].dtype, np.integer):
        trajectories["observations"] = trajectories["observations"] / np.float32(255)
    if np.issubdtype(trajectories["observations_full"].dtype, np.integer):
        trajectories["observations_full"] = trajectories[
            "observations_full"
        ] / np.float32(255)
    if action_combos is None:
        num_actions = get_shape(model, policy_logits_name)[-1]
        action_combos = list(map(lambda x: (str(x),), range(num_actions)))
        if env_name == "coinrun_old":
            action_combos = [
                (),
                ("RIGHT",),
                ("LEFT",),
                ("UP",),
                ("RIGHT", "UP"),
                ("LEFT", "UP"),
                ("DOWN",),
                ("A",),
                ("B",),
            ][:num_actions]
    if gae_gamma is None:
        gae_gamma = 0.999
    if gae_lambda is None:
        gae_lambda = 0.95

    layer_names = get_layer_names(
        model, [policy_logits_name, value_function_name], **layer_kwargs
    )
    if not layer_names:
        raise RuntimeError(
            "No appropriate layers found. "
            "Please adapt layer_kwargs to your architecture"
        )
    squash = lambda s: s.replace("/", "").replace("_", "")
    if len(set([squash(layer_key) for layer_key in layer_names.keys()])) < len(
        layer_names
    ):
        raise RuntimeError(
            "Error squashing abbreviated layer names. "
            "Different substitutions must be used"
        )
    mpi_enumerate = lambda l: (
        lambda indices: list(enumerate(l))[indices[rank] : indices[rank + 1]]
    )(np.linspace(0, len(l), comm.Get_size() + 1).astype(int))
    save_image = lambda image, path: save(
        image, os.path.join(output_dir, path), domain=(0, 1)
    )
    save_images = lambda images, path: save_image(
        concatenate_horizontally(images), path
    )
    json_preloaded = {}
    save_json = lambda data, path: (
        json_preloaded.update({path: data})
        if inline_large_json
        else save(data, os.path.join(output_dir, path), indent=None)
    )
    get_scrub_slice = lambda width: slice(
        int(np.round(scrub_range[0] * width)),
        int(
            np.maximum(
                np.round(scrub_range[1] * width), np.round(scrub_range[0] * width) + 1
            )
        ),
    )
    action_groups = [
        [action for action, combo in enumerate(action_combos) if group_fn(combo)]
        for group_fn in action_group_fns
    ]
    action_groups = list(
        filter(lambda action_group: len(action_group) > 1, action_groups)
    )

    for index, observation in mpi_enumerate(observations_full):
        observation_path = os.path.join(observations_subdir, f"{index}.png")
        save_image(observation, observation_path)
    for index, trajectory_observations in mpi_enumerate(
        trajectories["observations_full"]
    ):
        trajectory_path = os.path.join(trajectories_subdir, f"{index}.png")
        save_images(trajectory_observations, trajectory_path)
        scrub_slice = get_scrub_slice(trajectory_observations.shape[2])
        scrub = trajectory_observations[:, :, scrub_slice, :]
        scrub_path = os.path.join(trajectories_scrub_subdir, f"{index}.png")
        save_images(scrub, scrub_path)

    trajectories["policy_logits"] = []
    trajectories["values"] = []
    for trajectory_observations in trajectories["observations"]:
        trajectories["policy_logits"].append(
            batched_get(
                trajectory_observations,
                batch_size,
                lambda minibatch: get_acts(model, policy_logits_name, minibatch),
            )
        )
        trajectories["values"].append(
            batched_get(
                trajectory_observations,
                batch_size,
                lambda minibatch: get_acts(model, value_function_name, minibatch),
            )
        )
    trajectories["policy_logits"] = np.array(trajectories["policy_logits"])
    trajectories["values"] = np.array(trajectories["values"])
    trajectories["advantages"] = compute_gae(
        trajectories, gae_gamma=gae_gamma, gae_lambda=gae_lambda
    )
    if "dones" not in trajectories:
        trajectories["dones"] = np.concatenate(
            [
                trajectories["firsts"][:, 1:],
                np.zeros_like(trajectories["firsts"][:, :1]),
            ],
            axis=-1,
        )

    bookmarks = {
        "high": get_bookmarks(trajectories, sign=1, num=trajectory_bookmarks),
        "low": get_bookmarks(trajectories, sign=-1, num=trajectory_bookmarks),
    }

    nmf_kwargs = {"attr_layer_name": value_function_name}
    if nmf_attr_opts is not None:
        nmf_kwargs["attr_opts"] = nmf_attr_opts
    nmfs = {
        layer_key: LayerNMF(
            model,
            layer_name,
            observations,
            obses_full=observations_full,
            features=nmf_features,
            **nmf_kwargs,
        )
        for layer_key, layer_name in layer_names.items()
    }

    features = []
    attributions = []
    attribution_totals = []

    for layer_key, layer_name in layer_names.items():
        nmf = nmfs[layer_key]

        if rank == 0:
            thumbnails = []
            for number in range(nmf.features):
                thumbnail = nmf.vis_dataset_thumbnail(
                    number,
                    num_mult=vis_thumbnail_num_mult,
                    expand_mult=vis_thumbnail_expand_mult,
                )[0]
                thumbnail = rescale_opacity(thumbnail, max_scale=1, keep_zeros=True)
                thumbnails.append(thumbnail)
            thumbnails_path = os.path.join(
                thumbnails_subdir, f"{squash(layer_key)}.png"
            )
            save_images(thumbnails, thumbnails_path)

        for _, number in mpi_enumerate(range(nmf.features)):
            feature = {
                "layer": layer_key,
                "number": number,
                "images": [],
                "overlay_grids": [],
                "metadata": {"subdiv_mult": [], "expand_mult": []},
            }
            for subdiv_mult in vis_subdiv_mults:
                for expand_mult in vis_expand_mults:
                    image, overlay_grid = nmf.vis_dataset(
                        number, subdiv_mult=subdiv_mult, expand_mult=expand_mult
                    )
                    image = rescale_opacity(image)
                    filename_root = (
                        f"{squash(layer_key)}_"
                        f"feature{number}_"
                        f"{number_to_string(subdiv_mult)}_"
                        f"{number_to_string(expand_mult)}"
                    )
                    image_filename = filename_root + ".png"
                    overlay_grid_filename = filename_root + ".json"
                    image_path = os.path.join(features_subdir, image_filename)
                    overlay_grid_path = os.path.join(
                        features_grids_subdir, overlay_grid_filename
                    )
                    save_image(image, image_path)
                    save_json(overlay_grid, overlay_grid_path)
                    feature["images"].append(image_filename)
                    feature["overlay_grids"].append(overlay_grid_filename)
                    feature["metadata"]["subdiv_mult"].append(subdiv_mult)
                    feature["metadata"]["expand_mult"].append(expand_mult)
            features.append(feature)

    for layer_key, layer_name in (
        [(input_layer_name, None)] if input_layer_include else []
    ) + list(layer_names.items()):
        if layer_name is None:
            nmf = None
        else:
            nmf = nmfs[layer_key]

        for index, trajectory_observations in mpi_enumerate(
            trajectories["observations"]
        ):
            attribution = {
                "layer": layer_key,
                "trajectory": index,
                "images": [],
                "metadata": {"type": [], "data": [], "direction": [], "channel": []},
            }
            if layer_name is not None:
                totals = {
                    "layer": layer_key,
                    "trajectory": index,
                    "channels": [],
                    "residuals": [],
                    "metadata": {"type": [], "data": []},
                }

            def get_attr_minibatch(
                minibatch, output_name, *, score_fn=default_score_fn
            ):
                if layer_name is None:
                    return get_grad(model, output_name, minibatch, score_fn=score_fn)
                elif attr_max_paths is None:
                    return get_attr(
                        model,
                        output_name,
                        layer_name,
                        minibatch,
                        score_fn=score_fn,
                        integrate_steps=attr_integrate_steps,
                    )
                else:
                    return get_multi_path_attr(
                        model,
                        output_name,
                        layer_name,
                        minibatch,
                        nmf,
                        score_fn=score_fn,
                        integrate_steps=attr_integrate_steps,
                        max_paths=attr_max_paths,
                    )

            def get_attr_batched(output_name, *, score_fn=default_score_fn):
                return batched_get(
                    trajectory_observations,
                    batch_size,
                    lambda minibatch: get_attr_minibatch(
                        minibatch, output_name, score_fn=score_fn
                    ),
                )

            def transform_attr(attr):
                if layer_name is None:
                    return attr, None
                else:
                    attr_trans = nmf.transform(np.maximum(attr, 0)) - nmf.transform(
                        np.maximum(-attr, 0)
                    )
                    attr_res = (
                        attr
                        - (
                            nmf.inverse_transform(np.maximum(attr_trans, 0))
                            - nmf.inverse_transform(np.maximum(-attr_trans, 0))
                        )
                    ).sum(-1, keepdims=True)
                    nmf_norms = nmf.channel_dirs.sum(-1)
                    return attr_trans * nmf_norms[None, None, None], attr_res

            def save_attr(attr, attr_res, *, type_, data):
                if attr_res is None:
                    attr_res = np.zeros_like(attr).sum(-1, keepdims=True)
                filename_root = f"{squash(layer_key)}_{index}_{type_}"
                if data is not None:
                    filename_root = f"{filename_root}_{data}"
                if layer_name is not None:
                    channels_filename = f"{filename_root}_channels.json"
                    residuals_filename = f"{filename_root}_residuals.json"
                    channels_path = os.path.join(
                        attribution_totals_subdir, channels_filename
                    )
                    residuals_path = os.path.join(
                        attribution_totals_subdir, residuals_filename
                    )
                    save_json(attr.sum(-2).sum(-2), channels_path)
                    save_json(attr_res[..., 0].sum(-1).sum(-1), residuals_path)
                    totals["channels"].append(channels_filename)
                    totals["residuals"].append(residuals_filename)
                    totals["metadata"]["type"].append(type_)
                    totals["metadata"]["data"].append(data)
                attr_scale = np.median(attr.max(axis=(-3, -2, -1)))
                if attr_scale == 0:
                    attr_scale = attr.max()
                if attr_scale == 0:
                    attr_scale = 1
                attr_scaled = attr / attr_scale
                attr_res_scaled = attr_res / attr_scale
                channels = ["prin", "all"]
                if attr_single_channels and layer_name is not None:
                    channels += list(range(nmf.features)) + ["res"]
                for direction in ["abs", "pos", "neg"]:
                    if direction == "abs":
                        attr = np.abs(attr_scaled)
                        attr_res = np.abs(attr_res_scaled)
                    elif direction == "pos":
                        attr = np.maximum(attr_scaled, 0)
                        attr_res = np.maximum(attr_res_scaled, 0)
                    elif direction == "neg":
                        attr = np.maximum(-attr_scaled, 0)
                        attr_res = np.maximum(-attr_res_scaled, 0)
                    for channel in channels:
                        if isinstance(channel, int):
                            attr_single = attr.copy()
                            attr_single[..., :channel] = 0
                            attr_single[..., (channel + 1) :] = 0
                            images = channels_to_rgb(attr_single)
                        elif channel == "res":
                            images = attr_res.repeat(3, axis=-1)
                        else:
                            images = channels_to_rgb(attr)
                            if channel == "all":
                                images += attr_res.repeat(3, axis=-1)
                        images = brightness_to_opacity(
                            conv2d(images, filter_=norm_filter(15))
                        )
                        suffix = f"{direction}_{channel}"
                        images_filename = f"{filename_root}_{suffix}.png"
                        images_path = os.path.join(attribution_subdir, images_filename)
                        save_images(images, images_path)
                        scrub = images[:, :, get_scrub_slice(images.shape[2]), :]
                        scrub_path = os.path.join(
                            attribution_scrub_subdir, images_filename
                        )
                        save_images(scrub, scrub_path)
                        attribution["images"].append(images_filename)
                        attribution["metadata"]["type"].append(type_)
                        attribution["metadata"]["data"].append(data)
                        attribution["metadata"]["direction"].append(direction)
                        attribution["metadata"]["channel"].append(channel)

            attr_v = get_attr_batched(value_function_name)
            attr_v_trans, attr_v_res = transform_attr(attr_v)
            save_attr(attr_v_trans, attr_v_res, type_="v", data=None)
            if attr_policy:
                attr_actions = np.array(
                    [
                        get_attr_batched(
                            policy_logits_name, score_fn=lambda t: t[..., action],
                        )
                        for action in range(len(action_combos))
                    ]
                )
                # attr_pi = attr_actions.sum(axis=-1).transpose(
                #     (1, 2, 3, 0))
                # attr_pi = np.concatenate([
                #     attr_pi[..., group].sum(axis=-1, keepdims=True)
                #     for group in attr_action_groups
                # ],
                #                          axis=-1)
                # save_attr(attr_pi, None, type_='pi', data=None)
                for action, attr in enumerate(attr_actions):
                    attr_trans, attr_res = transform_attr(attr)
                    save_attr(attr_trans, attr_res, type_="action", data=action)
                for action_group, actions in enumerate(action_groups):
                    attr = attr_actions[actions].sum(axis=0)
                    attr_trans, attr_res = transform_attr(attr)
                    save_attr(
                        attr_trans, attr_res, type_="action_group", data=action_group
                    )
            attributions.append(attribution)
            if layer_name is not None:
                attribution_totals.append(totals)

    features = comm.gather(features, root=0)
    attributions = comm.gather(attributions, root=0)
    attribution_totals = comm.gather(attribution_totals, root=0)

    if rank == 0:
        features = [feature for l in features for feature in l]
        attributions = [attribution for l in attributions for attribution in l]
        attribution_totals = [totals for l in attribution_totals for totals in l]
        layer_keys = ([input_layer_name] if input_layer_include else []) + list(
            layer_names.keys()
        )
        action_colors = get_html_colors(
            len(action_combos),
            grayscale=True,
            mix_with=np.array([0.75, 0.75, 0.75]),
            mix_weight=0.25,
        )
        props = {
            "input_layer": input_layer_name,
            "layers": layer_keys,
            "features": features,
            "attributions": attributions,
            "attribution_policy": attr_policy,
            "attribution_single_channels": attr_single_channels,
            "attribution_totals": attribution_totals,
            "colors": {
                "features": get_html_colors(nmf_features),
                "actions": action_colors,
                "graphs": graph_colors,
                "trajectory": trajectory_color,
            },
            "action_combos": action_combos,
            "action_groups": action_groups,
            "trajectories": {
                "actions": trajectories["actions"],
                "rewards": trajectories["rewards"],
                "dones": trajectories["dones"],
                "policy_logits": trajectories["policy_logits"],
                "values": trajectories["values"],
                "advantages": trajectories["advantages"],
            },
            "bookmarks": bookmarks,
            "vis_defaults": {
                "subdiv_mult": vis_subdiv_mult_default,
                "expand_mult": vis_expand_mult_default,
            },
            "subdirs": {
                "observations": observations_subdir,
                "trajectories": trajectories_subdir,
                "trajectories_scrub": trajectories_scrub_subdir,
                "features": features_subdir,
                "thumbnails": thumbnails_subdir,
                "attribution": attribution_subdir,
                "attribution_scrub": attribution_scrub_subdir,
                "features_grids": features_grids_subdir,
                "attribution_totals": attribution_totals_subdir,
            },
            "formatting": {
                "video_height": video_height,
                "video_width": video_width,
                "video_speed": video_speed,
                "policy_display_height": policy_display_height,
                "policy_display_width": policy_display_width,
                "navigator_width": navigator_width,
                "scrubber_height": scrubber_height,
                "scrubber_width": scrubber_width,
                "scrubber_visible_duration": scrubber_visible_duration,
                "legend_item_height": legend_item_height,
                "legend_item_width": legend_item_width,
                "feature_viewer_height": feature_viewer_height,
                "feature_viewer_width": feature_viewer_width,
                "attribution_weight": attribution_weight,
            },
            "json_preloaded": json_preloaded,
        }

        if inline_js:
            js_path = js_source_path
        else:
            with open(js_source_path, "r") as fp:
                js_code = fp.read()
            js_path = os.path.join(output_dir, "interface.js")
            with write_handle(js_path, "w") as fp:
                fp.write(js_code)
        html_path = os.path.join(output_dir, "interface.html")
        compile_html(
            js_path,
            html_path=html_path,
            props=props,
            precision=numpy_precision,
            inline_js=inline_js,
            svelte_to_js=False,
        )
        if output_dir.startswith("gs://"):
            if not inline_js:
                subprocess.run(
                    [
                        "gsutil",
                        "setmeta",
                        "-h",
                        "Content-Type: text/javascript",
                        js_path,
                    ]
                )
            subprocess.run(
                ["gsutil", "setmeta", "-h", "Content-Type: text/html", html_path]
            )
        elif output_dir.startswith("https://"):
            output_dir_parsed = urllib.parse.urlparse(output_dir)
            az_account, az_hostname = output_dir_parsed.netloc.split(".", 1)
            if az_hostname == "blob.core.windows.net":
                az_container = removeprefix(output_dir_parsed.path, "/").split("/")[0]
                az_prefix = f"https://{az_account}.{az_hostname}/{az_container}/"
                if not inline_js:
                    js_az_name = removeprefix(js_path, az_prefix)
                    subprocess.run(
                        [
                            "az",
                            "storage",
                            "blob",
                            "update",
                            "--container-name",
                            az_container,
                            "--name",
                            js_az_name,
                            "--account-name",
                            az_account,
                            "--content-type",
                            "application/javascript",
                        ]
                    )
                html_az_name = removeprefix(html_path, az_prefix)
                subprocess.run(
                    [
                        "az",
                        "storage",
                        "blob",
                        "update",
                        "--container-name",
                        az_container,
                        "--name",
                        html_az_name,
                        "--account-name",
                        az_account,
                        "--content-type",
                        "text/html",
                    ]
                )
