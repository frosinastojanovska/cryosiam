import argparse
from pathlib import Path

import h5py
import yaml
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


def decode_strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def normalize_crop_size(value):
    if isinstance(value, (int, float)):
        size = [int(value)] * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        size = [int(v) for v in value]
    else:
        raise ValueError("registration_crop_size must be an integer or [z, y, x].")

    if any(v < 8 for v in size):
        raise ValueError("registration_crop_size values must be >= 8.")
    return tuple(size)


def crop_size_for_class(crop_size_config, class_name):
    if isinstance(crop_size_config, dict):
        value = crop_size_config.get(class_name, crop_size_config.get("default", 64))
    else:
        value = crop_size_config
    return normalize_crop_size(value)


def centered_crop_slices(center_zyx, crop_size, volume_shape):
    center_zyx = np.asarray(center_zyx, dtype=np.float64)
    crop_size = np.asarray(crop_size, dtype=np.int64)
    volume_shape = np.asarray(volume_shape, dtype=np.int64)

    start = np.floor(center_zyx - crop_size / 2.0).astype(np.int64)
    stop = start + crop_size

    for d in range(3):
        if start[d] < 0:
            stop[d] -= start[d]
            start[d] = 0
        if stop[d] > volume_shape[d]:
            start[d] -= stop[d] - volume_shape[d]
            stop[d] = volume_shape[d]
            start[d] = max(start[d], 0)

    return tuple(slice(int(start[d]), int(stop[d])) for d in range(3))


def keep_highest_similarity_points_from_crop(sim_crop, candidate_mask, crop_offset_zyx,
                                             center_zyx, max_points, min_points):
    valid_flat = np.flatnonzero(candidate_mask.reshape(-1))
    n = int(valid_flat.size)
    if n < int(min_points):
        return None, None

    similarities = sim_crop.reshape(-1)[valid_flat]
    n_keep = min(int(max_points), n)

    if n_keep < n:
        idx = np.argpartition(similarities, -n_keep)[-n_keep:]
        idx = idx[np.argsort(similarities[idx])[::-1]]
    else:
        idx = np.argsort(similarities)[::-1]

    selected_flat = valid_flat[idx]
    selected_scores = similarities[idx].astype(np.float32, copy=False)
    local_zyx = np.column_stack(np.unravel_index(selected_flat, sim_crop.shape)).astype(np.float64)
    selected_zyx = local_zyx + np.asarray(crop_offset_zyx, dtype=np.float64)[None, :]

    points_xyz = selected_zyx[:, ::-1]
    center_xyz = np.asarray(center_zyx, dtype=np.float64)[::-1]
    points_xyz = points_xyz - center_xyz[None, :]

    return points_xyz.astype(np.float32), selected_scores


def load_config(config_file_path):
    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    if "prediction_folder" not in cfg:
        raise KeyError("Config is missing top-level prediction_folder.")

    orient_cfg = cfg.get("initial_orientations") or {}
    if not isinstance(orient_cfg, dict):
        raise ValueError("initial_orientations must be a mapping in the config file.")

    reference_folder = orient_cfg.get("reference_folder") or None
    if reference_folder is None:
        raise ValueError(
            "Set initial_orientations.reference_folder in the config before saving references."
        )

    params = {
        "prediction_folder": cfg["prediction_folder"],
        "reference_folder": reference_folder,
        "max_points": int(orient_cfg.get("max_points", 1000)),
        "min_points": int(orient_cfg.get("min_points", 100)),
        "registration_crop_size": orient_cfg.get("registration_crop_size", 64),
        "use_distance_mask": bool(orient_cfg.get("use_distance_mask", False)),
        "exclude_other_instances": bool(orient_cfg.get("exclude_other_instances", False)),
        "similarity_threshold": float(orient_cfg.get("similarity_threshold", 0.1)),
    }

    return params


def prediction_path(prediction_folder, tomo_name):
    folder = Path(prediction_folder)

    direct = folder / f"{tomo_name}_preds.h5"
    if direct.is_file():
        return direct

    if tomo_name.endswith("_preds"):
        alternate = folder / f"{tomo_name}.h5"
        if alternate.is_file():
            return alternate

    raise FileNotFoundError(
        f'Could not find prediction file for tomogram "{tomo_name}" in {folder}. '
        f'Expected: {direct.name}'
    )


def expanded_crop_slices(crop_slices, volume_shape, margin=4):
    expanded = []
    inner = []
    for slc, size in zip(crop_slices, volume_shape):
        start = max(0, int(slc.start) - int(margin))
        stop = min(int(size), int(slc.stop) + int(margin))
        expanded.append(slice(start, stop))
        inner.append(slice(int(slc.start) - start, int(slc.stop) - start))
    return tuple(expanded), tuple(inner)


def other_filled_instances_in_crop(hf, instance_mask, seg_mask, class_label, class_name,
                                   target_instance_id, registration_slices, margin=4):
    solid_key = f"solid_{class_name}"
    if solid_key not in hf:
        raise KeyError(
            f'Prediction H5 is missing "{solid_key}". '
            'exclude_other_instances=True requires the existing filled class mask.'
        )

    expanded, inner = expanded_crop_slices(
        registration_slices, instance_mask.shape, margin=margin)
    instance_crop = instance_mask[expanded]
    seg_crop = seg_mask[expanded]

    class_marker_voxels = (seg_crop == int(class_label)) & (instance_crop > 0)
    if not np.any(class_marker_voxels):
        return np.zeros(tuple(s.stop - s.start for s in registration_slices), dtype=bool)

    marker_ids = np.unique(instance_crop[class_marker_voxels])
    if not np.any(marker_ids != int(target_instance_id)):
        return np.zeros(tuple(s.stop - s.start for s in registration_slices), dtype=bool)

    markers = np.zeros(instance_crop.shape, dtype=np.int32)
    markers[class_marker_voxels] = instance_crop[class_marker_voxels].astype(np.int32, copy=False)
    solid = hf[solid_key][expanded].astype(bool, copy=False)
    distance = distance_transform_edt(solid).astype(np.float32, copy=False)
    filled = watershed(-distance, markers=markers, mask=solid).astype(np.int32, copy=False)
    filled = filled[inner]

    return (filled > 0) & (filled != int(target_instance_id))


def extract_reference(hf, tomo_name, instance_id, max_points, min_points,
                      registration_crop_size=64,
                      use_distance_mask=False, exclude_other_instances=False,
                      similarity_threshold=0.1):
    if "instance_mask" not in hf:
        raise KeyError('Prediction H5 is missing "instance_mask".')

    seg_key = "seg_mask" if "seg_mask" in hf else "seg_mask_raw"
    if seg_key not in hf:
        raise KeyError('Prediction H5 is missing "seg_mask" and "seg_mask_raw".')

    if "class_names" not in hf.attrs:
        raise KeyError('Prediction H5 is missing H5 attribute "class_names".')

    instance_mask = hf["instance_mask"][()]
    seg_mask = hf[seg_key][()]
    class_names = decode_strings(hf.attrs["class_names"])

    instance = instance_mask == int(instance_id)
    n_voxels = int(instance.sum())
    if n_voxels == 0:
        raise ValueError(f"Instance ID {instance_id} does not exist in {tomo_name}.")

    labels = seg_mask[instance]
    labels = labels[labels > 0].astype(np.int64, copy=False)
    if len(labels) == 0:
        raise ValueError(f"Instance ID {instance_id} has no semantic class label.")

    class_label = int(np.bincount(labels).argmax())
    if class_label < 1 or class_label > len(class_names):
        raise ValueError(
            f"Instance ID {instance_id} has invalid semantic label {class_label}."
        )

    class_name = class_names[class_label - 1]
    sim_key = f"sim_{class_name}"
    if sim_key not in hf:
        raise KeyError(
            f'Prediction H5 is missing "{sim_key}". Save similarity maps during prediction.'
        )

    instance_coords_zyx = np.argwhere(instance)
    center_zyx = instance_coords_zyx.mean(axis=0)
    mean_sim = float(hf[sim_key][()][instance].mean())

    crop_size = crop_size_for_class(registration_crop_size, class_name)
    reg_slc = centered_crop_slices(center_zyx, crop_size, instance_mask.shape)
    sim_crop = hf[sim_key][reg_slc].astype(np.float32, copy=False)

    candidate_mask = sim_crop > float(similarity_threshold)
    if exclude_other_instances:
        other_filled = other_filled_instances_in_crop(
            hf, instance_mask, seg_mask, class_label, class_name,
            instance_id, reg_slc)
        candidate_mask &= ~other_filled

    if use_distance_mask:
        distance_key = f"distance_{class_name}"
        if distance_key not in hf:
            raise KeyError(
                f'use_distance_mask=True but prediction H5 is missing "{distance_key}".'
            )
        candidate_mask &= hf[distance_key][reg_slc] > 0

    offset_zyx = np.array([s.start for s in reg_slc], dtype=np.int64)
    points_xyz, point_scores = keep_highest_similarity_points_from_crop(
        sim_crop,
        candidate_mask,
        offset_zyx,
        center_zyx,
        max_points=max_points,
        min_points=min_points,
    )

    if points_xyz is None or len(points_xyz) < 10:
        raise ValueError(
            f"Instance ID {instance_id} has too few usable similarity points."
        )

    reference_id = f"{class_name}_{tomo_name}_inst{int(instance_id)}"

    return {
        "reference_id": reference_id,
        "class_name": class_name,
        "points_xyz": points_xyz,
        "point_scores": point_scores,
        "mean_sim": mean_sim,
        "tomo": tomo_name,
        "instance_id": int(instance_id),
        "center_z": float(center_zyx[0]),
        "center_y": float(center_zyx[1]),
        "center_x": float(center_zyx[2]),
        "n_voxels": n_voxels,
        "similarity_threshold": float(similarity_threshold),
    }


def save_reference(reference, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("points_xyz", data=reference["points_xyz"], compression="lzf")
        hf.create_dataset("scores", data=reference["point_scores"], compression="lzf")
        hf.attrs["class_name"] = reference["class_name"]
        hf.attrs["reference_id"] = reference["reference_id"]
        hf.attrs["mean_sim"] = float(reference["mean_sim"])
        hf.attrs["tomo"] = reference["tomo"]
        hf.attrs["instance_id"] = int(reference["instance_id"])
        hf.attrs["center_z"] = float(reference["center_z"])
        hf.attrs["center_y"] = float(reference["center_y"])
        hf.attrs["center_x"] = float(reference["center_x"])
        hf.attrs["n_voxels"] = int(reference["n_voxels"])
        hf.attrs["similarity_threshold"] = float(reference.get("similarity_threshold", 0.1))


def main(config_file, tomo_name, instance_ids):
    params = load_config(config_file)
    h5_path = prediction_path(params["prediction_folder"], tomo_name)
    reference_folder = Path(params["reference_folder"])
    reference_folder.mkdir(parents=True, exist_ok=True)

    print(f"Prediction: {h5_path}")
    print(f"Reference folder: {reference_folder}")
    print(f"Instance IDs: {instance_ids}")
    print(f"Exclude other filled instances: {params['exclude_other_instances']}")
    print(f"Similarity threshold: > {params['similarity_threshold']}")

    saved = 0
    with h5py.File(h5_path, "r") as hf:
        for instance_id in instance_ids:
            reference = extract_reference(
                hf,
                tomo_name=tomo_name,
                instance_id=instance_id,
                max_points=params["max_points"],
                min_points=params["min_points"],
                registration_crop_size=params["registration_crop_size"],
                use_distance_mask=params["use_distance_mask"],
                exclude_other_instances=params["exclude_other_instances"],
                similarity_threshold=params["similarity_threshold"],
            )

            out_path = reference_folder / f"{reference['reference_id']}.h5"
            save_reference(reference, out_path)
            saved += 1

            print(
                f"  saved instance {instance_id}: class={reference['class_name']}  "
                f"mean_sim={reference['mean_sim']:.5f}  "
                f"points={len(reference['points_xyz'])} -> {out_path.name}"
            )

    print(f"\nSaved {saved} reference(s) -> {reference_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save selected predicted particle instances as orientation references."
    )
    parser.add_argument("--config_file", required=True, help="Prediction YAML config file")
    parser.add_argument("--tomo_name", required=True, help="Tomogram name without _preds.h5")
    parser.add_argument("--instance_ids", required=True, nargs="+", type=int,
                        help="One or more final instance_mask IDs")
    args = parser.parse_args()

    main(args.config_file, args.tomo_name, args.instance_ids)
