import os
import warnings
import zlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import mrcfile
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects, binary_erosion, affine_transform, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage.segmentation import watershed
from tqdm import tqdm
from cryosiam.utils import parser_helper

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "This script requires Open3D. Install it with: pip install open3d"
    ) from exc

try:
    from pytorch3d.ops import knn_points
except ImportError:
    knn_points = None


# Public YAML interface:
#   initial_orientations.backend: cpu | gpu
# GPU worker devices are derived automatically from parameters.gpu_devices.
# backend=gpu accelerates both RANSAC and the [8,4,2,1] similarity refinement.


def decode_strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def prediction_files(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".h5":
            raise ValueError(f"Expected an .h5 prediction file, got: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    files = sorted(path.glob("*_preds.h5"))
    if not files:
        files = sorted(path.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 prediction files found in {path}")
    return files


def tomo_name_from_path(path):
    name = Path(path).stem
    if name.endswith("_preds"):
        name = name[:-6]
    return name


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


def resolve_bbox_slices(bounds):
    return tuple(slice(int(start), int(stop)) for start, stop in bounds)


def normalize_similarity_image(image, value_range=None):
    if value_range is None:
        lo = float(np.nanmin(image))
        hi = float(np.nanmax(image))
    else:
        lo, hi = value_range
        lo = float(lo)
        hi = float(hi)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        norm = np.zeros_like(image, dtype=np.float32)
    else:
        norm = (image.astype(np.float32) - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)

    return norm


def similarity_to_rgb(image, cmap_name="magma", value_range=(-1.0, 1.0),
                      mask=None, draw_contour=True):
    try:
        from matplotlib import colormaps
    except ImportError as exc:
        raise ImportError(
            "Saving particle views requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    norm = normalize_similarity_image(image, value_range=value_range)
    rgba = colormaps.get_cmap(cmap_name)(norm)
    rgb = np.round(255.0 * rgba[..., :3]).astype(np.uint8)

    if mask is not None:
        mask = mask.astype(bool, copy=False)
        rgb[~mask] = np.round(0.35 * rgb[~mask]).astype(np.uint8)
        if draw_contour and np.any(mask):
            contour = mask & ~binary_erosion(mask)
            rgb[contour] = np.array([255, 255, 255], dtype=np.uint8)

    return rgb


def scalar_to_rgb(image, source="similarity", cmap_name="magma",
                  value_range=(-1.0, 1.0), mask=None, draw_contour=True):
    if source == "tomogram":
        # Robust per-slice normalization is usually best for raw tomogram views.
        finite = np.isfinite(image)
        if np.any(finite):
            lo, hi = np.percentile(image[finite], [1.0, 99.0])
        else:
            lo, hi = 0.0, 1.0
        return similarity_to_rgb(image, cmap_name=cmap_name, value_range=(lo, hi),
                                 mask=mask, draw_contour=draw_contour)

    return similarity_to_rgb(image, cmap_name=cmap_name, value_range=value_range,
                             mask=mask, draw_contour=draw_contour)


def load_tomogram_crop(tomo_name, bbox, data_folder, file_extension):
    tomo_path = Path(data_folder) / f"{tomo_name}{file_extension}"
    if not tomo_path.exists():
        raise FileNotFoundError(f"Tomogram file not found: {tomo_path}")

    with mrcfile.open(tomo_path, permissive=True) as mrc:
        volume = mrc.data
        crop = volume[bbox].copy()

    return crop.astype(np.float32, copy=False)


def xyz_rotation_to_zyx(R_xyz):
    permutation = np.array([[0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]], dtype=np.float64)
    return permutation @ np.asarray(R_xyz, dtype=np.float64) @ permutation


def pad_particle_crop_for_rotation(sim_crop, mask_crop, center_local_zyx):
    shape = np.asarray(sim_crop.shape, dtype=np.int64)
    center_local_zyx = np.asarray(center_local_zyx, dtype=np.float64)

    if np.any(center_local_zyx < 0) or np.any(center_local_zyx > (shape - 1)):
        raise ValueError(
            f"Particle center {center_local_zyx.tolist()} lies outside crop "
            f"with shape {shape.tolist()}."
        )

    low_extent = center_local_zyx
    high_extent = (shape.astype(np.float64) - 1.0) - center_local_zyx
    max_extent = np.maximum(low_extent, high_extent)
    radius = float(np.linalg.norm(max_extent))

    side = int(2 * np.ceil(radius) + 3)
    if side % 2 == 0:
        side += 1

    out_shape = np.array([side, side, side], dtype=np.int64)
    out_center = (out_shape.astype(np.float64) - 1.0) / 2.0
    start = np.round(out_center - center_local_zyx).astype(np.int64)
    stop = start + shape

    if np.any(start < 0) or np.any(stop > out_shape):
        pad_before = np.maximum(-start, 0)
        pad_after = np.maximum(stop - out_shape, 0)
        extra = int(max(pad_before.max(), pad_after.max()))
        side += 2 * extra + 2
        if side % 2 == 0:
            side += 1
        out_shape[:] = side
        out_center = (out_shape.astype(np.float64) - 1.0) / 2.0
        start = np.round(out_center - center_local_zyx).astype(np.int64)
        stop = start + shape

    sim_padded = np.full(tuple(out_shape), -1.0, dtype=np.float32)
    mask_padded = np.zeros(tuple(out_shape), dtype=np.uint8)

    dst = tuple(slice(int(start[d]), int(stop[d])) for d in range(3))
    sim_padded[dst] = sim_crop.astype(np.float32, copy=False)
    mask_padded[dst] = mask_crop.astype(np.uint8, copy=False)

    return sim_padded, mask_padded, out_center


def rotate_particle_volume(sim_crop, mask_crop, center_local_zyx, R_xyz):
    sim_padded, mask_padded, center = pad_particle_crop_for_rotation(
        sim_crop, mask_crop, center_local_zyx)

    R_zyx = xyz_rotation_to_zyx(R_xyz)
    matrix = R_zyx.T
    offset = center - matrix @ center

    rotated_sim = affine_transform(sim_padded,
                                   matrix=matrix,
                                   offset=offset,
                                   output_shape=sim_padded.shape,
                                   order=1,
                                   mode="constant",
                                   cval=-1.0,
                                   prefilter=False)
    rotated_mask = affine_transform(mask_padded,
                                    matrix=matrix,
                                    offset=offset,
                                    output_shape=mask_padded.shape,
                                    order=0,
                                    mode="constant",
                                    cval=0,
                                    prefilter=False) > 0

    return rotated_sim, rotated_mask


def save_aligned_particle_views(records, result, out_dir, image_format="png",
                                cmap="magma", value_range=(-1.0, 1.0),
                                draw_contour=True, view_source="similarity",
                                data_folder=None, file_extension=".mrc",
                                show_score=True):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "Saving particle views requires Pillow. Install it with: pip install pillow"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_lookup = {
        (str(row.tomo), int(row.instance_id)): row
        for row in result.itertuples(index=False)
    }

    total = 0
    for rec in records:
        key = (rec["tomo"], int(rec["instance_id"]))
        row = result_lookup.get(key)
        if row is None:
            continue

        # Use the translation-corrected center from the orientation result so
        # diagnostic views match the coordinates that are written to the STAR
        # file and subsequently used for subtomogram averaging.
        center_global_zyx = np.array([row.center_z, row.center_y, row.center_x],
                                     dtype=np.float64)
        crop_size = tuple(
            int(stop - start) for start, stop in rec["registration_bounds"])

        with h5py.File(rec["h5_path"], "r") as hf:
            bbox = centered_crop_slices(
                center_global_zyx, crop_size, hf["instance_mask"].shape)
            offset_zyx = np.array([s.start for s in bbox], dtype=np.float64)
            center_local_zyx = center_global_zyx - offset_zyx

            if view_source == "similarity":
                value_key = f"sim_{rec['class_name']}"
                value_crop = hf[value_key][bbox]
            else:
                value_crop = load_tomogram_crop(rec["tomo"], bbox, data_folder, file_extension)
            inst_crop = hf["instance_mask"][bbox] == rec["instance_id"]

        R_xyz = np.array([[row.r00, row.r01, row.r02],
                          [row.r10, row.r11, row.r12],
                          [row.r20, row.r21, row.r22]], dtype=np.float64)

        rotated_value, rotated_mask = rotate_particle_volume(
            value_crop, inst_crop, center_local_zyx, R_xyz)

        z_mid = rotated_value.shape[0] // 2
        value_2d = rotated_value[z_mid]
        mask_2d = rotated_mask[z_mid]
        rgb = scalar_to_rgb(value_2d, source=view_source, cmap_name=cmap,
                            value_range=value_range, mask=mask_2d,
                            draw_contour=draw_contour)

        class_dir = out_dir / rec["class_name"]
        class_dir.mkdir(parents=True, exist_ok=True)

        reference_id = str(row.master_reference_id)
        safe_reference = reference_id.replace(os.sep, "_").replace(" ", "_")
        file_name = (f"{rec['tomo']}_inst{int(rec['instance_id']):05d}"
                     f"__{view_source}__ref_{safe_reference}.{image_format}")

        image = Image.fromarray(rgb)
        if show_score:
            score = float(getattr(row, "match_score", np.nan))
            if np.isfinite(score):
                draw = ImageDraw.Draw(image)
                text = f"score={score:.3f}"
                try:
                    bbox = draw.textbbox((4, 4), text)
                    draw.rectangle((2, 2, bbox[2] + 2, bbox[3] + 2), fill=(0, 0, 0))
                except AttributeError:
                    draw.rectangle((2, 2, 78, 16), fill=(0, 0, 0))
                draw.text((4, 4), text, fill=(255, 255, 255))

        image.save(class_dir / file_name)
        total += 1

    return total


def group_references_by_class(references):
    grouped = {}
    for ref in references:
        grouped.setdefault(ref["class_name"], []).append(ref)
    return grouped


def save_reference_views(refs_by_class, records, params, result=None):
    if not params["view_folder"]:
        return 0

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "Saving reference views requires Pillow. Install it with: pip install pillow"
        ) from exc

    record_lookup = {
        (str(rec["tomo"]), int(rec["instance_id"])): rec
        for rec in records
    }
    result_lookup = {}
    if result is not None and len(result) > 0:
        result_lookup = {
            (str(row.tomo), int(row.instance_id)): row
            for row in result.itertuples(index=False)
        }

    root = Path(params["view_folder"]) / "references"
    root.mkdir(parents=True, exist_ok=True)
    total = 0

    for class_name, refs in refs_by_class.items():
        if not refs:
            continue

        master_reference_id = str(refs[0]["reference_id"])
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for ref in refs:
            tomo = ref.get("tomo")
            instance_id = ref.get("instance_id")
            if tomo is None or instance_id is None:
                print(
                    f'WARNING: reference {ref["reference_id"]} has no tomo/instance_id metadata; '
                    "skipping its view."
                )
                continue

            key = (str(tomo), int(instance_id))
            rec = record_lookup.get(key)
            row = result_lookup.get(key)

            if rec is not None:
                if row is not None:
                    center = np.array(
                        [row.center_z, row.center_y, row.center_x],
                        dtype=np.float64)
                else:
                    center = np.array(
                        [rec["center_z"], rec["center_y"], rec["center_x"]],
                        dtype=np.float64)
                h5_path = rec["h5_path"]
            else:
                center_keys = ("center_z", "center_y", "center_x")
                if all(k in ref for k in center_keys):
                    center = np.array(
                        [ref["center_z"], ref["center_y"], ref["center_x"]],
                        dtype=np.float64)
                elif row is not None:
                    center = np.array(
                        [row.center_z, row.center_y, row.center_x],
                        dtype=np.float64)
                else:
                    print(
                        f'WARNING: no center metadata found for reference '
                        f'{ref["reference_id"]}; skipping its view.'
                    )
                    continue

                predictions = Path(params["predictions"])
                if predictions.is_file():
                    h5_path = predictions
                else:
                    direct = predictions / f"{tomo}_preds.h5"
                    alternate = predictions / f"{tomo}.h5"
                    if direct.is_file():
                        h5_path = direct
                    elif alternate.is_file():
                        h5_path = alternate
                    else:
                        print(
                            f'WARNING: prediction H5 not found for reference '
                            f'{ref["reference_id"]}; skipping its view.'
                        )
                        continue

            crop_size = crop_size_for_class(
                params["registration_crop_size"], class_name)

            with h5py.File(h5_path, "r") as hf:
                instance_shape = hf["instance_mask"].shape
                bbox = centered_crop_slices(center, crop_size, instance_shape)
                inst_crop = hf["instance_mask"][bbox] == int(instance_id)

                if params["view_source"] == "similarity":
                    value_key = f"sim_{class_name}"
                    if value_key not in hf:
                        raise KeyError(f'{h5_path}: missing "{value_key}".')
                    value_crop = hf[value_key][bbox].astype(np.float32, copy=True)
                    value_crop[value_crop <= params["similarity_threshold"]] = -1.0
                    if params["exclude_other_instances"]:
                        seg_key = "seg_mask" if "seg_mask" in hf else "seg_mask_raw"
                        class_names = decode_strings(hf.attrs["class_names"])
                        class_label = class_names.index(class_name) + 1
                        other_filled = other_filled_instances_in_crop(
                            hf, hf["instance_mask"], hf[seg_key], class_label, class_name,
                            int(instance_id), bbox)
                        value_crop[other_filled] = -1.0
                else:
                    value_crop = load_tomogram_crop(
                        str(tomo), bbox, params["data_folder"],
                        params["file_extension"])

            if "to_master" in ref:
                R_xyz = np.asarray(ref["to_master"][:3, :3], dtype=np.float64)
                ref_score = np.nan
            elif row is not None:
                R_xyz = np.array(
                    [[row.r00, row.r01, row.r02],
                     [row.r10, row.r11, row.r12],
                     [row.r20, row.r21, row.r22]],
                    dtype=np.float64)
                ref_score = float(getattr(row, "match_score", np.nan))
            elif str(ref["reference_id"]) == master_reference_id:
                R_xyz = np.eye(3, dtype=np.float64)
                ref_score = 1.0
            else:
                print(
                    f'WARNING: no orientation found for reference '
                    f'{ref["reference_id"]}; skipping its view.'
                )
                continue

            offset_zyx = np.array([sl.start for sl in bbox], dtype=np.float64)
            center_local_zyx = center - offset_zyx
            rotated_value, rotated_mask = rotate_particle_volume(
                value_crop, inst_crop, center_local_zyx, R_xyz)

            z_mid = rotated_value.shape[0] // 2
            rgb = scalar_to_rgb(
                rotated_value[z_mid],
                source=params["view_source"],
                cmap_name=params["view_cmap"],
                value_range=(-1.0, 1.0),
                mask=rotated_mask[z_mid],
                draw_contour=params["view_draw_contour"])

            image = Image.fromarray(rgb)
            if params["view_show_score"]:
                draw = ImageDraw.Draw(image)
                label = f'ref={ref["reference_id"]}'
                if str(ref["reference_id"]) == master_reference_id:
                    label += "  master"
                elif np.isfinite(ref_score):
                    label += f"  score={ref_score:.3f}"

                try:
                    text_bbox = draw.textbbox((4, 4), label)
                    draw.rectangle(
                        (2, 2, text_bbox[2] + 2, text_bbox[3] + 2),
                        fill=(0, 0, 0))
                except AttributeError:
                    draw.rectangle((2, 2, 180, 16), fill=(0, 0, 0))
                draw.text((4, 4), label, fill=(255, 255, 255))

            safe_ref = str(ref["reference_id"]).replace(os.sep, "_").replace(" ", "_")
            file_name = (
                f'{safe_ref}__{tomo}_inst{int(instance_id):05d}'
                f'__{params["view_source"]}__to_master.{params["view_format"]}'
            )
            image.save(class_dir / file_name)
            total += 1

    return total


def save_relion_star(result, csv_path):
    try:
        import starfile
    except ImportError as exc:
        raise ImportError(
            "Saving RELION STAR output requires starfile. Install it with: pip install starfile"
        ) from exc

    required = {
        "tomo", "instance_id", "center_x", "center_y", "center_z",
        "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(
            f"Cannot save RELION STAR file; missing columns: {sorted(missing)}"
        )

    rows = []
    for row in result.itertuples(index=False):
        R_particle_to_reference = np.array(
            [[row.r00, row.r01, row.r02],
             [row.r10, row.r11, row.r12],
             [row.r20, row.r21, row.r22]],
            dtype=np.float64)

        R_reference_to_particle = R_particle_to_reference.T
        rot, tilt, psi = rotation_to_zyz_degrees(R_reference_to_particle)

        tomo = str(row.tomo)
        star_row = {
            "rlnTomoName": tomo,
            "rlnMicrographName": tomo,
            "rlnCoordinateX": float(row.center_x),
            "rlnCoordinateY": float(row.center_y),
            "rlnCoordinateZ": float(row.center_z),
            "rlnAngleRot": float(rot),
            "rlnAngleTilt": float(tilt),
            "rlnAnglePsi": float(psi),
            "rlnInstanceId": int(row.instance_id),
        }
        if hasattr(row, "class"):
            star_row["rlnClassLabel"] = str(getattr(row, "class"))
        rows.append(star_row)

    star_path = Path(csv_path).with_suffix(".star")
    starfile.write(pd.DataFrame(rows), star_path, overwrite=True)
    return star_path


def transform_points(points_xyz, T):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    return points_xyz @ R.T + t[None, :]


def trimmed_source_to_target_distance(source_points_xyz, target_points_xyz,
                                      trim_fraction=0.85):
    source_points_xyz = np.asarray(source_points_xyz, dtype=np.float64)
    target_points_xyz = np.asarray(target_points_xyz, dtype=np.float64)
    if len(source_points_xyz) == 0 or len(target_points_xyz) == 0:
        return np.nan

    target_tree = cKDTree(target_points_xyz)
    distances = target_tree.query(source_points_xyz, k=1)[0]
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return np.nan

    distances.sort()
    n_keep = max(1, int(np.ceil(float(trim_fraction) * len(distances))))
    return float(distances[:n_keep].mean())


def distance_to_match_score(distance_value, scale):
    if not np.isfinite(distance_value):
        return np.nan
    scale = max(float(scale), 1e-6)
    return float(1.0 / (1.0 + distance_value / scale))


def keep_highest_similarity_points_from_crop(sim_crop, candidate_mask, crop_offset_zyx,
                                             center_zyx, max_points, min_points):
    """Select only the strongest candidate voxels without materializing all XYZ coordinates."""
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
    """Return only neighboring filled-instance voxels inside one registration crop.

    The watershed is intentionally local. This avoids the expensive full-tomogram
    distance transform/watershed that was previously run before RANSAC.
    """
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


def read_particles(h5_path, max_points=1000,
                   min_points=100, only_classes=None,
                   registration_crop_size=64, use_distance_mask=False,
                   exclude_other_instances=False, similarity_threshold=0.1):
    records = []
    tomo = tomo_name_from_path(h5_path)

    with h5py.File(h5_path, "r") as hf:
        if "instance_mask" not in hf:
            raise KeyError(
                f'{h5_path}: missing "instance_mask". Run the current '
                "postprocessing/prediction script first."
            )

        seg_key = "seg_mask" if "seg_mask" in hf else "seg_mask_raw"
        if seg_key not in hf:
            raise KeyError(f'{h5_path}: missing "seg_mask" and "seg_mask_raw".')

        if "class_names" not in hf.attrs:
            raise KeyError(f'{h5_path}: missing H5 attribute "class_names".')

        class_names = decode_strings(hf.attrs["class_names"])
        instance_mask = hf["instance_mask"][()]
        seg_mask = hf[seg_key][()]
        boxes = find_objects(instance_mask)

        iterator = tqdm(enumerate(boxes, start=1), total=len(boxes),
                        desc=f"Extracting particles ({tomo})", unit="particle", leave=False)
        for instance_id, slc in iterator:
            if slc is None:
                continue

            inst_crop = instance_mask[slc] == instance_id
            n_voxels = int(inst_crop.sum())
            if n_voxels == 0:
                continue

            labels = seg_mask[slc][inst_crop]
            labels = labels[labels > 0].astype(np.int64, copy=False)
            if len(labels) == 0:
                continue

            class_label = int(np.bincount(labels).argmax())
            if class_label < 1 or class_label > len(class_names):
                continue
            class_name = class_names[class_label - 1]

            if only_classes is not None and class_name not in only_classes:
                continue

            sim_key = f"sim_{class_name}"
            if sim_key not in hf:
                raise KeyError(
                    f'{h5_path}: missing "{sim_key}". Similarity maps must be '
                    "saved by the prediction script."
                )

            local_zyx = np.argwhere(inst_crop)
            offset_zyx = np.array([s.start for s in slc], dtype=np.int64)
            instance_coords_zyx = local_zyx + offset_zyx[None, :]
            center_zyx = instance_coords_zyx.mean(axis=0)

            instance_similarities = hf[sim_key][slc][inst_crop].astype(np.float32, copy=False)
            mean_sim = float(instance_similarities.mean())

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
                        f'{h5_path}: use_distance_mask=True but "{distance_key}" is missing.'
                    )
                candidate_mask &= hf[distance_key][reg_slc] > 0

            reg_offset_zyx = np.array([s.start for s in reg_slc], dtype=np.int64)
            points_xyz, point_scores = keep_highest_similarity_points_from_crop(
                sim_crop,
                candidate_mask,
                reg_offset_zyx,
                center_zyx,
                max_points=max_points,
                min_points=min_points)

            if points_xyz is None or len(points_xyz) < 10:
                continue

            records.append({"uid": f"{tomo}:{instance_id}",
                            "tomo": tomo,
                            "h5_path": str(h5_path),
                            "instance_id": int(instance_id),
                            "class_name": class_name,
                            "n_voxels": n_voxels,
                            "mean_sim": mean_sim,
                            "center_z": float(center_zyx[0]),
                            "center_y": float(center_zyx[1]),
                            "center_x": float(center_zyx[2]),
                            "points_xyz": points_xyz,
                            "point_scores": point_scores,
                            "registration_bounds": tuple((int(s.start), int(s.stop)) for s in reg_slc)})

    return records


def save_reference(reference, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("points_xyz", data=reference["points_xyz"], compression="lzf")
        if reference.get("point_scores") is not None:
            hf.create_dataset("scores", data=reference["point_scores"], compression="lzf")
        hf.attrs["class_name"] = reference["class_name"]
        hf.attrs["reference_id"] = reference["reference_id"]
        hf.attrs["mean_sim"] = float(reference.get("mean_sim", np.nan))
        if "tomo" in reference:
            hf.attrs["tomo"] = reference["tomo"]
        if "instance_id" in reference:
            hf.attrs["instance_id"] = int(reference["instance_id"])
        for key in ("center_z", "center_y", "center_x"):
            if key in reference:
                hf.attrs[key] = float(reference[key])
        if "n_voxels" in reference:
            hf.attrs["n_voxels"] = int(reference["n_voxels"])


def load_external_references(folder, only_classes=None):
    refs = []
    files = sorted(Path(folder).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 reference files found in {folder}")

    for path in files:
        with h5py.File(path, "r") as hf:
            if "points_xyz" not in hf:
                raise KeyError(
                    f'{path}: external references must contain dataset "points_xyz".'
                )
            if "class_name" not in hf.attrs:
                raise KeyError(
                    f'{path}: external references must contain H5 attribute "class_name".'
                )

            class_name = hf.attrs["class_name"]
            if isinstance(class_name, bytes):
                class_name = class_name.decode()
            class_name = str(class_name)

            if only_classes is not None and class_name not in only_classes:
                continue

            reference_id = hf.attrs.get("reference_id", path.stem)
            if isinstance(reference_id, bytes):
                reference_id = reference_id.decode()

            ref = {"reference_id": str(reference_id),
                   "class_name": class_name,
                   "points_xyz": hf["points_xyz"][()].astype(np.float32),
                   "point_scores": hf["scores"][()].astype(np.float32) if "scores" in hf else None,
                   "mean_sim": float(hf.attrs.get("mean_sim", np.nan)),
                   "source_file": str(path)}
            if "tomo" in hf.attrs:
                tomo = hf.attrs["tomo"]
                if isinstance(tomo, bytes):
                    tomo = tomo.decode()
                ref["tomo"] = str(tomo)
            if "instance_id" in hf.attrs:
                ref["instance_id"] = int(hf.attrs["instance_id"])
            for key in ("center_z", "center_y", "center_x"):
                if key in hf.attrs:
                    ref[key] = float(hf.attrs[key])
            if "n_voxels" in hf.attrs:
                ref["n_voxels"] = int(hf.attrs["n_voxels"])
            refs.append(ref)

    if not refs:
        raise ValueError("No external references matched the requested class.")
    return refs


def choose_automatic_references(records, n_references, out_dir):
    refs = []
    by_class = {}
    for rec in records:
        by_class.setdefault(rec["class_name"], []).append(rec)

    for class_name, class_records in by_class.items():
        class_records = sorted(class_records, key=lambda x: x["mean_sim"], reverse=True)
        selected = class_records[:min(n_references, len(class_records))]

        print(f"\n{class_name}: automatic reference(s)")
        for rank, rec in enumerate(selected, start=1):
            ref_id = f"{class_name}_ref{rank:02d}_{rec['tomo']}_inst{rec['instance_id']}"
            ref = {"reference_id": ref_id,
                   "class_name": class_name,
                   "points_xyz": rec["points_xyz"],
                   "point_scores": rec["point_scores"],
                   "mean_sim": rec["mean_sim"],
                   "tomo": rec["tomo"],
                   "instance_id": rec["instance_id"],
                   "center_z": rec["center_z"],
                   "center_y": rec["center_y"],
                   "center_x": rec["center_x"],
                   "n_voxels": rec["n_voxels"],
                   "h5_path": rec["h5_path"],
                   "registration_bounds": rec["registration_bounds"],
                   "source_uid": rec["uid"]}
            refs.append(ref)
            print(f"  {rank}: {rec['uid']}  mean_sim={rec['mean_sim']:.5f}  "
                  f"points={len(rec['points_xyz'])}")

            save_reference(ref, Path(out_dir) / f"{ref_id}.h5")

    return refs


def prediction_file_for_tomo(predictions, tomo_name):
    predictions = Path(predictions)
    if predictions.is_file():
        return predictions

    direct = predictions / f"{tomo_name}_preds.h5"
    alternate = predictions / f"{tomo_name}.h5"
    if direct.is_file():
        return direct
    if alternate.is_file():
        return alternate
    raise FileNotFoundError(
        f'Could not find prediction H5 for tomogram "{tomo_name}" in {predictions}.'
    )


def load_similarity_refinement_volume(item, predictions, registration_crop_size,
                                      use_distance_mask=False,
                                      exclude_other_instances=False,
                                      similarity_threshold=0.1,
                                      downsample=1):
    class_name = item["class_name"]
    tomo = item.get("tomo")
    instance_id = item.get("instance_id")
    center_keys = ("center_z", "center_y", "center_x")
    if tomo is None or instance_id is None or not all(k in item for k in center_keys):
        raise ValueError(
            f'Reference/particle {item.get("reference_id", item.get("uid", "unknown"))} '
            'is missing tomo, instance_id, or center metadata required for similarity refinement.'
        )

    h5_path = item.get("h5_path")
    if h5_path is None:
        h5_path = prediction_file_for_tomo(predictions, tomo)

    center = np.array([item["center_z"], item["center_y"], item["center_x"]],
                      dtype=np.float64)
    crop_size = crop_size_for_class(registration_crop_size, class_name)

    with h5py.File(h5_path, "r") as hf:
        sim_key = f"sim_{class_name}"
        if sim_key not in hf:
            raise KeyError(f'{h5_path}: missing "{sim_key}".')
        if "instance_mask" not in hf:
            raise KeyError(f'{h5_path}: missing "instance_mask".')

        bbox = centered_crop_slices(center, crop_size, hf["instance_mask"].shape)
        sim = hf[sim_key][bbox].astype(np.float32, copy=False)
        valid = np.ones(sim.shape, dtype=bool)

        seg_key = "seg_mask" if "seg_mask" in hf else "seg_mask_raw"
        if exclude_other_instances:
            if seg_key not in hf:
                raise KeyError(f'{h5_path}: missing "seg_mask" and "seg_mask_raw".')
            class_names = decode_strings(hf.attrs["class_names"])
            if class_name not in class_names:
                raise ValueError(f'Class {class_name} not found in {h5_path}.')
            class_label = class_names.index(class_name) + 1
            other_filled = other_filled_instances_in_crop(
                hf, hf["instance_mask"], hf[seg_key], class_label, class_name,
                int(instance_id), bbox)
            valid &= ~other_filled

        if use_distance_mask:
            distance_key = f"distance_{class_name}"
            if distance_key not in hf:
                raise KeyError(
                    f'{h5_path}: use_distance_mask=True but "{distance_key}" is missing.'
                )
            valid &= hf[distance_key][bbox] > 0

    # Values <= threshold contribute nothing. Subtracting the threshold makes the
    # retained positive similarity the actual weight used by the local refinement.
    volume = np.maximum(sim - float(similarity_threshold), 0.0)
    volume[~valid] = 0.0

    offset = np.array([s.start for s in bbox], dtype=np.float64)
    center_local = center - offset

    downsample = max(1, int(downsample))
    if downsample > 1:
        volume = volume[::downsample, ::downsample, ::downsample]
        center_local = center_local / float(downsample)

    return volume.astype(np.float32, copy=False), center_local


def rotate_similarity_to_target(source_volume, source_center_zyx,
                                target_shape, target_center_zyx, R_xyz):
    R_zyx = xyz_rotation_to_zyx(R_xyz)
    matrix = R_zyx.T
    offset = (np.asarray(source_center_zyx, dtype=np.float64) -
              matrix @ np.asarray(target_center_zyx, dtype=np.float64))
    return affine_transform(
        source_volume,
        matrix=matrix,
        offset=offset,
        output_shape=tuple(int(v) for v in target_shape),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False).astype(np.float32, copy=False)


def similarity_cosine_score(target_volume, source_volume):
    target = np.asarray(target_volume, dtype=np.float32).reshape(-1)
    source = np.asarray(source_volume, dtype=np.float32).reshape(-1)
    denom = float(np.sqrt(np.dot(target, target) * np.dot(source, source)))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(target, source) / denom)


def refine_rotation_with_similarity(initial_R, source_volume, source_center_zyx,
                                    target_volume, target_center_zyx,
                                    angles_deg=(8.0, 4.0, 2.0, 1.0)):
    """Original SciPy/CPU similarity refinement."""
    best_R = np.asarray(initial_R, dtype=np.float64)
    rotated = rotate_similarity_to_target(
        source_volume, source_center_zyx, target_volume.shape,
        target_center_zyx, best_R)
    best_score = similarity_cosine_score(target_volume, rotated)

    for angle in angles_deg:
        angle = float(angle)
        if angle <= 0:
            continue
        stage_R = best_R
        stage_score = best_score
        for ax in (-angle, 0.0, angle):
            for ay in (-angle, 0.0, angle):
                for az in (-angle, 0.0, angle):
                    if ax == 0.0 and ay == 0.0 and az == 0.0:
                        continue
                    delta = Rotation.from_euler('xyz', [ax, ay, az], degrees=True).as_matrix()
                    candidate_R = delta @ best_R
                    candidate = rotate_similarity_to_target(
                        source_volume, source_center_zyx, target_volume.shape,
                        target_center_zyx, candidate_R)
                    score = similarity_cosine_score(target_volume, candidate)
                    if np.isfinite(score) and (not np.isfinite(stage_score) or score > stage_score):
                        stage_R = candidate_R
                        stage_score = score
        best_R = stage_R
        best_score = stage_score

    return best_R, best_score


def _torch_similarity_offsets(target_shape, target_center_zyx, device):
    """Output voxel offsets in ZYX order, centered exactly like scipy affine_transform."""
    D, H, W = [int(v) for v in target_shape]
    target_center = torch.as_tensor(
        np.asarray(target_center_zyx, dtype=np.float32),
        dtype=torch.float32, device=device)
    z = torch.arange(D, device=device, dtype=torch.float32) - target_center[0]
    y = torch.arange(H, device=device, dtype=torch.float32) - target_center[1]
    x = torch.arange(W, device=device, dtype=torch.float32) - target_center[2]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    return torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)


def _torch_rotate_similarity_batch(source_tensor, source_center_zyx, target_shape,
                                   rotations_xyz, offsets_zyx):
    """Rotate one source similarity volume for a batch of XYZ rotations.

    This reproduces rotate_similarity_to_target() using grid_sample.  The explicit
    validity mask is important: scipy.ndimage mode='constant' returns cval for any
    coordinate outside the image, whereas grid_sample otherwise interpolates with
    zero in the half-voxel border region.
    """
    device = source_tensor.device
    rotations_xyz = np.asarray(rotations_xyz, dtype=np.float32)
    if rotations_xyz.ndim == 2:
        rotations_xyz = rotations_xyz[None, ...]

    # P @ R_xyz @ P is equivalent to reversing both matrix axes.  With row
    # vectors, offsets @ R_zyx equals R_zyx.T @ offsets in the SciPy expression.
    rotations_zyx = np.ascontiguousarray(rotations_xyz[:, ::-1, ::-1])
    R = torch.from_numpy(rotations_zyx).to(device=device, dtype=torch.float32)
    source_center = torch.as_tensor(
        np.asarray(source_center_zyx, dtype=np.float32),
        device=device, dtype=torch.float32)
    coords = torch.matmul(offsets_zyx.unsqueeze(0), R) + source_center[None, None, :]

    Ds, Hs, Ws = [int(v) for v in source_tensor.shape[-3:]]
    gz = 2.0 * coords[..., 0] / float(max(Ds - 1, 1)) - 1.0
    gy = 2.0 * coords[..., 1] / float(max(Hs - 1, 1)) - 1.0
    gx = 2.0 * coords[..., 2] / float(max(Ws - 1, 1)) - 1.0

    D, H, W = [int(v) for v in target_shape]
    grid = torch.stack((gx, gy, gz), dim=-1).reshape(len(rotations_xyz), D, H, W, 3)

    source_batch = source_tensor.expand(len(rotations_xyz), -1, -1, -1, -1)
    rotated = F.grid_sample(
        source_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)[:, 0]

    valid = (
            (coords[..., 0] >= 0.0) & (coords[..., 0] <= float(Ds - 1)) &
            (coords[..., 1] >= 0.0) & (coords[..., 1] <= float(Hs - 1)) &
            (coords[..., 2] >= 0.0) & (coords[..., 2] <= float(Ws - 1))
    ).reshape(len(rotations_xyz), D, H, W)
    return rotated * valid.to(rotated.dtype)


def _torch_similarity_cosine_scores(target_tensor, candidate_batch):
    target_flat = target_tensor.reshape(1, -1)
    candidates_flat = candidate_batch.reshape(candidate_batch.shape[0], -1)
    numerator = (candidates_flat * target_flat).sum(dim=1)
    denominator = torch.sqrt(
        candidates_flat.square().sum(dim=1) * target_flat.square().sum(dim=1))
    scores = numerator / denominator.clamp_min(1e-12)
    nan_values = torch.full_like(scores, float('nan'))
    return torch.where(denominator > 1e-12, scores, nan_values)


@torch.inference_mode()
def refine_rotation_with_similarity_gpu(initial_R, source_volume, source_center_zyx,
                                        target_volume, target_center_zyx,
                                        angles_deg=(8.0, 4.0, 2.0, 1.0),
                                        device='cuda:0', batch_size=26):
    """GPU equivalent of refine_rotation_with_similarity using batched grid_sample.

    Each [8,4,2,1]-degree stage still evaluates the same 26 neighboring rotations
    around the current best orientation. Only the interpolation and cosine scoring
    are batched on the GPU; the search itself is unchanged.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        raise RuntimeError('GPU similarity refinement requires a CUDA device.')

    if torch.is_tensor(source_volume):
        source_tensor = source_volume.to(device=device, dtype=torch.float32)
    else:
        source_tensor = torch.as_tensor(
            np.asarray(source_volume, dtype=np.float32), device=device, dtype=torch.float32)
    if source_tensor.ndim == 3:
        source_tensor = source_tensor[None, None]
    elif source_tensor.ndim != 5:
        raise ValueError(f'Unexpected source similarity tensor shape: {tuple(source_tensor.shape)}')

    if torch.is_tensor(target_volume):
        target_tensor = target_volume.to(device=device, dtype=torch.float32)
    else:
        target_tensor = torch.as_tensor(
            np.asarray(target_volume, dtype=np.float32), device=device, dtype=torch.float32)
    if target_tensor.ndim == 5:
        target_tensor = target_tensor[0, 0]
    elif target_tensor.ndim == 4:
        target_tensor = target_tensor[0]
    elif target_tensor.ndim != 3:
        raise ValueError(f'Unexpected target similarity tensor shape: {tuple(target_tensor.shape)}')

    target_shape = tuple(int(v) for v in target_tensor.shape)
    offsets = _torch_similarity_offsets(target_shape, target_center_zyx, device)
    batch_size = max(1, int(batch_size))

    def score_rotations(rotations):
        score_parts = []
        for start in range(0, len(rotations), batch_size):
            chunk = rotations[start:start + batch_size]
            rotated = _torch_rotate_similarity_batch(
                source_tensor, source_center_zyx, target_shape, chunk, offsets)
            score_parts.append(_torch_similarity_cosine_scores(target_tensor, rotated))
        return torch.cat(score_parts, dim=0)

    best_R = np.asarray(initial_R, dtype=np.float64)
    best_score = float(score_rotations([best_R])[0].item())

    for angle in angles_deg:
        angle = float(angle)
        if angle <= 0:
            continue

        candidates = []
        for ax in (-angle, 0.0, angle):
            for ay in (-angle, 0.0, angle):
                for az in (-angle, 0.0, angle):
                    if ax == 0.0 and ay == 0.0 and az == 0.0:
                        continue
                    delta = Rotation.from_euler(
                        'xyz', [ax, ay, az], degrees=True).as_matrix()
                    candidates.append(delta @ best_R)

        scores = score_rotations(candidates).detach().cpu().numpy()
        stage_R = best_R
        stage_score = best_score
        # Preserve the CPU implementation's strict '>' rule and candidate order.
        for candidate_R, score in zip(candidates, scores):
            score = float(score)
            if np.isfinite(score) and (not np.isfinite(stage_score) or score > stage_score):
                stage_R = candidate_R
                stage_score = score

        best_R = stage_R
        best_score = stage_score

    return best_R, best_score


def stable_registration_seed(base_seed, particle_uid, reference_id):
    """Stable per particle-reference seed, independent of worker scheduling."""
    if base_seed is None:
        return None
    token = f"{particle_uid}|{reference_id}".encode("utf-8")
    return (int(base_seed) + int(zlib.crc32(token))) & 0x7FFFFFFF


def seed_open3d(seed):
    """Seed Open3D's RNG when the installed version exposes utility.random.seed."""
    if seed is None:
        return
    try:
        o3d.utility.random.seed(int(seed))
    except (AttributeError, TypeError):
        pass


def prepare_point_cloud(points_xyz, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(float(voxel_size))

    if len(pcd.points) < 10:
        raise ValueError("Too few points after downsampling.")

    normal_radius = max(float(voxel_size) * 2.0, 1.0)
    feature_radius = max(float(voxel_size) * 5.0, 2.0)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                           o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius,
                                                                                                max_nn=100))

    return pcd, fpfh


def resolve_ransac_device(device):
    device = str(device or "auto").lower()
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f'GPU RANSAC requested on "{device}", but CUDA is not available.')
    return torch.device(device)


def normalize_ransac_devices(value):
    """Normalize one or more CUDA device specifications to strings like cuda:0."""
    if value is None:
        return []
    if isinstance(value, (str, int)):
        value = [value]

    devices = []
    for item in value:
        if isinstance(item, int):
            device = f"cuda:{int(item)}"
        else:
            item = str(item).strip().lower()
            if item.isdigit():
                device = f"cuda:{int(item)}"
            elif item == "cuda":
                device = "cuda:0"
            else:
                device = item
        resolved = resolve_ransac_device(device)
        if resolved.type != "cuda":
            raise ValueError(f"GPU RANSAC requires CUDA devices, got {device!r}.")
        canonical = f"cuda:{0 if resolved.index is None else int(resolved.index)}"
        if canonical not in devices:
            devices.append(canonical)
    return devices


def feature_correspondences(source_fpfh, target_fpfh, mutual_filter=False):
    # Keep Open3D's FPFH correspondence construction so the GPU path differs
    # only in the RANSAC hypothesis generation/validation step.
    correspondences = o3d.pipelines.registration.correspondences_from_features(
        source_fpfh, target_fpfh, mutual_filter=bool(mutual_filter))
    correspondences = np.asarray(correspondences, dtype=np.int64)
    if correspondences.ndim != 2 or correspondences.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected FPFH correspondence shape: {correspondences.shape}")
    return correspondences


def batched_rigid_alignment(source_samples, target_samples):
    """Batched rigid Kabsch alignment for row-vector coordinates.

    Returns R, t such that target ~= source @ R + t.  R is therefore the
    transpose of the 3x3 rotation block used by Open3D's column-vector 4x4
    transformation.  float64 is intentional here to stay close to Open3D's
    point-to-point estimator; nearest-neighbour validation is float32.
    """
    source_samples = source_samples.to(dtype=torch.float64)
    target_samples = target_samples.to(dtype=torch.float64)

    source_mean = source_samples.mean(dim=1, keepdim=True)
    target_mean = target_samples.mean(dim=1, keepdim=True)
    source_centered = source_samples - source_mean
    target_centered = target_samples - target_mean

    covariance = torch.bmm(
        source_centered.transpose(1, 2), target_centered)
    U, _, Vh = torch.linalg.svd(covariance, full_matrices=False)
    R = torch.bmm(U, Vh)

    reflection = torch.linalg.det(R) < 0
    if reflection.any():
        U = U.clone()
        U[reflection, :, -1] *= -1.0
        R = torch.bmm(U, Vh)

    t = target_mean[:, 0] - torch.bmm(source_mean, R)[:, 0]
    return R, t


def ransac_edge_length_mask(source_samples, target_samples, threshold):
    if threshold <= 0:
        return torch.ones(
            source_samples.shape[0], dtype=torch.bool,
            device=source_samples.device)

    source_edges = torch.cdist(source_samples, source_samples)
    target_edges = torch.cdist(target_samples, target_samples)
    n = source_samples.shape[1]
    upper = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=source_samples.device),
        diagonal=1)
    valid = ((source_edges >= float(threshold) * target_edges) &
             (target_edges >= float(threshold) * source_edges))
    return valid[:, upper].all(dim=1)


def ransac_sample_distance_mask(source_samples, target_samples, R, t,
                                max_distance):
    transformed = torch.bmm(source_samples, R) + t[:, None, :]
    distances = torch.linalg.vector_norm(target_samples - transformed, dim=-1)
    return (distances <= float(max_distance)).all(dim=1)


def nearest_neighbor_squared_distances(source_transformed, target_points):
    """Batched source->target NN squared distances using PyTorch3D."""
    if knn_points is None:
        raise ImportError(
            'backend="gpu" requires PyTorch3D. '
            'Install a PyTorch3D build compatible with your PyTorch/CUDA version.')

    # PyTorch3D KNN is fastest and most broadly supported in float32.  The
    # hypotheses themselves were fitted in float64 above.
    source_knn = source_transformed.to(dtype=torch.float32).contiguous()
    target_knn = target_points.to(dtype=torch.float32)
    B = source_knn.shape[0]
    target_batch = target_knn.unsqueeze(0).expand(B, -1, -1).contiguous()
    return knn_points(
        source_knn, target_batch, K=1, return_nn=False).dists[..., 0]


def validate_ransac_hypotheses(source_points, target_points, R, t,
                               max_distance, validation_batch_size=64):
    fitness_chunks = []
    rmse_chunks = []
    threshold_sq = float(max_distance) ** 2

    for start in range(0, R.shape[0], int(validation_batch_size)):
        stop = min(start + int(validation_batch_size), R.shape[0])
        R_batch = R[start:stop]
        t_batch = t[start:stop]
        B = stop - start

        source_batch = source_points.unsqueeze(0).expand(B, -1, -1)
        transformed = torch.bmm(source_batch, R_batch) + t_batch[:, None, :]
        distances_sq = nearest_neighbor_squared_distances(
            transformed, target_points)

        inliers = distances_sq <= threshold_sq
        counts = inliers.sum(dim=1)
        fitness = counts.to(torch.float32) / float(source_points.shape[0])
        rmse = torch.zeros_like(fitness)
        nonzero = counts > 0
        if nonzero.any():
            rmse[nonzero] = torch.sqrt(
                (distances_sq[nonzero] * inliers[nonzero]).sum(dim=1) /
                counts[nonzero].to(distances_sq.dtype))

        fitness_chunks.append(fitness)
        rmse_chunks.append(rmse)

    return torch.cat(fitness_chunks), torch.cat(rmse_chunks)


def correspondence_inlier_ratio(source_corr, target_corr, R, t, max_distance):
    transformed = source_corr @ R + t
    distances = torch.linalg.vector_norm(target_corr - transformed, dim=-1)
    return float((distances < float(max_distance)).float().mean().item())


def better_ransac_candidate(fitness, rmse, best_fitness, best_rmse):
    return (fitness > best_fitness or
            (fitness == best_fitness and rmse < best_rmse))


@torch.inference_mode()
def pytorch3d_batched_ransac(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        max_correspondence_distance, ransac_max_iterations=50000,
        ransac_confidence=0.999, ransac_n=3,
        ransac_edge_length=0.9, ransac_mutual_filter=False,
        device="auto", batch_size=4096, validation_batch_size=64, seed=0):
    """GPU RANSAC which mirrors the Open3D feature-matching RANSAC flow.

    Open3D still supplies voxel downsampling, normals, FPFH correspondences and
    the final GICP.  PyTorch/PyTorch3D are used only for batched RANSAC fitting
    and nearest-neighbour validation, which is the expensive global step.
    """
    device = resolve_ransac_device(device)
    if device.type != "cuda":
        raise RuntimeError(
            'backend="gpu" requires a CUDA device.')
    if knn_points is None:
        raise ImportError(
            'backend="gpu" requires PyTorch3D.')

    correspondences = feature_correspondences(
        source_fpfh, target_fpfh, mutual_filter=ransac_mutual_filter)
    if len(correspondences) < int(ransac_n):
        raise RuntimeError(
            f"Only {len(correspondences)} FPFH correspondences are available, "
            f"but ransac_n={ransac_n}.")

    source_points = torch.as_tensor(
        np.asarray(source_pcd.points), dtype=torch.float64, device=device)
    target_points = torch.as_tensor(
        np.asarray(target_pcd.points), dtype=torch.float64, device=device)
    corr = torch.as_tensor(correspondences, dtype=torch.long, device=device)
    source_corr = source_points[corr[:, 0]]
    target_corr = target_points[corr[:, 1]]

    generator = torch.Generator(device=device)
    if seed is None:
        seed = int(torch.seed() & 0x7FFFFFFF)
    generator.manual_seed(int(seed))

    max_iterations = int(ransac_max_iterations)
    batch_size = max(1, int(batch_size))
    validation_batch_size = max(1, int(validation_batch_size))
    estimated_iterations = max_iterations
    iterations_done = 0
    validations = 0

    best_R = None
    best_t = None
    best_fitness = 0.0
    best_rmse = 0.0

    while iterations_done < min(max_iterations, estimated_iterations):
        remaining = min(max_iterations, estimated_iterations) - iterations_done
        current_batch = min(batch_size, max(1, remaining))

        # Open3D samples each correspondence independently, i.e. with
        # replacement.  Keeping that behaviour minimizes algorithmic changes.
        sampled = torch.randint(
            0, len(correspondences),
            (current_batch, int(ransac_n)),
            generator=generator, device=device)
        src_samples = source_corr[sampled]
        tgt_samples = target_corr[sampled]

        R, t = batched_rigid_alignment(src_samples, tgt_samples)
        finite = (torch.isfinite(R).reshape(R.shape[0], -1).all(dim=1) &
                  torch.isfinite(t).reshape(t.shape[0], -1).all(dim=1))
        if finite.any():
            R = R[finite]
            t = t[finite]
            src_valid = src_samples[finite]
            tgt_valid = tgt_samples[finite]

            edge_valid = ransac_edge_length_mask(
                src_valid, tgt_valid, float(ransac_edge_length))
            if edge_valid.any():
                R = R[edge_valid]
                t = t[edge_valid]
                src_valid = src_valid[edge_valid]
                tgt_valid = tgt_valid[edge_valid]

                distance_valid = ransac_sample_distance_mask(
                    src_valid, tgt_valid, R, t,
                    max_correspondence_distance)
                if distance_valid.any():
                    R = R[distance_valid]
                    t = t[distance_valid]
                    validations += int(R.shape[0])

                    fitness, rmse = validate_ransac_hypotheses(
                        source_points, target_points, R, t,
                        max_correspondence_distance,
                        validation_batch_size=validation_batch_size)

                    top_fitness = fitness.max()
                    tied = torch.nonzero(
                        fitness == top_fitness,
                        as_tuple=False).flatten()
                    batch_best = int(
                        tied[torch.argmin(rmse[tied])].item())
                    candidate_fitness = float(fitness[batch_best].item())
                    candidate_rmse = float(rmse[batch_best].item())

                    if better_ransac_candidate(
                            candidate_fitness, candidate_rmse,
                            best_fitness, best_rmse):
                        best_fitness = candidate_fitness
                        best_rmse = candidate_rmse
                        best_R = R[batch_best].clone()
                        best_t = t[batch_best].clone()

                        # Match Open3D's confidence-based early termination:
                        # estimate the inlier ratio on the FPFH correspondence
                        # set under the current best transform.
                        inlier_ratio = correspondence_inlier_ratio(
                            source_corr, target_corr,
                            best_R, best_t,
                            max_correspondence_distance)
                        success_prob = inlier_ratio ** int(ransac_n)
                        if 0.0 < success_prob < 1.0:
                            estimate = (
                                    np.log(1.0 - float(ransac_confidence)) /
                                    np.log(1.0 - success_prob))
                            if np.isfinite(estimate) and estimate >= 0:
                                estimated_iterations = min(
                                    estimated_iterations,
                                    max(1, int(np.ceil(estimate))))
                        elif success_prob >= 1.0:
                            estimated_iterations = 1

        iterations_done += current_batch

    if best_R is None:
        return (np.eye(4, dtype=np.float64),
                0.0, 0.0, iterations_done, validations)

    # Convert row-vector R into Open3D's column-vector 4x4 convention.
    T_open3d = np.eye(4, dtype=np.float64)
    T_open3d[:3, :3] = best_R.detach().cpu().numpy().T
    T_open3d[:3, 3] = best_t.detach().cpu().numpy()
    return (T_open3d, best_fitness, best_rmse,
            iterations_done, validations)


def register_prepared(source_pcd, target_pcd, source_fpfh, target_fpfh,
                      voxel_size, ransac_max_iterations=50000,
                      ransac_confidence=0.999, ransac_n=4,
                      ransac_edge_length=0.9, ransac_mutual_filter=False,
                      ransac_backend="open3d", ransac_device="auto",
                      ransac_batch_size=4096,
                      ransac_validation_batch_size=64, ransac_seed=0,
                      refine_registration=True, refine_max_iterations=30,
                      refine_max_correspondence_factor=2.0):
    max_corr = max(float(voxel_size) * 1.5, 1.0)
    backend = str(ransac_backend).lower()
    if backend == "gpu":
        backend = "pytorch3d"

    if backend == "pytorch3d":
        T, ransac_fitness, ransac_rmse, _, _ = pytorch3d_batched_ransac(
            source_pcd, target_pcd, source_fpfh, target_fpfh, max_corr,
            ransac_max_iterations=ransac_max_iterations,
            ransac_confidence=ransac_confidence,
            ransac_n=ransac_n,
            ransac_edge_length=ransac_edge_length,
            ransac_mutual_filter=ransac_mutual_filter,
            device=ransac_device,
            batch_size=ransac_batch_size,
            validation_batch_size=ransac_validation_batch_size,
            seed=ransac_seed)
    elif backend == "open3d":
        global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd,
            target_pcd,
            source_fpfh,
            target_fpfh,
            mutual_filter=bool(ransac_mutual_filter),
            max_correspondence_distance=max_corr,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=int(ransac_n),
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    float(ransac_edge_length)),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                int(ransac_max_iterations), float(ransac_confidence)))

        T = np.asarray(global_result.transformation, dtype=np.float64)
        ransac_fitness = float(global_result.fitness)
        ransac_rmse = float(global_result.inlier_rmse)
    else:
        raise ValueError(
            'ransac_backend must be "open3d", "pytorch3d", or "gpu".')

    if not refine_registration:
        return T, ransac_fitness, ransac_rmse, ransac_fitness, ransac_rmse

    # Keep exactly the same Open3D GICP refinement as the trusted CPU path.
    refine_corr = max(float(voxel_size) * float(refine_max_correspondence_factor), 1.0)
    refined = o3d.pipelines.registration.registration_generalized_icp(
        source_pcd,
        target_pcd,
        refine_corr,
        T,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(refine_max_iterations)))

    return (np.asarray(refined.transformation, dtype=np.float64),
            float(refined.fitness),
            float(refined.inlier_rmse),
            ransac_fitness,
            ransac_rmse)


def make_reference_frames(references, voxel_size, ransac_max_iterations=50000,
                          ransac_confidence=0.999, ransac_n=4,
                          ransac_edge_length=0.9, ransac_mutual_filter=False,
                          ransac_backend="open3d", ransac_device="auto",
                          ransac_batch_size=4096,
                          ransac_validation_batch_size=64, ransac_seed=0,
                          refine_registration=True, refine_max_iterations=30,
                          refine_max_correspondence_factor=2.0):
    """
    *Prepare references and align secondary references to the highest-ranked*
    *reference of the same class. Returns refs grouped by class.*
    """
    by_class = {}
    for ref in references:
        by_class.setdefault(ref["class_name"], []).append(ref)

    for class_name, refs in by_class.items():
        # Keep provided order. In automatic mode this is descending mean_sim.
        for ref in refs:
            ref["pcd"], ref["fpfh"] = prepare_point_cloud(
                ref["points_xyz"], voxel_size
            )

        master = refs[0]
        master["to_master"] = np.eye(4, dtype=np.float64)
        print(f"\n{class_name}: master reference = {master['reference_id']}")

        for ref in refs[1:]:
            T, fitness, rmse, _, _ = register_prepared(
                ref["pcd"], master["pcd"],
                ref["fpfh"], master["fpfh"],
                voxel_size,
                ransac_max_iterations=ransac_max_iterations,
                ransac_confidence=ransac_confidence,
                ransac_n=ransac_n,
                ransac_edge_length=ransac_edge_length,
                ransac_mutual_filter=ransac_mutual_filter,
                ransac_backend=ransac_backend,
                ransac_device=ransac_device,
                ransac_batch_size=ransac_batch_size,
                ransac_validation_batch_size=ransac_validation_batch_size,
                ransac_seed=stable_registration_seed(
                    ransac_seed, ref["reference_id"], master["reference_id"]),
                refine_registration=refine_registration,
                refine_max_iterations=refine_max_iterations,
                refine_max_correspondence_factor=refine_max_correspondence_factor)
            ref["to_master"] = T
            print(f"  reference {ref['reference_id']} -> master: "
                  f"fitness={fitness:.3f}, rmse={rmse:.3f}")

    return by_class


def rotation_to_zyz_degrees(R):
    # Upper-case sequence = intrinsic rotations in scipy.
    # Gimbal lock only makes the Euler-angle decomposition non-unique;
    # the saved rotation matrix itself remains valid.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected.*",
                category=UserWarning)
            angles = Rotation.from_matrix(R).as_euler("ZYZ", degrees=True)
        return tuple(float(v) for v in angles)
    except Exception:
        return (np.nan, np.nan, np.nan)


def determine_orientations(records, refs_by_class, voxel_size,
                           ransac_max_iterations=50000, ransac_confidence=0.999,
                           ransac_n=4, ransac_edge_length=0.9,
                           ransac_mutual_filter=False,
                           ransac_backend="open3d", ransac_device="auto",
                           ransac_batch_size=4096,
                           ransac_validation_batch_size=64,
                           refine_registration=True, refine_max_iterations=30,
                           refine_max_correspondence_factor=2.0,
                           match_trim_fraction=0.85, predictions=None,
                           registration_crop_size=64, use_distance_mask=False,
                           exclude_other_instances=False, similarity_threshold=0.1,
                           sim_refine_angles_deg=(8.0, 4.0, 2.0, 1.0),
                           sim_refine_downsample=2, sim_refine_batch_size=26,
                           ransac_seed=0, show_progress=True):
    rows = []

    # Cache prepared particle clouds once, then reuse against all references.
    iterator = tqdm(records, desc="Aligning particles", unit="particle", disable=not show_progress)
    for i, rec in enumerate(iterator, start=1):
        class_name = rec["class_name"]
        refs = refs_by_class.get(class_name, [])
        if not refs:
            print(f"WARNING: no reference for class {class_name}; skipping {rec['uid']}")
            continue

        try:
            source_pcd, source_fpfh = prepare_point_cloud(
                rec["points_xyz"], voxel_size
            )
        except ValueError as exc:
            print(f"WARNING: {rec['uid']}: {exc}")
            continue

        best = None
        source_sim = None
        source_sim_center = None
        source_sim_gpu = None
        use_gpu_similarity = str(ransac_backend).lower() in ("pytorch3d", "gpu")
        sim_device = resolve_ransac_device(ransac_device) if use_gpu_similarity else None
        if sim_refine_angles_deg:
            try:
                source_sim, source_sim_center = load_similarity_refinement_volume(
                    rec, predictions, registration_crop_size,
                    use_distance_mask=use_distance_mask,
                    exclude_other_instances=exclude_other_instances,
                    similarity_threshold=similarity_threshold,
                    downsample=sim_refine_downsample)
                if use_gpu_similarity:
                    source_sim_gpu = torch.as_tensor(
                        source_sim, dtype=torch.float32, device=sim_device)[None, None]
            except Exception as exc:
                print(f"WARNING: similarity refinement input failed for {rec['uid']}: {exc}")

        for ref in refs:
            # Automatic reference aligned to itself: exact identity is preferable.
            if rec["uid"] == ref.get("source_uid"):
                T_particle_to_ref = np.eye(4, dtype=np.float64)
                fitness = 1.0
                rmse = 0.0
                ransac_fitness = 1.0
                ransac_rmse = 0.0
            else:
                try:
                    pair_seed = stable_registration_seed(
                        ransac_seed, rec["uid"], ref["reference_id"])
                    if str(ransac_backend).lower() == "open3d":
                        seed_open3d(pair_seed)
                    T_particle_to_ref, fitness, rmse, ransac_fitness, ransac_rmse = register_prepared(
                        source_pcd,
                        ref["pcd"],
                        source_fpfh,
                        ref["fpfh"],
                        voxel_size,
                        ransac_max_iterations=ransac_max_iterations,
                        ransac_confidence=ransac_confidence,
                        ransac_n=ransac_n,
                        ransac_edge_length=ransac_edge_length,
                        ransac_mutual_filter=ransac_mutual_filter,
                        ransac_backend=ransac_backend,
                        ransac_device=ransac_device,
                        ransac_batch_size=ransac_batch_size,
                        ransac_validation_batch_size=ransac_validation_batch_size,
                        ransac_seed=pair_seed,
                        refine_registration=refine_registration,
                        refine_max_iterations=refine_max_iterations,
                        refine_max_correspondence_factor=refine_max_correspondence_factor)
                except Exception as exc:
                    print(
                        f"WARNING: registration failed for {rec['uid']} -> "
                        f"{ref['reference_id']}: {exc}"
                    )
                    continue

            # Preserve the rigid transform as returned by RANSAC/GICP before the
            # later similarity step changes rotation only. The translation is
            # geometrically consistent with this registration rotation.
            T_registration_particle_to_ref = T_particle_to_ref.copy()
            T_registration_particle_to_master = (
                    ref["to_master"] @ T_registration_particle_to_ref)
            registration_R_to_master = np.asarray(
                T_registration_particle_to_master[:3, :3], dtype=np.float64)
            registration_t_to_master = np.asarray(
                T_registration_particle_to_master[:3, 3], dtype=np.float64)
            registration_center_shift_xyz = (
                    -registration_R_to_master.T @ registration_t_to_master)

            transformed_points_xyz = transform_points(rec["points_xyz"], T_particle_to_ref)
            match_distance = trimmed_source_to_target_distance(
                transformed_points_xyz, ref["points_xyz"], trim_fraction=match_trim_fraction)
            point_match_score = distance_to_match_score(match_distance, voxel_size)

            sim_refine_score = np.nan
            if source_sim is not None:
                try:
                    if "sim_refine_volume" not in ref:
                        ref["sim_refine_volume"], ref["sim_refine_center"] = (
                            load_similarity_refinement_volume(
                                ref, predictions, registration_crop_size,
                                use_distance_mask=use_distance_mask,
                                exclude_other_instances=exclude_other_instances,
                                similarity_threshold=similarity_threshold,
                                downsample=sim_refine_downsample))
                    if use_gpu_similarity:
                        cache_key = "_sim_refine_volume_gpu"
                        cache_device_key = "_sim_refine_volume_gpu_device"
                        device_name = str(sim_device)
                        if (cache_key not in ref or
                                ref.get(cache_device_key) != device_name):
                            ref[cache_key] = torch.as_tensor(
                                ref["sim_refine_volume"], dtype=torch.float32,
                                device=sim_device)
                            ref[cache_device_key] = device_name
                        refined_R, sim_refine_score = refine_rotation_with_similarity_gpu(
                            T_particle_to_ref[:3, :3],
                            source_sim_gpu, source_sim_center,
                            ref[cache_key], ref["sim_refine_center"],
                            angles_deg=sim_refine_angles_deg,
                            device=sim_device,
                            batch_size=sim_refine_batch_size)
                    else:
                        refined_R, sim_refine_score = refine_rotation_with_similarity(
                            T_particle_to_ref[:3, :3],
                            source_sim, source_sim_center,
                            ref["sim_refine_volume"], ref["sim_refine_center"],
                            angles_deg=sim_refine_angles_deg)
                    T_particle_to_ref = T_particle_to_ref.copy()
                    T_particle_to_ref[:3, :3] = refined_R
                except Exception as exc:
                    print(
                        f"WARNING: similarity refinement failed for {rec['uid']} -> "
                        f"{ref['reference_id']}: {exc}"
                    )

            # particle -> chosen ref -> master ref
            T_particle_to_master = ref["to_master"] @ T_particle_to_ref
            match_score = (sim_refine_score if np.isfinite(sim_refine_score)
                           else point_match_score)

            candidate = {"ref": ref,
                         "T": T_particle_to_master,
                         "T_particle_to_ref": T_particle_to_ref,
                         "fitness": fitness,
                         "rmse": rmse,
                         "ransac_fitness": ransac_fitness,
                         "ransac_rmse": ransac_rmse,
                         "match_distance": match_distance,
                         "point_match_score": point_match_score,
                         "sim_refine_score": sim_refine_score,
                         "match_score": match_score,
                         "center_shift_xyz": registration_center_shift_xyz,
                         "registration_translation_xyz": registration_t_to_master}

            if best is None:
                best = candidate
            elif candidate["match_score"] > best["match_score"]:
                best = candidate
            elif np.isclose(candidate["match_score"], best["match_score"]) and candidate["fitness"] > best["fitness"]:
                best = candidate
            elif (np.isclose(candidate["match_score"], best["match_score"]) and
                  np.isclose(candidate["fitness"], best["fitness"]) and
                  candidate["rmse"] < best["rmse"]):
                best = candidate

        if best is None:
            continue

        # Final orientation may include the local similarity rotation refinement,
        # but the center correction comes from the RANSAC/GICP rigid transform
        # saved before that rotation-only refinement. This keeps R and t from the
        # registration step geometrically consistent.
        R = np.asarray(best["T"][:3, :3], dtype=np.float64)

        # Remove tiny numerical deviations from SO(3).
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        original_center_xyz = np.array(
            [rec["center_x"], rec["center_y"], rec["center_z"]],
            dtype=np.float64)
        center_shift_xyz = np.asarray(best["center_shift_xyz"], dtype=np.float64)
        translation_xyz = np.asarray(
            best["registration_translation_xyz"], dtype=np.float64)
        refined_center_xyz = original_center_xyz + center_shift_xyz

        z1, y, z2 = rotation_to_zyz_degrees(R)

        row = {"tomo": rec["tomo"],
               "instance_id": rec["instance_id"],
               "class": class_name,
               "center_x": float(refined_center_xyz[0]),
               "center_y": float(refined_center_xyz[1]),
               "center_z": float(refined_center_xyz[2]),
               "original_center_x": float(original_center_xyz[0]),
               "original_center_y": float(original_center_xyz[1]),
               "original_center_z": float(original_center_xyz[2]),
               "center_shift_x": float(center_shift_xyz[0]),
               "center_shift_y": float(center_shift_xyz[1]),
               "center_shift_z": float(center_shift_xyz[2]),
               "registration_translation_x": float(translation_xyz[0]),
               "registration_translation_y": float(translation_xyz[1]),
               "registration_translation_z": float(translation_xyz[2]),
               "n_voxels": rec["n_voxels"],
               "mean_sim": rec["mean_sim"],
               "reference_id": best["ref"]["reference_id"],
               "master_reference_id": refs[0]["reference_id"],
               "ransac_fitness": best["ransac_fitness"],
               "ransac_rmse": best["ransac_rmse"],
               "registration_fitness": best["fitness"],
               "registration_rmse": best["rmse"],
               "match_distance": best["match_distance"],
               "point_match_score": best["point_match_score"],
               "sim_refine_score": best["sim_refine_score"],
               "match_score": best["match_score"],
               "zyz_z1_deg": z1,
               "zyz_y_deg": y,
               "zyz_z2_deg": z2}

        for r in range(3):
            for c in range(3):
                row[f"r{r}{c}"] = float(R[r, c])

        rows.append(row)

    return pd.DataFrame(rows)


_WORKER_REFS_BY_CLASS = None
_WORKER_ORIENTATION_KWARGS = None


def _reference_payload_for_workers(refs_by_class):
    """Strip non-picklable Open3D objects; workers rebuild them once."""
    payload = {}
    for class_name, refs in refs_by_class.items():
        payload[class_name] = []
        for ref in refs:
            worker_ref = {
                key: value for key, value in ref.items()
                if key not in ("pcd", "fpfh", "sim_refine_volume", "sim_refine_center",
                               "_sim_refine_volume_gpu", "_sim_refine_volume_gpu_device")
            }
            payload[class_name].append(worker_ref)
    return payload


def _orientation_worker_init(reference_payload, orientation_kwargs):
    global _WORKER_REFS_BY_CLASS, _WORKER_ORIENTATION_KWARGS

    voxel_size = float(orientation_kwargs["voxel_size"])
    refs_by_class = {}

    for class_name, refs in reference_payload.items():
        refs_by_class[class_name] = []
        for ref_data in refs:
            ref = dict(ref_data)
            ref["pcd"], ref["fpfh"] = prepare_point_cloud(
                ref["points_xyz"], voxel_size)
            refs_by_class[class_name].append(ref)

    _WORKER_REFS_BY_CLASS = refs_by_class
    _WORKER_ORIENTATION_KWARGS = dict(orientation_kwargs)
    _WORKER_ORIENTATION_KWARGS["show_progress"] = False

    backend = str(_WORKER_ORIENTATION_KWARGS.get("ransac_backend", "open3d")).lower()
    if backend in ("pytorch3d", "gpu"):
        device = resolve_ransac_device(_WORKER_ORIENTATION_KWARGS.get("ransac_device", "auto"))
        torch.cuda.set_device(0 if device.index is None else int(device.index))


def _orientation_worker(index_and_record):
    index, rec = index_and_record
    result = determine_orientations(
        [rec], _WORKER_REFS_BY_CLASS, **_WORKER_ORIENTATION_KWARGS)
    if len(result) == 0:
        return index, None
    return index, result.iloc[0].to_dict()


def determine_orientations_multiprocess(
        records, refs_by_class, cpu_workers=1, gpu_devices=None, **orientation_kwargs):
    """Parallel wrapper. CPU backend uses CPU workers; GPU backend uses one process per GPU."""
    cpu_workers = int(cpu_workers)
    backend = str(orientation_kwargs.get("ransac_backend", "open3d")).lower()
    if backend == "gpu":
        backend = "pytorch3d"

    # GPU path: assign a disjoint particle shard to one dedicated process per GPU.
    # This avoids multiple workers contending for the same CUDA context while keeping
    # the trusted per-particle registration/refinement routine unchanged.
    if backend == "pytorch3d":
        devices = normalize_ransac_devices(gpu_devices)
        if not devices:
            devices = normalize_ransac_devices(
                [orientation_kwargs.get("ransac_device", "auto")])

        if len(devices) == 1:
            kwargs = dict(orientation_kwargs)
            kwargs["ransac_device"] = devices[0]
            torch.cuda.set_device(torch.device(devices[0]).index or 0)
            return determine_orientations(records, refs_by_class, **kwargs)

        reference_payload = _reference_payload_for_workers(refs_by_class)
        rows = [None] * len(records)
        context = mp.get_context("spawn")
        executors = []
        futures = []

        try:
            for device in devices:
                worker_kwargs = dict(orientation_kwargs)
                worker_kwargs["ransac_device"] = device
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                    initializer=_orientation_worker_init,
                    initargs=(reference_payload, worker_kwargs))
                executors.append(executor)

            # Round-robin sharding is deterministic and keeps all GPUs busy when
            # particles have broadly similar registration cost.
            for index, rec in enumerate(records):
                executor = executors[index % len(executors)]
                futures.append(executor.submit(_orientation_worker, (index, rec)))

            for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Aligning particles ({len(devices)} GPUs)",
                    unit="particle"):
                index, row = future.result()
                rows[index] = row
        finally:
            for executor in executors:
                executor.shutdown(wait=True)

        return pd.DataFrame([row for row in rows if row is not None])

    # Trusted Open3D CPU path.
    if cpu_workers <= 1:
        return determine_orientations(
            records, refs_by_class, **orientation_kwargs)

    reference_payload = _reference_payload_for_workers(refs_by_class)
    rows = [None] * len(records)

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
            max_workers=cpu_workers,
            mp_context=context,
            initializer=_orientation_worker_init,
            initargs=(reference_payload, orientation_kwargs)) as executor:

        futures = [
            executor.submit(_orientation_worker, (index, rec))
            for index, rec in enumerate(records)
        ]

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Aligning particles",
                unit="particle"):
            index, row = future.result()
            rows[index] = row

    return pd.DataFrame([row for row in rows if row is not None])


def load_orientation_config(config_file_path):
    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    orient_cfg = cfg.get("initial_orientations") or {}
    if not isinstance(orient_cfg, dict):
        raise ValueError("initial_orientations must be a mapping in the config file.")

    if "prediction_folder" not in cfg:
        raise KeyError("Config is missing top-level prediction_folder.")

    prediction_path = orient_cfg.get("predictions", cfg["prediction_folder"])
    output_folder = cfg.get("output_folder", cfg["prediction_folder"])
    output_file = orient_cfg.get("output_file", "initial_orientations.csv")
    output_path = orient_cfg.get("output", os.path.join(output_folder, output_file))

    class_names = orient_cfg.get("class_names", None)
    class_name = orient_cfg.get("class_name", None)
    if class_names is not None and class_name is not None:
        raise ValueError("Use only one of initial_orientations.class_names or class_name.")
    if class_names is None and class_name is not None:
        class_names = [str(class_name)]
    elif isinstance(class_names, str):
        class_names = [class_names]
    elif class_names is not None:
        class_names = [str(name) for name in class_names]

    test_tomograms = orient_cfg.get("test_tomograms", None)
    if isinstance(test_tomograms, str):
        test_tomograms = [test_tomograms]
    elif test_tomograms is not None:
        test_tomograms = [str(name) for name in test_tomograms]
    if test_tomograms:
        # Accept bare tomogram names as well as paths / *_preds.h5 filenames.
        test_tomograms = [tomo_name_from_path(name) for name in test_tomograms]

    # Public backend is deliberately simple: "cpu" or "gpu".
    # GPU device selection comes only from top-level parameters.gpu_devices.
    # With an integer N, CUDA-visible devices cuda:0..cuda:N-1 are used.
    # A list/tuple is also accepted and interpreted as CUDA-visible indices.
    backend = orient_cfg.get("backend", None)
    if backend is None:
        # Backward compatibility for older configs; new configs should use backend.
        legacy_backend = str(orient_cfg.get("ransac_backend", "cpu")).lower()
        if legacy_backend in ("open3d", "cpu"):
            backend = "cpu"
        elif legacy_backend in ("pytorch3d", "gpu"):
            backend = "gpu"
        else:
            backend = legacy_backend
    backend = str(backend).lower()
    gpu_devices_cfg = (cfg.get("parameters") or {}).get("gpu_devices", 1)

    params = {"predictions": prediction_path,
              "output": output_path,
              "reuse_orientations": bool(orient_cfg.get("reuse_orientations", False)),
              "class_names": class_names,
              "test_tomograms": test_tomograms,
              "n_references": int(orient_cfg.get("n_references", 1)),
              "reference_folder": orient_cfg.get("reference_folder") or None,
              "max_points": int(orient_cfg.get("max_points", 1000)),
              "min_points": int(orient_cfg.get("min_points", 100)),
              "voxel_size": float(orient_cfg.get("voxel_size", 2.0)),
              "ransac_max_iterations": int(orient_cfg.get("ransac_max_iterations", 50000)),
              "ransac_confidence": float(orient_cfg.get("ransac_confidence", 0.999)),
              "ransac_n": int(orient_cfg.get("ransac_n", 4)),
              "ransac_edge_length": float(orient_cfg.get("ransac_edge_length", 0.9)),
              "ransac_mutual_filter": bool(orient_cfg.get("ransac_mutual_filter", False)),
              "backend": backend,
              "gpu_devices": gpu_devices_cfg,
              "ransac_batch_size": int(orient_cfg.get("ransac_batch_size", 4096)),
              "ransac_validation_batch_size": int(orient_cfg.get("ransac_validation_batch_size", 64)),
              "registration_crop_size": orient_cfg.get("registration_crop_size", 64),
              "use_distance_mask": bool(orient_cfg.get("use_distance_mask", False)),
              "exclude_other_instances": bool(orient_cfg.get("exclude_other_instances", False)),
              "similarity_threshold": float(orient_cfg.get("similarity_threshold", 0.1)),
              "sim_refine_angles_deg": [float(v) for v in orient_cfg.get("sim_refine_angles_deg", [8, 4, 2, 1])],
              "sim_refine_downsample": int(orient_cfg.get("sim_refine_downsample", 2)),
              "sim_refine_batch_size": int(orient_cfg.get("sim_refine_batch_size", 26)),
              "refine_registration": bool(orient_cfg.get("refine_registration", True)),
              "refine_max_iterations": int(orient_cfg.get("refine_max_iterations", 30)),
              "refine_max_correspondence_factor": float(orient_cfg.get("refine_max_correspondence_factor", 2.0)),
              "match_trim_fraction": float(orient_cfg.get("match_trim_fraction", 0.85)),
              "cpu_workers": int(orient_cfg.get("cpu_workers", 1)),
              "ransac_seed": (None if orient_cfg.get("ransac_seed", None) is None
                              else int(orient_cfg.get("ransac_seed"))),
              "auto_reference_folder": orient_cfg.get("auto_reference_folder") or None,
              "view_folder": orient_cfg.get("view_folder") or None,
              "view_format": str(orient_cfg.get("view_format", "png")).lower(),
              "view_cmap": str(orient_cfg.get("view_cmap", "magma")),
              "view_draw_contour": bool(orient_cfg.get("view_draw_contour", True)),
              "view_show_score": bool(orient_cfg.get("view_show_score", True)),
              "view_source": str(orient_cfg.get("view_source", "similarity")).lower(),
              "data_folder": cfg.get("data_folder"),
              "file_extension": str(cfg.get("file_extension", ".mrc"))}

    return cfg, params


def validate_params(params):
    if params["n_references"] < 1:
        raise ValueError("initial_orientations.n_references must be >= 1.")
    if params["max_points"] < 10:
        raise ValueError("initial_orientations.max_points must be >= 10.")
    if params["min_points"] < 10:
        raise ValueError("initial_orientations.min_points must be >= 10.")
    if params["max_points"] < params["min_points"]:
        raise ValueError("initial_orientations.max_points must be >= min_points.")
    if params["voxel_size"] <= 0:
        raise ValueError("initial_orientations.voxel_size must be > 0.")
    if params["ransac_max_iterations"] < 1:
        raise ValueError("initial_orientations.ransac_max_iterations must be >= 1.")
    if not (0 < params["ransac_confidence"] < 1):
        raise ValueError("initial_orientations.ransac_confidence must be in (0, 1).")
    if params["ransac_n"] < 3:
        raise ValueError("initial_orientations.ransac_n must be >= 3.")
    if not (0 < params["ransac_edge_length"] <= 1):
        raise ValueError("initial_orientations.ransac_edge_length must be in (0, 1].")
    if params["backend"] not in ("cpu", "gpu"):
        raise ValueError('initial_orientations.backend must be "cpu" or "gpu".')
    if params["ransac_batch_size"] < 1:
        raise ValueError("initial_orientations.ransac_batch_size must be >= 1.")
    if params["ransac_validation_batch_size"] < 1:
        raise ValueError("initial_orientations.ransac_validation_batch_size must be >= 1.")
    if params["backend"] == "gpu":
        if knn_points is None:
            raise ImportError('initial_orientations.backend="gpu" requires PyTorch3D.')
        if not torch.cuda.is_available():
            raise RuntimeError('initial_orientations.backend="gpu" requires CUDA.')

        gpu_value = params["gpu_devices"]
        if isinstance(gpu_value, int):
            if gpu_value < 1:
                raise ValueError('parameters.gpu_devices must be >= 1 when backend="gpu".')
            devices = normalize_ransac_devices(list(range(int(gpu_value))))
        elif isinstance(gpu_value, (list, tuple)):
            if len(gpu_value) < 1:
                raise ValueError('parameters.gpu_devices must not be empty when backend="gpu".')
            devices = normalize_ransac_devices(list(gpu_value))
        else:
            raise ValueError(
                'parameters.gpu_devices must be an integer GPU count or a list of CUDA-visible device ids.')

        visible = torch.cuda.device_count()
        for device in devices:
            idx = torch.device(device).index or 0
            if idx < 0 or idx >= visible:
                raise ValueError(
                    f'GPU configuration requests {device}, but this process sees only {visible} CUDA device(s).')
        params["gpu_devices"] = devices
    else:
        params["gpu_devices"] = []
    if isinstance(params["registration_crop_size"], dict):
        for value in params["registration_crop_size"].values():
            normalize_crop_size(value)
    else:
        normalize_crop_size(params["registration_crop_size"])
    if not np.isfinite(params["similarity_threshold"]):
        raise ValueError("initial_orientations.similarity_threshold must be finite.")
    if params["sim_refine_downsample"] < 1:
        raise ValueError("initial_orientations.sim_refine_downsample must be >= 1.")
    if params["sim_refine_batch_size"] < 1:
        raise ValueError("initial_orientations.sim_refine_batch_size must be >= 1.")
    if any(float(v) <= 0 for v in params["sim_refine_angles_deg"]):
        raise ValueError("initial_orientations.sim_refine_angles_deg values must be > 0.")
    if params["refine_max_iterations"] < 1:
        raise ValueError("initial_orientations.refine_max_iterations must be >= 1.")
    if params["refine_max_correspondence_factor"] <= 0:
        raise ValueError("initial_orientations.refine_max_correspondence_factor must be > 0.")
    if not (0 < params["match_trim_fraction"] <= 1):
        raise ValueError("initial_orientations.match_trim_fraction must be in (0, 1].")
    if params["cpu_workers"] < 1:
        raise ValueError("initial_orientations.cpu_workers must be >= 1.")
    if params["view_format"] not in ("png", "jpg", "jpeg"):
        raise ValueError("initial_orientations.view_format must be png, jpg, or jpeg.")
    if params["view_source"] not in ("similarity", "tomogram"):
        raise ValueError("initial_orientations.view_source must be 'similarity' or 'tomogram'.")
    if params["view_source"] == "tomogram" and not params["data_folder"]:
        raise ValueError("Tomogram views require the top-level data_folder in the config file.")


def main(config_file_path):
    _, params = load_orientation_config(config_file_path)
    validate_params(params)

    print("Initial orientation settings:")
    print(f"  predictions: {params['predictions']}")
    selected_classes = ", ".join(params["class_names"]) if params["class_names"] else "all"
    print(f"  classes: {selected_classes}")
    print(f"  test tomograms: {params['test_tomograms'] or 'all'}")
    print(f"  references/class: {params['n_references']}")
    print(f"  reference folder: {params['reference_folder'] or 'automatic'}")
    print(f"  max points/particle: {params['max_points']}")
    print(f"  min points/particle: {params['min_points']}")
    print(f"  FPFH/RANSAC voxel size: {params['voxel_size']}")
    print(f"  RANSAC iterations: {params['ransac_max_iterations']}")
    print(f"  RANSAC confidence: {params['ransac_confidence']}")
    print(f"  RANSAC sample size: {params['ransac_n']}")
    print(f"  RANSAC edge-length checker: {params['ransac_edge_length']}")
    print(f"  RANSAC mutual filter: {params['ransac_mutual_filter']}")
    print(f"  backend: {params['backend']}")
    if params["backend"] == "gpu":
        print(f"  GPU devices (from parameters.gpu_devices): {params['gpu_devices']}")
        print(f"  GPU workers: {len(params['gpu_devices'])}")
        print(f"  RANSAC hypothesis batch size/GPU: {params['ransac_batch_size']}")
        print(f"  RANSAC validation batch size/GPU: {params['ransac_validation_batch_size']}")
    print(f"  registration crop size: {params['registration_crop_size']}")
    print(f"  distance mask for registration: {params['use_distance_mask']}")
    print(f"  exclude other filled instances: {params['exclude_other_instances']}")
    print(f"  similarity threshold: > {params['similarity_threshold']}")
    print(f"  sim-map local refinement angles: {params['sim_refine_angles_deg']}")
    print(f"  sim-map refinement downsample: {params['sim_refine_downsample']}")
    print(f"  similarity refinement backend: {params['backend']}")
    if params["backend"] == "gpu":
        print(f"  similarity refinement batch size/GPU: {params['sim_refine_batch_size']}")
    print(f"  GICP refinement: {params['refine_registration']}")
    if params["refine_registration"]:
        print(f"  GICP iterations: {params['refine_max_iterations']}")
    print(f"  match trim fraction: {params['match_trim_fraction']}")
    effective_cpu_workers = params["cpu_workers"]
    if params["backend"] == "gpu":
        # One dedicated Python worker is created per CUDA-visible GPU requested
        # through parameters.gpu_devices. cpu_workers is CPU-backend only.
        effective_cpu_workers = 1
        print(f"  GPU worker processes: {len(params['gpu_devices'])}")
    else:
        print(f"  CPU worker processes: {effective_cpu_workers}")
    print(f"  stable RANSAC seed: {params['ransac_seed']}")
    if effective_cpu_workers > 1 and params["ransac_seed"] is None:
        print("  WARNING: multiprocessing + unseeded RANSAC can give different orientations between runs.")
    print(f"  particle views: {params['view_folder'] or 'disabled'}")
    if params["view_folder"]:
        print(f"  view source: {params['view_source']}")
        print(f"  view format: {params['view_format']}")
        print(f"  view colormap: {params['view_cmap']}")
        print(f"  show match score on views: {params['view_show_score']}")
        if params["view_source"] == "tomogram":
            print(f"  data folder: {params['data_folder']}")
    print(f"  output: {params['output']}")
    print(f"  reuse saved orientations: {params['reuse_orientations']}")

    files = prediction_files(params["predictions"])
    print(f"Found {len(files)} prediction H5 file(s)")

    if params["test_tomograms"]:
        file_by_tomo = {tomo_name_from_path(path): path for path in files}
        missing = [name for name in params["test_tomograms"] if name not in file_by_tomo]
        files = [file_by_tomo[name] for name in params["test_tomograms"]
                 if name in file_by_tomo]

        if missing:
            print(f"WARNING: requested test tomogram(s) not found: {missing}")
        if not files:
            raise RuntimeError("None of the requested test_tomograms have prediction H5 files.")

        print(f"Using {len(files)} test tomogram(s): "
              f"{[tomo_name_from_path(path) for path in files]}")

    records = []
    for path in files:
        recs = read_particles(path, max_points=params["max_points"],
                              min_points=params["min_points"],
                              only_classes=params["class_names"],
                              registration_crop_size=params["registration_crop_size"],
                              use_distance_mask=params["use_distance_mask"],
                              exclude_other_instances=params["exclude_other_instances"],
                              similarity_threshold=params["similarity_threshold"])
        print(f"  {Path(path).name}: {len(recs)} particle(s)")
        records.extend(recs)

    if not records:
        raise RuntimeError("No usable particle instances found.")

    print(f"\nTotal particles: {len(records)}")

    output_path = Path(params["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if params["reuse_orientations"]:
        if not output_path.is_file():
            raise FileNotFoundError(
                f"reuse_orientations=True but orientation CSV does not exist: {output_path}"
            )
        result = pd.read_csv(output_path)
        required_columns = {"tomo", "instance_id", "master_reference_id",
                            "r00", "r01", "r02", "r10", "r11", "r12",
                            "r20", "r21", "r22"}
        missing = required_columns.difference(result.columns)
        if missing:
            raise ValueError(
                f"Saved orientation CSV is missing required columns: {sorted(missing)}"
            )
        if params["class_names"] and "class" in result.columns:
            result = result[result["class"].isin(params["class_names"])].copy()
        if params["test_tomograms"] and "tomo" in result.columns:
            result = result[result["tomo"].isin(params["test_tomograms"])].copy()
        if len(result) == 0:
            raise RuntimeError("No saved orientations matched the selected particles/classes.")
        print(f"\nReusing {len(result)} saved orientation(s) from {output_path}")
        star_path = save_relion_star(result, output_path)
        print(f"Saved RELION STAR orientations -> {star_path}")

        if params["view_folder"]:
            n_views = save_aligned_particle_views(records,
                                                  result,
                                                  out_dir=params["view_folder"],
                                                  image_format=params["view_format"],
                                                  cmap=params["view_cmap"],
                                                  value_range=(-1.0, 1.0),
                                                  draw_contour=params["view_draw_contour"],
                                                  view_source=params["view_source"],
                                                  data_folder=params["data_folder"],
                                                  file_extension=params["file_extension"],
                                                  show_score=params["view_show_score"])
            print(f"Saved {n_views} aligned particle view(s) -> {params['view_folder']}")

            if params["reference_folder"]:
                view_references = load_external_references(
                    params["reference_folder"], only_classes=params["class_names"])
            else:
                ref_out = params["auto_reference_folder"]
                if ref_out is None:
                    ref_out = output_path.parent / "auto_references"
                view_references = load_external_references(
                    ref_out, only_classes=params["class_names"])

            n_ref_views = save_reference_views(
                group_references_by_class(view_references),
                records,
                params,
                result=result)
            print(f"Saved {n_ref_views} reference view(s) -> "
                  f"{Path(params['view_folder']) / 'references'}")
        return

    if params["reference_folder"]:
        references = load_external_references(params["reference_folder"], only_classes=params["class_names"])
        print(f"Loaded {len(references)} external reference(s)")
    else:
        ref_out = params["auto_reference_folder"]
        if ref_out is None:
            ref_out = output_path.parent / "auto_references"
        references = choose_automatic_references(records, n_references=params["n_references"], out_dir=ref_out)
        print(f"\nSaved {len(references)} automatic reference file(s) to {ref_out}")

    # Keep the public YAML simple while preserving the existing internal code paths.
    internal_ransac_backend = "pytorch3d" if params["backend"] == "gpu" else "open3d"
    first_gpu_device = params["gpu_devices"][0] if params["backend"] == "gpu" else "auto"

    refs_by_class = make_reference_frames(
        references,
        voxel_size=params["voxel_size"],
        ransac_max_iterations=params["ransac_max_iterations"],
        ransac_confidence=params["ransac_confidence"],
        ransac_n=params["ransac_n"],
        ransac_edge_length=params["ransac_edge_length"],
        ransac_mutual_filter=params["ransac_mutual_filter"],
        ransac_backend=internal_ransac_backend,
        ransac_device=first_gpu_device,
        ransac_batch_size=params["ransac_batch_size"],
        ransac_validation_batch_size=params["ransac_validation_batch_size"],
        ransac_seed=params["ransac_seed"],
        refine_registration=params["refine_registration"],
        refine_max_iterations=params["refine_max_iterations"],
        refine_max_correspondence_factor=params["refine_max_correspondence_factor"])

    result = determine_orientations_multiprocess(
        records,
        refs_by_class,
        cpu_workers=effective_cpu_workers,
        gpu_devices=params["gpu_devices"],
        voxel_size=params["voxel_size"],
        ransac_max_iterations=params["ransac_max_iterations"],
        ransac_confidence=params["ransac_confidence"],
        ransac_n=params["ransac_n"],
        ransac_edge_length=params["ransac_edge_length"],
        ransac_mutual_filter=params["ransac_mutual_filter"],
        ransac_backend=internal_ransac_backend,
        ransac_device=first_gpu_device,
        ransac_batch_size=params["ransac_batch_size"],
        ransac_validation_batch_size=params["ransac_validation_batch_size"],
        refine_registration=params["refine_registration"],
        refine_max_iterations=params["refine_max_iterations"],
        refine_max_correspondence_factor=params["refine_max_correspondence_factor"],
        match_trim_fraction=params["match_trim_fraction"],
        predictions=params["predictions"],
        registration_crop_size=params["registration_crop_size"],
        use_distance_mask=params["use_distance_mask"],
        exclude_other_instances=params["exclude_other_instances"],
        similarity_threshold=params["similarity_threshold"],
        sim_refine_angles_deg=params["sim_refine_angles_deg"],
        sim_refine_downsample=params["sim_refine_downsample"],
        sim_refine_batch_size=params["sim_refine_batch_size"],
        ransac_seed=params["ransac_seed"])

    if len(result) == 0:
        raise RuntimeError("No orientations could be determined.")

    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result)} initial orientations -> {output_path}")
    star_path = save_relion_star(result, output_path)
    print(f"Saved RELION STAR orientations -> {star_path}")

    if params["view_folder"]:
        n_views = save_aligned_particle_views(records,
                                              result,
                                              out_dir=params["view_folder"],
                                              image_format=params["view_format"],
                                              cmap=params["view_cmap"],
                                              value_range=(-1.0, 1.0),
                                              draw_contour=params["view_draw_contour"],
                                              view_source=params["view_source"],
                                              data_folder=params["data_folder"],
                                              file_extension=params["file_extension"],
                                              show_score=params["view_show_score"])
        print(f"Saved {n_views} aligned particle view(s) -> {params['view_folder']}")

        n_ref_views = save_reference_views(
            refs_by_class,
            records,
            params,
            result=result)
        print(f"Saved {n_ref_views} reference view(s) -> "
              f"{Path(params['view_folder']) / 'references'}")


if __name__ == "__main__":
    parser = parser_helper("Determine initial global particle orientations from prototype similarities")
    args = parser.parse_args()
    main(args.config_file)
