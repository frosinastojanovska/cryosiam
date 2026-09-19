import os
import sys
import subprocess
import socket
import uuid
import csv
import yaml
import h5py
import mrcfile
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.transform import resize
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from skimage.morphology import convex_hull_image, remove_small_objects
from monai.transforms import (
    Compose,
    LoadImaged,
    SpatialPad,
    EnsureTyped,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, PatchIter, MrcWriter
from cryosiam.apps.dense_simsiam_semantic import load_backbone_model, load_prediction_model


def get_lamella_thickness_row(input_file, thickness_voxels, normal_axis):
    voxel_size_zyx = np.full(3, np.nan, dtype=np.float64)
    try:
        with mrcfile.open(input_file, permissive=True,
                          header_only=True) as tomo:
            voxel_size_zyx = np.asarray([
                float(tomo.voxel_size.z),
                float(tomo.voxel_size.y),
                float(tomo.voxel_size.x)], dtype=np.float64)
        if (not np.all(np.isfinite(voxel_size_zyx)) or
                np.any(voxel_size_zyx <= 0)):
            raise ValueError(f'invalid voxel size {voxel_size_zyx.tolist()}')
    except Exception as error:
        voxel_size_zyx[:] = np.nan
        print(f'  WARNING: could not read voxel size from {input_file}: '
              f'{error}')

    effective_voxel_size = float('nan')
    thickness_angstrom = float('nan')
    if (normal_axis is not None and np.isfinite(thickness_voxels) and
            np.all(np.isfinite(voxel_size_zyx)) and
            np.all(voxel_size_zyx > 0)):
        normal_axis = np.asarray(normal_axis, dtype=np.float64)
        effective_voxel_size = 1.0 / np.linalg.norm(
            normal_axis / voxel_size_zyx)
        thickness_angstrom = (
                float(thickness_voxels) * effective_voxel_size)

    row = {'tomogram': os.path.basename(input_file),
           'thickness_voxels': float(thickness_voxels),
           'thickness_angstrom': thickness_angstrom,
           'voxel_size_x_angstrom': voxel_size_zyx[2],
           'voxel_size_y_angstrom': voxel_size_zyx[1],
           'voxel_size_z_angstrom': voxel_size_zyx[0],
           'effective_voxel_size_angstrom': effective_voxel_size}

    print(f'  Lamella thickness: {thickness_voxels:.2f} voxels, '
          f'{thickness_angstrom:.2f} Å')
    return row


def update_lamella_thickness_table(output_file, new_rows):
    """Update the shared thickness CSV once, avoiding cross-node write races."""
    fieldnames = [
        'tomogram', 'thickness_voxels', 'thickness_angstrom',
        'voxel_size_x_angstrom', 'voxel_size_y_angstrom',
        'voxel_size_z_angstrom', 'effective_voxel_size_angstrom']

    by_tomogram = {}
    if os.path.isfile(output_file):
        with open(output_file, newline='') as file:
            for existing in csv.DictReader(file):
                if existing.get('tomogram'):
                    by_tomogram[existing['tomogram']] = {
                        name: existing.get(name, '') for name in fieldnames}

    for row in new_rows:
        if row and row.get('tomogram'):
            by_tomogram[row['tomogram']] = {
                name: row.get(name, '') for name in fieldnames}

    rows = [by_tomogram[name] for name in sorted(by_tomogram)]
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Updated thickness table: {output_file}')


def select_lamella_component(probability, threshold=0.5,
                             smoothing_sigma=(3, 3, 3),
                             min_component_size=10000,
                             max_points=100000,
                             max_thickness_ratio=0.6,
                             min_lateral_coverage=0.05,
                             center_weight=0.25, random_seed=0):
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 3:
        raise ValueError(
            f'Lamella selection expects a 3D volume, '
            f'got shape {probability.shape}.')

    smoothed = gaussian(probability, sigma=tuple(smoothing_sigma),
                        mode='constant', preserve_range=True)
    binary = smoothed > threshold
    binary = remove_small_objects(
        binary, min_size=int(min_component_size))
    components = label(binary, connectivity=3)

    if components.max() == 0:
        print('  WARNING: no lamella component survived thresholding.')
        return np.zeros_like(binary, dtype=bool)

    rng = np.random.default_rng(random_seed)
    volume_center = 0.5 * (np.asarray(probability.shape) - 1)
    half_diagonal = np.linalg.norm(0.5 * np.asarray(probability.shape))
    sorted_shape = np.sort(np.asarray(probability.shape, dtype=np.float64))
    reference_lateral_area = sorted_shape[-1] * sorted_shape[-2]

    best_label = None
    best_score = -np.inf
    best_values = None

    for region in regionprops(components, intensity_image=probability):
        coordinates = region.coords.astype(np.float64, copy=False)
        if len(coordinates) > max_points:
            indices = rng.choice(
                len(coordinates), size=int(max_points), replace=False)
            coordinates = coordinates[indices]

        centered = coordinates - coordinates.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        projected = centered @ eigenvectors[:, order]

        lower, upper = np.percentile(projected, [2, 98], axis=0)
        lateral_1, lateral_2, thickness = upper - lower + 1.0
        if lateral_2 <= 0 or thickness <= 0:
            continue

        thickness_ratio = thickness / lateral_2
        lateral_coverage = np.clip(
            lateral_1 * lateral_2 / reference_lateral_area, 0.0, 1.0)

        if thickness_ratio > max_thickness_ratio:
            continue
        if lateral_coverage < min_lateral_coverage:
            continue

        oriented_box_volume = lateral_1 * lateral_2 * thickness
        rectangularity = np.clip(
            float(region.area) / max(oriented_box_volume, 1.0), 0.0, 1.0)
        center_distance = np.linalg.norm(
            np.asarray(region.centroid) - volume_center)
        centrality = 1.0 - np.clip(
            center_distance / max(half_diagonal, 1.0), 0.0, 1.0)
        thinness = 1.0 - np.clip(
            thickness_ratio / max(max_thickness_ratio, 1e-6), 0.0, 1.0)

        score = (
                float(region.mean_intensity) *
                lateral_coverage *
                (0.5 + 0.5 * rectangularity) *
                (0.5 + 0.5 * thinness) *
                ((1.0 - center_weight) + center_weight * centrality)
        )

        if score > best_score:
            best_score = score
            best_label = region.label
            best_values = {'lateral_1': lateral_1,
                           'lateral_2': lateral_2,
                           'thickness': thickness,
                           'thickness_ratio': thickness_ratio,
                           'lateral_coverage': lateral_coverage,
                           'rectangularity': rectangularity,
                           'mean_probability': float(region.mean_intensity),
                           'centrality': centrality,
                           'score': score}

    if best_label is None:
        print('  WARNING: no component satisfied the lamella criteria. '
              'Try increasing max_thickness_ratio or decreasing '
              'min_lateral_coverage in the lamella configuration.')
        return np.zeros_like(binary, dtype=bool)

    print('  Selected lamella component: '
          f'lateral=({best_values["lateral_1"]:.1f}, '
          f'{best_values["lateral_2"]:.1f}), '
          f'thickness={best_values["thickness"]:.1f}, '
          f'ratio={best_values["thickness_ratio"]:.3f}, '
          f'coverage={best_values["lateral_coverage"]:.3f}, '
          f'rectangularity={best_values["rectangularity"]:.3f}, '
          f'mean_prob={best_values["mean_probability"]:.3f}, '
          f'centrality={best_values["centrality"]:.3f}, '
          f'score={best_values["score"]:.5f}')

    return components == best_label


def fit_lamella_mask(component, max_fit_size=192,
                     trim_percentile=2.0, trim_iterations=3,
                     surface_bins=64, plane_margin=None):
    component = np.asarray(component, dtype=bool)
    if component.ndim != 3:
        raise ValueError(
            f'Lamella fitting expects a 3D mask, '
            f'got shape {component.shape}.')
    if not component.any():
        return np.zeros_like(component, dtype=bool), float('nan'), None

    stride = max(
        1, int(np.ceil(max(component.shape) / float(max_fit_size))))
    sampled = component[::stride, ::stride, ::stride]
    coordinates = np.argwhere(sampled).astype(np.float64)
    coordinates *= stride

    if len(coordinates) < 3:
        print('  WARNING: too few lamella voxels for parallel-plane fitting; '
              'using the selected component unchanged.')
        return component.copy(), float('nan'), None

    trim_percentile = float(trim_percentile)
    if not 0 <= trim_percentile < 50:
        raise ValueError('lamella.fit.trim_percentile must be in [0, 50).')

    fitting_points = coordinates
    for _ in range(max(1, int(trim_iterations))):
        center = fitting_points.mean(axis=0)
        centered = fitting_points - center
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

        normal_coordinates = centered @ axes[:, 2]
        lower, upper = np.percentile(
            normal_coordinates,
            [trim_percentile, 100.0 - trim_percentile])
        keep = ((normal_coordinates >= lower) &
                (normal_coordinates <= upper))

        if keep.all() or keep.sum() < 3:
            break
        fitting_points = fitting_points[keep]

    center = fitting_points.mean(axis=0)
    centered = fitting_points - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

    # PCA determines the lamella normal reliably, but its two in-plane axes are
    # ambiguous for a nearly square lamella. Anchor them to the tomogram X
    # direction to keep surface binning stable.
    normal_axis = axes[:, 2]
    reference_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference_axis, normal_axis)) > 0.9:
        reference_axis = np.array([0.0, 1.0, 0.0])
    lateral_axis_1 = (
            reference_axis -
            np.dot(reference_axis, normal_axis) * normal_axis)
    lateral_axis_1 /= max(np.linalg.norm(lateral_axis_1), 1e-12)
    lateral_axis_2 = np.cross(normal_axis, lateral_axis_1)
    lateral_axis_2 /= max(np.linalg.norm(lateral_axis_2), 1e-12)
    axes = np.column_stack(
        [lateral_axis_1, lateral_axis_2, normal_axis])

    projected = (coordinates - center) @ axes
    lateral_1 = projected[:, 0]
    lateral_2 = projected[:, 1]
    normal_coordinates = projected[:, 2]

    u_min, u_max = np.percentile(
        lateral_1, [trim_percentile, 100.0 - trim_percentile])
    v_min, v_max = np.percentile(
        lateral_2, [trim_percentile, 100.0 - trim_percentile])

    if plane_margin is None:
        plane_margin = 0.5 * float(stride)

    n_bins = max(4, int(surface_bins))
    u_scale = max(u_max - u_min, 1e-6)
    v_scale = max(v_max - v_min, 1e-6)
    u_bin = np.clip(
        ((lateral_1 - u_min) / u_scale * n_bins).astype(int),
        0, n_bins - 1)
    v_bin = np.clip(
        ((lateral_2 - v_min) / v_scale * n_bins).astype(int),
        0, n_bins - 1)
    bin_index = u_bin * n_bins + v_bin

    lower_surface = np.full(n_bins * n_bins, np.inf)
    upper_surface = np.full(n_bins * n_bins, -np.inf)
    np.minimum.at(lower_surface, bin_index, normal_coordinates)
    np.maximum.at(upper_surface, bin_index, normal_coordinates)
    valid_bins = np.isfinite(lower_surface) & np.isfinite(upper_surface)

    if not valid_bins.any():
        print('  WARNING: no valid surface columns found; using the selected '
              'component unchanged.')
        return component.copy(), float('nan'), normal_axis

    lower_plane = float(np.median(lower_surface[valid_bins]))
    upper_plane = float(np.median(upper_surface[valid_bins]))
    if lower_plane > upper_plane:
        lower_plane, upper_plane = upper_plane, lower_plane
    lower_plane -= float(plane_margin)
    upper_plane += float(plane_margin)
    thickness_voxels = upper_plane - lower_plane

    output = np.zeros_like(component, dtype=bool)
    height, width = component.shape[1:]
    yy, xx = np.indices((height, width), dtype=np.float32)
    dy = yy - np.float32(center[1])
    dx = xx - np.float32(center[2])

    base_n = axes[1, 2] * dy + axes[2, 2] * dx

    for z_index in range(component.shape[0]):
        dz = float(z_index) - center[0]
        normal = base_n + axes[0, 2] * dz
        output[z_index] = (
                (normal >= lower_plane) & (normal <= upper_plane)
        )

    print('  Fitted lamella mask: '
          f'sampled_lateral_extent=({u_max - u_min:.1f}, '
          f'{v_max - v_min:.1f}), '
          f'thickness={thickness_voxels:.1f}, '
          f'fit_stride={stride}, valid_surface_bins={valid_bins.sum()}')
    return output, thickness_voxels, normal_axis


class ShardedPatchDataset(IterableDataset):
    """Shard PatchIter output across GPU ranks and DataLoader workers."""

    def __init__(self, img, patch_iter, shard_rank=0, num_shards=1):
        super().__init__()
        self.img = img
        self.patch_iter = patch_iter
        self.shard_rank = int(shard_rank)
        self.num_shards = int(num_shards)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            n_workers = 1
        else:
            worker_id = worker.id
            n_workers = worker.num_workers

        global_shard = self.shard_rank * n_workers + worker_id
        total_shards = self.num_shards * n_workers

        for patch_idx, item in enumerate(self.patch_iter(self.img)):
            if patch_idx % total_shards == global_shard:
                yield item


def local_patch_count(input_size, patch_size, group_rank, group_size, inference_workers, overlap=0.5):
    counts = []
    for size, patch in zip(input_size, patch_size):
        stride = max(1, int(round(patch * (1.0 - overlap))))
        if size <= patch:
            counts.append(1)
        else:
            counts.append(int(np.ceil((size - patch) / stride)) + 1)

    total_patches = int(np.prod(counts))
    n_workers = max(1, int(inference_workers))
    total_shards = group_size * n_workers
    first_shard = group_rank * n_workers
    count = 0

    for worker_id in range(n_workers):
        shard = first_shard + worker_id
        if shard < total_patches:
            count += (total_patches - 1 - shard) // total_shards + 1

    return count


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f'LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} '
                'CUDA device(s) are visible.')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')

    return distributed, rank, world_size, local_rank, device


def build_node_process_group(distributed, rank, world_size):
    """
    *Build one torch.distributed process group per physical node.*

    *Every GPU process on a node belongs to the same group. A tomogram is never*
    *split across nodes: all local GPUs cooperate on one tomogram at a time.*
    """
    hostname = socket.gethostname()

    if not distributed:
        return {'hostname': hostname,
                'node_id': 0,
                'n_nodes': 1,
                'group': None,
                'group_rank': 0,
                'group_size': 1,
                'leader_rank': 0,
                'group_ranks': [0],
                'all_hosts': [hostname]}

    hosts = [None] * world_size
    dist.all_gather_object(hosts, hostname)

    host_order = []
    host_to_ranks = {}
    for global_rank, host in enumerate(hosts):
        if host not in host_to_ranks:
            host_order.append(host)
            host_to_ranks[host] = []
        host_to_ranks[host].append(global_rank)

    # Every rank must create every process group in exactly the same order.
    groups = {}
    for host in host_order:
        ranks = host_to_ranks[host]
        groups[host] = dist.new_group(ranks=ranks)

    group_ranks = host_to_ranks[hostname]
    return {'hostname': hostname,
            'node_id': host_order.index(hostname),
            'n_nodes': len(host_order),
            'group': groups[hostname],
            'group_rank': group_ranks.index(rank),
            'group_size': len(group_ranks),
            'leader_rank': group_ranks[0],
            'group_ranks': group_ranks,
            'all_hosts': host_order}


def make_run_temp_dir(temp_root, distributed, rank):
    """
    *Create the same run-specific subdirectory name on every node.*

    *If temp_root is /dev/shm, each node gets its own local RAM-backed directory.*
    *That is safe because a tomogram is constrained to one node group.*
    """
    if distributed:
        run_id = [uuid.uuid4().hex[:12] if rank == 0 else None]
        dist.broadcast_object_list(run_id, src=0)
        run_id = run_id[0]
    else:
        run_id = uuid.uuid4().hex[:12]

    path = os.path.join(temp_root, f'cryosiam_lamella_{run_id}')
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def _shared_prediction_paths(tmp_dir, tomo_name, leader_rank):
    base = os.path.join(
        tmp_dir, f'.{_safe_name(tomo_name)}.lamella_group{leader_rank}')
    return {
        'probs': base + '.probs.f32',
        'distances': base + '.distances.f32',
    }


def _open_shared_arrays(paths, num_classes, input_size, create=False):
    mode = 'w+' if create else 'r+'
    shape = (num_classes, *input_size)
    probs = np.memmap(paths['probs'], dtype=np.float32, mode=mode, shape=shape)
    distances = np.memmap(
        paths['distances'], dtype=np.float32, mode=mode, shape=shape)
    if create:
        # Freshly expanded memmap files are zero-filled; flush metadata now so
        # the other ranks can safely open them after the barrier.
        probs.flush()
        distances.flush()
    return probs, distances


def _close_memmap(arr):
    if arr is None:
        return
    try:
        arr.flush()
    except Exception:
        pass
    base = arr
    seen = set()
    while getattr(base, 'base', None) is not None and id(base) not in seen:
        seen.add(id(base))
        base = base.base
    if isinstance(base, np.memmap):
        try:
            base._mmap.close()
        except Exception:
            pass


def _cleanup_shared_files(paths):
    if not paths:
        return
    for path in paths.values():
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def h5_create_dataset(hf, name, data):
    return hf.create_dataset(name, data=data)


def reusable_prediction(path):
    if not os.path.isfile(path):
        return False
    try:
        with h5py.File(path, 'r') as f:
            return all(key in f for key in ('labels', 'probs', 'distances'))
    except OSError:
        return False


def load_reusable_prediction(path):
    with h5py.File(path, 'r') as f:
        return f['labels'][()], f['probs'][()], f['distances'][()]


def _broadcast_bool(value, process_group, leader_rank, group_rank, group_size):
    if group_size == 1:
        return bool(value)
    obj = [bool(value) if group_rank == 0 else None]
    dist.broadcast_object_list(obj, src=leader_rank, group=process_group)
    return bool(obj[0])


def _patch_slices(c_batch, input_size, patch_size):
    """Return non-overlapping output slices and corresponding patch slices."""
    slices = tuple(
        slice(c[0], c[1] - p // 4) if c[0] == 0
        else slice(c[0] + p // 4, c[1]) if c[1] >= s
        else slice(c[0] + p // 4, c[1] - p // 4)
        for c, s, p in zip(c_batch, input_size, patch_size)
    )
    slices2 = tuple(
        slice(0, 3 * p // 4) if c[0] == 0
        else slice(p // 4, p - (c[1] - s)) if c[1] >= s
        else slice(p // 4, 3 * p // 4)
        for c, s, p in zip(c_batch, input_size, patch_size)
    )
    return slices, slices2


def distributed_patch_prediction(img, backbone, prediction_model, patch_iter, patch_size, input_size,
                                 num_classes, batch_size, device, tomo_name,
                                 process_group=None, group_rank=0, group_size=1, leader_rank=0,
                                 shared_tmp_dir=None, inference_workers=0):
    """Predict one tomogram, sharding its patches across the assigned GPUs."""
    distributed_tomo = group_size > 1
    paths = None

    if distributed_tomo:
        os.makedirs(shared_tmp_dir, exist_ok=True)
        paths = _shared_prediction_paths(shared_tmp_dir, tomo_name, leader_rank)
        if group_rank == 0:
            _cleanup_shared_files(paths)
            probs_out, distances_out = _open_shared_arrays(
                paths, num_classes, input_size, create=True)
        else:
            probs_out = distances_out = None

        dist.barrier(group=process_group)

        if group_rank != 0:
            probs_out, distances_out = _open_shared_arrays(
                paths, num_classes, input_size, create=False)
    else:
        probs_out = np.zeros((num_classes, *input_size), dtype=np.float32)
        distances_out = np.zeros((num_classes, *input_size), dtype=np.float32)

    patch_dataset = ShardedPatchDataset(
        img, patch_iter, shard_rank=group_rank, num_shards=group_size)
    loader = DataLoader(patch_dataset,
                        batch_size=batch_size,
                        num_workers=int(inference_workers),
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=bool(inference_workers > 0)
                        )

    progress_total = local_patch_count(input_size, patch_size, group_rank, group_size, inference_workers)
    progress = tqdm(total=progress_total,
                    desc=f'  {tomo_name} | leader GPU shard',
                    unit='patch',
                    leave=False,
                    disable=(group_rank != 0))

    with torch.inference_mode():
        for item in loader:
            patches = item[0].to(device, non_blocking=True)
            coords = item[1].numpy().astype(int)
            if group_rank == 0:
                progress.update(patches.shape[0])

            z, _ = backbone.forward_predict(patches)
            out, d_out = prediction_model(z)
            if num_classes == 1:
                out = torch.sigmoid(out)
            else:
                out = torch.softmax(out, dim=1)

            out = out.float().cpu().numpy()
            d_out = d_out.float().cpu().numpy()

            for batch_i in range(patches.shape[0]):
                c_batch = coords[batch_i][1:]

                # Ignore the trailing patches that start entirely inside the
                # trimmed overlap margin, matching the original implementation.
                if any(
                        c_batch[d][0] >= input_size[d] - patch_size[d] // 4
                        for d in range(len(input_size))):
                    continue

                slices, slices2 = _patch_slices(
                    c_batch, input_size, patch_size)
                probs_out[(slice(None),) + slices] = \
                    out[batch_i][(slice(None),) + slices2]
                distances_out[(slice(None),) + slices] = \
                    d_out[batch_i][(slice(None),) + slices2]

    if group_rank == 0:
        progress.close()

    if isinstance(probs_out, np.memmap):
        probs_out.flush()
        distances_out.flush()

    if distributed_tomo:
        dist.barrier(group=process_group)
        if group_rank != 0:
            _close_memmap(probs_out)
            _close_memmap(distances_out)
            return None, None, paths

    return probs_out, distances_out, paths


def labels_from_probabilities(probs_out, threshold, num_classes):
    if num_classes == 1:
        return (probs_out[0] > threshold).astype(np.uint8)

    thresholded = probs_out.copy()
    thresholded[thresholded < threshold] = 0
    return np.argmax(thresholded, axis=0).astype(np.uint8)


def read_requested_topology(cfg):
    """Read the requested cluster topology from the YAML configuration."""
    parameters = cfg.get('parameters', {})
    nodes_value = parameters.get('nodes', 1)
    gpu_devices_value = parameters.get('gpu_devices', 1)

    try:
        n_nodes = int(nodes_value)
    except (TypeError, ValueError):
        raise ValueError(
            'parameters.nodes must be an integer number of physical nodes, '
            f'got {nodes_value!r}.')

    try:
        gpus_per_node = int(gpu_devices_value)
    except (TypeError, ValueError):
        raise ValueError(
            'parameters.gpu_devices must be an integer number of GPUs per node, '
            f'got {gpu_devices_value!r}.')

    if n_nodes < 1:
        raise ValueError(
            f'parameters.nodes must be >= 1, got {n_nodes}.')
    if gpus_per_node < 1:
        raise ValueError(
            f'parameters.gpu_devices must be >= 1, got {gpus_per_node}.')

    return n_nodes, gpus_per_node


def maybe_launch_distributed_from_config(config_file_path):
    """Auto-launch local workers for one node; multi-node jobs are launched externally."""
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        return False

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    n_nodes, gpus_per_node = read_requested_topology(cfg)

    if n_nodes > 1:
        raise RuntimeError(
            f'The config requests parameters.nodes={n_nodes} with '
            f'parameters.gpu_devices={gpus_per_node} GPUs per node. Multi-node '
            'jobs must first be launched across the allocated nodes with your '
            'scheduler/torchrun so that RANK, LOCAL_RANK and WORLD_SIZE are set. '
            f'The expected WORLD_SIZE is {n_nodes * gpus_per_node}.')

    if gpus_per_node <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but CUDA is not available.')

    visible_gpus = torch.cuda.device_count()
    if gpus_per_node > visible_gpus:
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but only {visible_gpus} CUDA '
            'device(s) are visible. Check the allocation and CUDA_VISIBLE_DEVICES.')

    cmd = [
        sys.executable,
        '-m', 'torch.distributed.run',
        '--standalone',
        '--max_restarts=0',
        f'--nproc_per_node={gpus_per_node}',
        os.path.abspath(__file__),
        *sys.argv[1:],
    ]

    print(
        f'Launching distributed lamella inference on 1 node with '
        f'{gpus_per_node} GPU worker(s)...')
    launch_env = os.environ.copy()
    launch_env.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run(cmd, check=True, env=launch_env)
    return True


def main(config_file_path, filename=None):
    distributed, rank, world_size, local_rank, device = init_distributed()

    with open(config_file_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    checkpoint_path = cfg['trained_model']
    backbone = load_backbone_model(checkpoint_path, device)
    prediction_model = load_prediction_model(checkpoint_path, device)
    backbone.eval()
    prediction_model.eval()

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    net_config = checkpoint['hyper_parameters']['config']

    test_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    reuse_predictions = bool(cfg.get('reuse_predictions', False))
    mask_folder = cfg.get('mask_folder') or None
    lamella_cfg = cfg.get('lamella', {})
    lamella_fit_cfg = lamella_cfg.get('fit', {})
    num_classes = int(net_config['parameters']['network']['out_channels'])
    threshold = float(cfg['parameters']['network']['threshold'])
    patch_size = list(net_config['parameters']['data']['patch_size'])
    spatial_dims = int(net_config['parameters']['network']['spatial_dims'])
    batch_size = int(cfg['hyper_parameters']['batch_size'])
    inference_workers = int(cfg.get('inference_workers', 0))

    temp_root = cfg.get(
        'temp_dir',
        cfg.get('distributed_tmp_dir', prediction_folder))

    os.makedirs(prediction_folder, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    requested_nodes, requested_gpus_per_node = read_requested_topology(cfg)
    node_info = build_node_process_group(
        distributed, rank, world_size)

    expected_world_size = requested_nodes * requested_gpus_per_node
    actual_world_size = world_size if distributed else 1
    if actual_world_size != expected_world_size:
        raise RuntimeError(
            f'Distributed topology mismatch: config requests '
            f'{requested_nodes} node(s) x {requested_gpus_per_node} GPU(s) per node '
            f'= {expected_world_size} process(es), but WORLD_SIZE is '
            f'{actual_world_size}. Check the scheduler/torchrun launch.')
    if node_info['n_nodes'] != requested_nodes:
        raise RuntimeError(
            f'Distributed topology mismatch: config requests '
            f'parameters.nodes={requested_nodes}, but the launched ranks span '
            f'{node_info["n_nodes"]} physical hostname(s): '
            f'{node_info["all_hosts"]}.')
    if node_info['group_size'] != requested_gpus_per_node:
        raise RuntimeError(
            f'Distributed topology mismatch on host {node_info["hostname"]}: '
            f'config requests {requested_gpus_per_node} GPU process(es) per node, '
            f'but this node has {node_info["group_size"]} distributed rank(s).')

    shared_tmp_dir = make_run_temp_dir(
        temp_root, distributed, rank)

    if filename:
        files = [filename]
    else:
        files = cfg.get('test_files')
        if files is None:
            files = [
                x for x in os.listdir(test_folder)
                if os.path.isfile(os.path.join(test_folder, x))
                   and x.endswith(cfg['file_extension'])
            ]
    files = sorted(files)

    if rank == 0:
        print(f'Using {world_size if distributed else 1} GPU process(es)')
        print(f'Test tomograms: {len(files)}')
        print('H5 compression: disabled')
        print(
            f'Requested topology: {requested_nodes} node(s) x '
            f'{requested_gpus_per_node} GPU(s) per node = '
            f'{expected_world_size} total GPU process(es)')
        print(f'Physical nodes detected: {node_info["n_nodes"]}')
        print('Node groups:')
        for node_id, host in enumerate(node_info['all_hosts']):
            print(f'  node {node_id}: {host}')
        print(f'Temporary prediction storage root: {temp_root}')

    if not files:
        if distributed:
            dist.destroy_process_group()
        return

    reader = MrcReader(read_in_mem=True)
    writer = MrcWriter()
    transforms = Compose([LoadImaged(keys='image', reader=reader),
                          EnsureChannelFirstd(keys='image'),
                          NumpyToTensord(keys='image'),
                          ScaleIntensityRanged(keys='image',
                                               a_min=cfg['parameters']['data']['min'],
                                               a_max=cfg['parameters']['data']['max'],
                                               b_min=0, b_max=1, clip=True),
                          NormalizeIntensityd(keys='image',
                                              subtrahend=cfg['parameters']['data']['mean'],
                                              divisor=cfg['parameters']['data']['std']),
                          EnsureTyped(keys='image', data_type='tensor')])

    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='constant')
    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0),
                               overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0),
                               overlap=(0, 0.5, 0.5, 0.5))

    n_files = len(files)
    jobs = []

    # One tomogram per physical node group.
    #
    # Example: 2 nodes x 4 GPUs:
    #   node 0 / ranks 0-3 -> tomo 0, then tomo 2, ...
    #   node 1 / ranks 4-7 -> tomo 1, then tomo 3, ...
    #
    # A tomogram is never split across nodes. If there are fewer tomograms
    # than nodes, extra nodes remain idle rather than joining another node's
    # tomogram.
    for file_idx in range(node_info['node_id'], n_files, node_info['n_nodes']):
        jobs.append({
            'file_idx': file_idx,
            'group': node_info['group'],
            'group_rank': node_info['group_rank'],
            'group_size': node_info['group_size'],
            'leader_rank': node_info['leader_rank'],
            'group_ranks': node_info['group_ranks'],
        })

    if node_info['group_rank'] == 0:
        assigned_names = [os.path.basename(files[j['file_idx']]) for j in jobs]
        print(
            f'Node {node_info["node_id"]} ({node_info["hostname"]}) '
            f'uses ranks {node_info["group_ranks"]} / '
            f'{node_info["group_size"]} GPU(s)')
        print(
            f'  assigned tomograms: '
            f'{assigned_names if assigned_names else "none"}')
        print(f'  local temp dir: {shared_tmp_dir}')

    local_thickness_rows = []

    for job in jobs:
        file = files[job['file_idx']]
        current_file = os.path.join(test_folder, file)
        root = os.path.basename(file).split(cfg['file_extension'])[0]
        out_file = os.path.join(prediction_folder, os.path.basename(file))
        preds_h5 = os.path.join(prediction_folder, f'{root}_preds.h5')
        leader = job['group_rank'] == 0

        can_reuse = (
                leader and reuse_predictions and reusable_prediction(preds_h5))
        can_reuse = _broadcast_bool(
            can_reuse, job['group'], job['leader_rank'],
            job['group_rank'], job['group_size'])

        shared_paths = None
        probs_out = distances_out = None
        shared_probs_handle = None
        shared_distances_handle = None

        if can_reuse:
            if leader:
                print(f'Loading saved predictions for file {current_file}')
                labels_out, probs_out, distances_out = \
                    load_reusable_prediction(preds_h5)
                original_size = list(labels_out.shape)
            else:
                # The group leader alone reloads and postprocesses cached maps.
                # Other ranks wait until that tomogram is finished.
                if job['group_size'] > 1:
                    dist.barrier(group=job['group'])
                continue
        else:
            # Every assigned rank loads the same tomogram, then computes only
            # its own non-overlapping patch shard.
            sample = transforms({
                'image': current_file,
                'file_name': current_file
            })
            original_size = list(sample['image'][0].shape)
            img = pad_transform(sample['image'])
            input_size = list(img[0].shape)

            if leader:
                print(f'Running prediction for file {current_file}')
                print(
                    f'  size={original_size}, padded={input_size}, '
                    f'GPUs={job["group_size"]}')

            probs_out, distances_out, shared_paths = \
                distributed_patch_prediction(img=img,
                                             backbone=backbone,
                                             prediction_model=prediction_model,
                                             patch_iter=patch_iter,
                                             patch_size=patch_size,
                                             input_size=input_size,
                                             num_classes=num_classes,
                                             batch_size=batch_size,
                                             device=device,
                                             tomo_name=root,
                                             process_group=job['group'],
                                             group_rank=job['group_rank'],
                                             group_size=job['group_size'],
                                             leader_rank=job['leader_rank'],
                                             shared_tmp_dir=shared_tmp_dir,
                                             inference_workers=inference_workers)

            if not leader:
                # Wait here until the leader has finished postprocessing and no
                # longer needs the shared prediction files.
                if job['group_size'] > 1:
                    dist.barrier(group=job['group'])
                continue

            shared_probs_handle = probs_out if isinstance(probs_out, np.memmap) else None
            shared_distances_handle = (
                distances_out if isinstance(distances_out, np.memmap) else None)

            crop = tuple(slice(0, n) for n in original_size)
            probs_out = probs_out[(slice(None),) + crop]
            distances_out = distances_out[(slice(None),) + crop]
            labels_out = labels_from_probabilities(
                probs_out, threshold, num_classes)

        # Only the leader postprocesses / writes this tomogram.
        if mask_folder:
            print(f'Masking out of file {current_file}')
            mask_name = f'{root}_preds.h5'
            with h5py.File(os.path.join(mask_folder, mask_name), 'r') as f:
                mask = f['labels'][()].astype(np.int8)
            if list(mask.shape) != list(original_size):
                mask = resize(mask, original_size, mode='constant', preserve_range=True).astype(np.int8)
            labels_out = labels_out * mask
            probs_out = probs_out * mask
            distances_out = distances_out * mask

        print(f'Refining lamella prediction for file {current_file}')
        lamella_component = select_lamella_component(
            probs_out[0],
            threshold=threshold,
            smoothing_sigma=lamella_cfg.get('smoothing_sigma', (3, 3, 3)),
            min_component_size=lamella_cfg.get('min_component_size', 10000),
            max_points=lamella_cfg.get('max_points', 100000),
            max_thickness_ratio=lamella_cfg.get('max_thickness_ratio', 0.6),
            min_lateral_coverage=lamella_cfg.get('min_lateral_coverage', 0.05),
            center_weight=lamella_cfg.get('center_weight', 0.25)
        )

        if spatial_dims == 3:
            labels_out, thickness_voxels, normal_axis = fit_lamella_mask(
                lamella_component,
                max_fit_size=lamella_fit_cfg.get('max_size', 192),
                trim_percentile=lamella_fit_cfg.get('trim_percentile', 2.0),
                trim_iterations=lamella_fit_cfg.get('trim_iterations', 3),
                surface_bins=lamella_fit_cfg.get('surface_bins', 64),
                plane_margin=lamella_fit_cfg.get('plane_margin', 5)
            )
            labels_out = labels_out.astype(int)
        else:
            thickness_voxels = float('nan')
            normal_axis = None
            labels_out = np.zeros_like(
                lamella_component, dtype=np.uint8)
            for ind in range(lamella_component.shape[0]):
                if np.sum(lamella_component[ind]) == 0:
                    continue
                labels_out[ind] = convex_hull_image(
                    lamella_component[ind])

        local_thickness_rows.append(
            get_lamella_thickness_row(
                current_file, thickness_voxels, normal_axis))

        print(f'Saving predictions for file {current_file}')
        if cfg.get('save_internal_files', False):
            with h5py.File(preds_h5, 'w') as f:
                h5_create_dataset(f, 'labels', labels_out)
                h5_create_dataset(f, 'probs', np.asarray(probs_out))
                h5_create_dataset(f, 'distances', np.asarray(distances_out))
        else:
            if cfg.get('save_original_file_extension', False):
                writer.set_data_array(
                    labels_out.astype(np.uint8), channel_dim=None)
                writer.write(out_file)
            else:
                with h5py.File(preds_h5, 'w') as f:
                    h5_create_dataset(f, 'labels', labels_out.astype(np.uint8))

        if shared_probs_handle is not None:
            _close_memmap(shared_probs_handle)
        elif isinstance(probs_out, np.memmap):
            _close_memmap(probs_out)
        if shared_distances_handle is not None:
            _close_memmap(shared_distances_handle)
        elif isinstance(distances_out, np.memmap):
            _close_memmap(distances_out)

        if job['group_size'] > 1:
            # Release worker ranks only after the leader has finished using the
            # shared memmaps, then remove temporary files.
            dist.barrier(group=job['group'])
            _cleanup_shared_files(shared_paths)

    # Gather thickness metadata and update the shared CSV exactly once.
    if distributed:
        gathered_rows = [None] * world_size if rank == 0 else None
        dist.gather_object(
            local_thickness_rows,
            gathered_rows,
            dst=0)
        dist.barrier()
    else:
        gathered_rows = [local_thickness_rows]

    if rank == 0:
        all_rows = []
        for rows in gathered_rows:
            if rows:
                all_rows.extend(rows)
        if all_rows:
            thickness_file = os.path.join(
                prediction_folder, 'lamella_thickness.csv')
            update_lamella_thickness_table(
                thickness_file, all_rows)

    if distributed:
        dist.barrier()

    # shared_tmp_dir is node-local when temp_dir=/dev/shm. Every node leader
    # removes its own empty run directory independently.
    if node_info['group_rank'] == 0:
        try:
            os.rmdir(shared_tmp_dir)
        except OSError:
            pass

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    if not maybe_launch_distributed_from_config(args.config_file):
        main(args.config_file, getattr(args, 'filename', None))
