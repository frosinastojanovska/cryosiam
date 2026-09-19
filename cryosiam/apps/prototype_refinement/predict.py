import os
import sys
import uuid
import socket
import subprocess
import h5py
import yaml
import torch
import torch.distributed as dist
import starfile
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import edt
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max
from skimage.morphology import binary_closing, ball
from skimage.segmentation import watershed, relabel_sequential
from skimage.transform import resize
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from monai.transforms import SpatialPad
from cryosiam.utils import parser_helper
from cryosiam.data import PatchIter
from cryosiam.apps.prototype_refinement.utils import (
    load_backbone_model, load_decoder_model, load_class_names, load_temperature,
    load_prototype_bank, prototypes_dict_from_bank, get_background_prototypes,
    build_prototypes_from_support, load_prototypes_file,
    build_tomo_transforms, padded_size_for_even_tiling)


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
            worker_id, n_workers = 0, 1
        else:
            worker_id, n_workers = worker.id, worker.num_workers

        global_shard = self.shard_rank * n_workers + worker_id
        total_shards = self.num_shards * n_workers
        for patch_idx, item in enumerate(self.patch_iter(self.img)):
            if patch_idx % total_shards == global_shard:
                yield item


def patch_grid_count(input_size, patch_size, overlap=0.5):
    counts = []
    for size, patch in zip(input_size, patch_size):
        stride = max(1, int(round(patch * (1.0 - overlap))))
        counts.append(1 if size <= patch else int(np.ceil((size - patch) / stride)) + 1)
    return int(np.prod(counts))


def local_patch_count(input_size, patch_size, group_rank, group_size, inference_workers, overlap=0.5):
    total_patches = patch_grid_count(input_size, patch_size, overlap=overlap)
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
                f'LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA device(s) are visible.')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cpu'

    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
    return distributed, rank, world_size, local_rank, device


def build_node_process_group(distributed, rank, world_size):
    """Create one process group per physical node."""
    hostname = socket.gethostname()
    if not distributed:
        return {'hostname': hostname, 'node_id': 0, 'n_nodes': 1, 'group': None, 'group_rank': 0,
                'group_size': 1, 'leader_rank': 0, 'group_ranks': [0], 'all_hosts': [hostname]}

    hosts = [None] * world_size
    dist.all_gather_object(hosts, hostname)
    host_order, host_to_ranks = [], {}
    for global_rank, host in enumerate(hosts):
        if host not in host_to_ranks:
            host_order.append(host)
            host_to_ranks[host] = []
        host_to_ranks[host].append(global_rank)

    groups = {}
    for host in host_order:
        groups[host] = dist.new_group(ranks=host_to_ranks[host])

    group_ranks = host_to_ranks[hostname]
    return {'hostname': hostname, 'node_id': host_order.index(hostname), 'n_nodes': len(host_order),
            'group': groups[hostname], 'group_rank': group_ranks.index(rank), 'group_size': len(group_ranks),
            'leader_rank': group_ranks[0], 'group_ranks': group_ranks, 'all_hosts': host_order}


def make_run_temp_dir(temp_root, distributed, rank):
    run_id = [uuid.uuid4().hex[:12] if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(run_id, src=0)
    path = os.path.join(temp_root, f'cryosiam_prototype_prediction_{run_id[0]}')
    os.makedirs(path, exist_ok=True)
    return path


def read_requested_topology(cfg):
    parameters = cfg.get('parameters', {})
    nodes = int(parameters.get('nodes', 1))
    gpus_per_node = int(parameters.get('gpu_devices', 1))
    if nodes < 1 or gpus_per_node < 1:
        raise ValueError('parameters.nodes and parameters.gpu_devices must both be >= 1.')
    return nodes, gpus_per_node


def group_broadcast_bool(value, group, leader_rank, group_rank, group_size):
    if group_size == 1:
        return bool(value)
    obj = [bool(value) if group_rank == 0 else None]
    dist.broadcast_object_list(obj, src=leader_rank, group=group)
    return bool(obj[0])


def _safe_tomo_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def _shared_prediction_paths(tmp_dir, tomo_name, leader_rank):
    base = os.path.join(tmp_dir, f'.{_safe_tomo_name(tomo_name)}.group{leader_rank}')
    return {'sim': base + '.sim.f32', 'prob': base + '.prob.f32', 'distance': base + '.distance.f32'}


def _open_shared_prediction_volumes(paths, input_size, n_sim_channels, n_distance_channels,
                                    create=False, roi_mask=None):
    mode = 'w+' if create else 'r+'
    sim_vol = np.memmap(paths['sim'], dtype=np.float32, mode=mode, shape=(n_sim_channels, *input_size))
    prob_vol = np.memmap(paths['prob'], dtype=np.float32, mode=mode, shape=(n_sim_channels, *input_size))
    distance_vol = None
    if n_distance_channels > 0:
        distance_vol = np.memmap(paths['distance'], dtype=np.float32, mode=mode,
                                 shape=(n_distance_channels, *input_size))

    if create:
        if roi_mask is not None:
            sim_vol[:] = -1.0
            prob_vol[0] = 1.0
        sim_vol.flush()
        prob_vol.flush()
        if distance_vol is not None:
            distance_vol.flush()
    return sim_vol, prob_vol, distance_vol


def close_memmap(arr):
    if arr is None:
        return
    base, seen = arr, set()
    while getattr(base, 'base', None) is not None and id(base) not in seen:
        seen.add(id(base))
        base = base.base
    if isinstance(base, np.memmap):
        try:
            base.flush()
        except Exception:
            pass
        try:
            base._mmap.close()
        except Exception:
            pass


def cleanup_shared_prediction_files(paths):
    if not paths:
        return
    for path in paths.values():
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def h5_create_dataset(hf, name, data):
    return hf.create_dataset(name, data=data)


def load_reusable_prediction(h5_out, use_distance_head):
    if not os.path.exists(h5_out):
        return None

    with h5py.File(h5_out, 'r') as hf:
        stored_names = list(hf.attrs.get('class_names', []))
        stored_names = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in stored_names
        ]
        distance_maps_available = (
                not use_distance_head or
                all(f'distance_{name}' in hf for name in stored_names)
        )

        valid = (
                ('seg_mask_raw' in hf or 'seg_mask' in hf) and
                stored_names and
                all(f'prob_{name}' in hf for name in stored_names) and
                all(f'sim_{name}' in hf for name in stored_names) and
                distance_maps_available
        )
        if not valid:
            return None

        seg_key = 'seg_mask_raw' if 'seg_mask_raw' in hf else 'seg_mask'
        seg_mask = hf[seg_key][()]
        prototype_probs = np.stack(
            [hf[f'prob_{name}'][()] for name in stored_names])
        prototype_sims = np.stack(
            [hf[f'sim_{name}'][()] for name in stored_names])

        distance_maps = None
        if use_distance_head:
            distance_maps = np.stack(
                [hf[f'distance_{name}'][()] for name in stored_names])

    return stored_names, seg_mask, prototype_probs, prototype_sims, distance_maps


def unpack_decoder_output(decoder, decoder_output):
    use_dual_head = bool(getattr(decoder, 'use_dual_head', False))
    use_distance_head = bool(getattr(decoder, 'use_distance_head', False))
    if use_dual_head and use_distance_head:
        feat, dual_seg_logits, distance_logits = decoder_output
    elif use_dual_head:
        feat, dual_seg_logits = decoder_output
        distance_logits = None
    elif use_distance_head:
        feat, distance_logits = decoder_output
        dual_seg_logits = None
    else:
        feat = decoder_output
        dual_seg_logits = None
        distance_logits = None
    return feat, dual_seg_logits, distance_logits


def get_selected_distance_channels(ordered_names, all_class_names):
    if all_class_names is None:
        raise ValueError('all_class_names is required when the distance head is used.')
    name_to_channel = {name: index for index, name in enumerate(all_class_names)}
    missing = [name for name in ordered_names if name not in name_to_channel]
    if missing:
        raise ValueError(f'Distance head has no channels for: {missing}')
    return [name_to_channel[name] for name in ordered_names]


def extract_predictions(img, backbone, decoder, temperature, prototypes, bg_prototypes,
                        class_names, patch_size, input_size, batch_size, patch_iter, device,
                        tomo_name='tomo', roi_mask=None, all_class_names=None, use_seg_head_probs=False,
                        process_group=None, group_rank=0, group_size=1, group_leader_rank=0,
                        shared_tmp_dir=None, inference_workers=0):
    ordered_names = [name for name in class_names if name in prototypes]
    if not ordered_names:
        raise ValueError('None of class_names have a prototype available.')
    if bg_prototypes is None:
        raise ValueError('Prototype prediction requires background prototypes to match training.')
    if roi_mask is not None and roi_mask.shape != tuple(input_size):
        raise ValueError(f'ROI mask shape {roi_mask.shape} does not match prediction volume {tuple(input_size)}.')

    proto_tensors = [F.normalize(prototypes[name].float(), dim=1).to(device) for name in ordered_names]
    bg_proto = F.normalize(bg_prototypes.float(), dim=1).to(device)
    channel_protos = [bg_proto] + proto_tensors
    n_sim_channels = 1 + len(proto_tensors)
    use_distance_head = bool(getattr(decoder, 'use_distance_head', False))

    seg_head_channels = None
    if use_seg_head_probs:
        if all_class_names is None:
            raise ValueError('all_class_names is required when use_seg_head_probs=True.')
        name_to_channel = {name: i + 1 for i, name in enumerate(all_class_names)}
        missing = [name for name in ordered_names if name not in name_to_channel]
        if missing:
            raise ValueError(f'use_seg_head_probs=True but {missing} are not in all_class_names -- '
                             f'the segmentation head has no channel for them.')
        seg_head_channels = [name_to_channel[name] for name in ordered_names]

    distance_head_channels = get_selected_distance_channels(ordered_names, all_class_names) \
        if use_distance_head else None
    sim_stride = (1, 1, 1)
    distributed_tomo = group_size > 1
    shared_paths = None

    if distributed_tomo:
        if shared_tmp_dir is None:
            raise ValueError('shared_tmp_dir is required for multi-GPU tomogram inference.')
        os.makedirs(shared_tmp_dir, exist_ok=True)
        shared_paths = _shared_prediction_paths(shared_tmp_dir, tomo_name, group_leader_rank)

        if group_rank == 0:
            cleanup_shared_prediction_files(shared_paths)
            sim_vol, prob_vol, distance_vol = _open_shared_prediction_volumes(
                shared_paths, input_size, n_sim_channels,
                len(ordered_names) if use_distance_head else 0, create=True, roi_mask=roi_mask)
        else:
            sim_vol = prob_vol = distance_vol = None

        dist.barrier(group=process_group)
        if group_rank != 0:
            sim_vol, prob_vol, distance_vol = _open_shared_prediction_volumes(
                shared_paths, input_size, n_sim_channels,
                len(ordered_names) if use_distance_head else 0, create=False)
    else:
        sim_vol = np.full((n_sim_channels, *input_size), -1, dtype=np.float32) \
            if roi_mask is not None else np.zeros((n_sim_channels, *input_size), dtype=np.float32)
        prob_vol = np.zeros((n_sim_channels, *input_size), dtype=np.float32)
        distance_vol = np.zeros((len(ordered_names), *input_size), dtype=np.float32) \
            if use_distance_head else None
        if roi_mask is not None:
            prob_vol[0] = 1.0

    if group_rank == 0:
        gpu_text = f'{group_size} GPU(s) for this tomogram' if distributed_tomo else '1 GPU'
        print(f'  Prediction volume: {input_size}  ({n_sim_channels} channel(s), '
              f'K={channel_protos[0].shape[0]} sub-prototypes each; {gpu_text})')
        if use_seg_head_probs:
            print('  Probabilities: dual-head segmentation softmax '
                  '(unselected foreground classes are merged into background)')
        else:
            print('  Probabilities: temperature-scaled softmax over prototype similarities')
        print(f'  Distance prediction: {"ENABLED" if use_distance_head else "disabled"}')

    patch_ds = ShardedPatchDataset(img, patch_iter, shard_rank=group_rank, num_shards=group_size)
    loader = DataLoader(patch_ds, batch_size=batch_size, num_workers=int(inference_workers),
                        pin_memory=torch.cuda.is_available(), persistent_workers=bool(inference_workers > 0))
    progress_total = local_patch_count(input_size, patch_size, group_rank, group_size, inference_workers)
    progress = tqdm(total=progress_total, desc=f'  {tomo_name} | leader GPU shard', unit='patch',
                    leave=False, disable=(group_rank != 0))

    eligible_patches = 0
    skipped_patches = 0

    with torch.inference_mode():
        for item in loader:
            patches = item[0]
            coords = item[1].numpy().astype(int)
            if group_rank == 0:
                progress.update(patches.shape[0])

            active_patches = []
            for batch_i in range(patches.shape[0]):
                c_batch = coords[batch_i][1:]
                if (c_batch[0][0] >= input_size[0] - patch_size[0] // 4 or
                        c_batch[1][0] >= input_size[1] - patch_size[1] // 4 or
                        c_batch[2][0] >= input_size[2] - patch_size[2] // 4):
                    continue

                eligible_patches += 1
                slices = tuple(slice(c[0], c[1] - p // 4) if c[0] == 0
                               else slice(c[0] + p // 4, c[1]) if c[1] >= s
                else slice(c[0] + p // 4, c[1] - p // 4)
                               for c, s, p in zip(c_batch, input_size, patch_size))
                slices2 = tuple(slice(0, 3 * p // 4) if c[0] == 0
                                else slice(p // 4, p - (c[1] - s)) if c[1] >= s
                else slice(p // 4, 3 * p // 4)
                                for c, s, p in zip(c_batch, input_size, patch_size))

                if roi_mask is not None and not roi_mask[slices].any():
                    skipped_patches += 1
                    continue
                active_patches.append((batch_i, slices, slices2))

            if not active_patches:
                continue

            active_indices = torch.tensor([item[0] for item in active_patches], dtype=torch.long)
            patches = patches.index_select(0, active_indices).to(device, non_blocking=True)
            s_feats, _ = backbone.get_encoder_features_list(patches)
            decoder_output = decoder(s_feats, output_size=patches.shape[-3:])
            feat, dual_seg_logits, distance_logits = unpack_decoder_output(decoder, decoder_output)

            per_channel_sims = []
            for proto_k in channel_protos:
                sim_k = torch.einsum('bcdhw,kc->bkdhw', feat, proto_k)
                per_channel_sims.append(sim_k.max(dim=1).values)
            raw_sim = torch.stack(per_channel_sims, dim=1)

            if use_seg_head_probs:
                full_probs = torch.softmax(dual_seg_logits, dim=1)
                selected_fg = full_probs[:, seg_head_channels]
                background_prob = (1.0 - selected_fg.sum(dim=1, keepdim=True)).clamp_min(0.0)
                probs = torch.cat([background_prob, selected_fg], dim=1)
            else:
                probs = torch.softmax(raw_sim / temperature, dim=1)

            distance_prediction = None
            if distance_logits is not None:
                distance_prediction = torch.tanh(distance_logits[:, distance_head_channels])

            raw_sim_np = raw_sim.float().cpu().numpy()
            probs_np = probs.float().cpu().numpy()
            distance_np = distance_prediction.float().cpu().numpy() if distance_prediction is not None else None

            for output_i, (_, slices, slices2) in enumerate(active_patches):
                sim_vol[(slice(None),) + slices] = raw_sim_np[output_i][(slice(None),) + slices2]
                prob_vol[(slice(None),) + slices] = probs_np[output_i][(slice(None),) + slices2]
                if distance_np is not None:
                    distance_vol[(slice(None),) + slices] = distance_np[output_i][(slice(None),) + slices2]

    if group_rank == 0:
        progress.close()

    if isinstance(sim_vol, np.memmap):
        sim_vol.flush()
        prob_vol.flush()
        if distance_vol is not None:
            distance_vol.flush()

    if distributed_tomo:
        counter = torch.tensor([eligible_patches, skipped_patches], dtype=torch.long, device=device)
        dist.reduce(counter, dst=group_leader_rank, op=dist.ReduceOp.SUM, group=process_group)
        dist.barrier(group=process_group)

        if group_rank == 0 and roi_mask is not None:
            print(f'  Skipped {int(counter[1].item())}/{int(counter[0].item())} '
                  f'prediction patches outside the external mask')
        if group_rank != 0:
            close_memmap(sim_vol)
            close_memmap(prob_vol)
            close_memmap(distance_vol)
            return None, None, None, ordered_names, sim_stride, shared_paths
    elif roi_mask is not None:
        print(f'  Skipped {skipped_patches}/{eligible_patches} prediction patches outside the external mask')

    return sim_vol, prob_vol, distance_vol, ordered_names, sim_stride, shared_paths


def upsample_sim_maps(sim_maps, stride, original_size):
    if sim_maps is None:
        return None
    if sim_maps.shape[0] == 0:
        return sim_maps
    real_fpn_size = [int(np.ceil(original_size[i] / stride[i])) for i in range(3)]
    real_fpn_size = [min(real_fpn_size[i], sim_maps.shape[1 + i]) for i in range(3)]
    cropped = sim_maps[:, :real_fpn_size[0], :real_fpn_size[1], :real_fpn_size[2]]
    if list(cropped.shape[1:]) == list(original_size):
        return cropped
    ups = []
    for k in range(cropped.shape[0]):
        up = F.interpolate(
            torch.from_numpy(cropped[k]).unsqueeze(0).unsqueeze(0),
            size=list(original_size), mode='trilinear', align_corners=False
        )[0, 0].numpy()
        ups.append(up)
    return np.stack(ups)


def apply_roi_mask(maps, mask):
    if maps is None:
        return None
    if mask.shape != maps.shape[1:]:
        mask = resize(mask.astype(np.uint8), maps.shape[1:], order=0,
                      mode='constant', preserve_range=True,
                      anti_aliasing=False).astype(bool)
    return maps * mask[np.newaxis, ...]


def load_roi_mask(mask_folder, tomo_name, dataset_key, tomo_shape):
    if not mask_folder:
        return None
    mask_path = os.path.join(mask_folder, f'{tomo_name}_preds.h5')
    if not os.path.isfile(mask_path):
        print(f'  WARNING: mask not found at {mask_path} — skipping')
        return None
    print(f'  Applying external mask: {mask_path}')
    with h5py.File(mask_path, 'r') as f:
        if dataset_key not in f:
            raise KeyError(f'Mask file {mask_path} does not contain dataset "{dataset_key}".')
        mask = f[dataset_key][()]
    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f'Expected a 3D mask in {mask_path}, got shape {mask.shape}.')
    tomo_shape = tuple(int(n) for n in tomo_shape)
    if mask.shape != tomo_shape:
        print(f'  Resizing external mask: {list(mask.shape)} -> {list(tomo_shape)}')
        mask = resize(mask.astype(np.uint8), tomo_shape, order=0,
                      mode='constant', preserve_range=True,
                      anti_aliasing=False)
    return mask > 0


def apply_activation(sim_maps):
    t = torch.from_numpy(sim_maps)
    return torch.softmax(t, dim=0).numpy()


def extract_seg_mask(probs_all, threshold=0.5):
    if probs_all.shape[0] < 2:
        raise ValueError('Expected background plus at least one foreground class.')
    seg_mask = probs_all.argmax(axis=0).astype(np.int16)
    winning_prob = probs_all.max(axis=0)
    seg_mask[(seg_mask > 0) & (winning_prob <= threshold)] = 0
    return seg_mask


def get_adjacent_pairs(instances):
    pairs = set()
    for axis in range(3):
        slices1 = [slice(None)] * 3
        slices2 = [slice(None)] * 3
        slices1[axis] = slice(None, -1)
        slices2[axis] = slice(1, None)
        a = instances[tuple(slices1)]
        b = instances[tuple(slices2)]
        mask = (a > 0) & (b > 0) & (a != b)
        if np.any(mask):
            values = np.stack([a[mask], b[mask]], axis=1)
            values.sort(axis=1)
            for pair in np.unique(values, axis=0):
                pairs.add((int(pair[0]), int(pair[1])))
    return pairs


def get_label_geometry(instances):
    """Vectorized voxel counts and centroids for a 3D integer label volume."""
    flat = instances.reshape(-1)
    valid = flat > 0
    n_labels = int(flat.max()) + 1 if flat.size else 1

    counts = np.zeros(n_labels, dtype=np.int64)
    centers = np.full((n_labels, 3), np.nan, dtype=np.float32)

    if not np.any(valid):
        return counts, centers

    labels = flat[valid].astype(np.int64, copy=False)
    flat_indices = np.flatnonzero(valid)

    counts = np.bincount(labels, minlength=n_labels).astype(np.int64, copy=False)

    H, W = instances.shape[1], instances.shape[2]
    hw = H * W
    z = flat_indices // hw
    rem = flat_indices - z * hw
    y = rem // W
    x = rem - y * W

    z_sums = np.bincount(labels, weights=z, minlength=n_labels)
    y_sums = np.bincount(labels, weights=y, minlength=n_labels)
    x_sums = np.bincount(labels, weights=x, minlength=n_labels)

    present = counts > 0
    centers[present, 0] = z_sums[present] / counts[present]
    centers[present, 1] = y_sums[present] / counts[present]
    centers[present, 2] = x_sums[present] / counts[present]

    return counts, centers


def merge_watershed_instances(instances, max_center_distance, max_single_voxels):
    if max_center_distance <= 0 or max_single_voxels <= 0:
        return instances, 0

    instances = instances.astype(np.int32, copy=True)
    total_merges = 0

    while True:
        counts, centers = get_label_geometry(instances)
        present = np.flatnonzero(counts > 0)
        present = present[present > 0]
        if len(present) <= 1:
            break

        candidates = []
        for id1, id2 in get_adjacent_pairs(instances):
            if id1 >= len(counts) or id2 >= len(counts):
                continue
            if counts[id1] == 0 or counts[id2] == 0:
                continue

            combined_voxels = int(counts[id1] + counts[id2])
            if combined_voxels > max_single_voxels:
                continue

            center_distance = float(np.linalg.norm(centers[id1] - centers[id2]))
            if center_distance <= max_center_distance:
                candidates.append((center_distance, combined_voxels, id1, id2))

        if not candidates:
            break

        candidates.sort()
        used = set()
        merges = 0
        label_map = np.arange(int(instances.max()) + 1, dtype=np.int32)

        for _, _, id1, id2 in candidates:
            if id1 in used or id2 in used:
                continue
            label_map[id2] = id1
            used.add(id1)
            used.add(id2)
            merges += 1
            total_merges += 1

        if merges == 0:
            break

        instances = label_map[instances]
        instances, _, _ = relabel_sequential(instances, offset=1)
        instances = instances.astype(np.int32, copy=False)

    return instances, total_merges


def make_closed_solid_proxy(binary, closing_radius=3):
    solid = binary.astype(bool, copy=True)
    if closing_radius > 0:
        solid = binary_closing(solid, footprint=ball(int(closing_radius)))
    solid = binary_fill_holes(solid)
    return solid


def fast_edt_3d(binary):
    """Fast 3D Euclidean distance transform on the processed solid proxy."""
    n_threads = max(1, min(8, os.cpu_count() or 1))
    return edt.edt(
        np.ascontiguousarray(binary, dtype=np.uint8),
        black_border=True,
        parallel=n_threads
    ).astype(np.float32, copy=False)


def get_instance_statistics(instances, original_binary, prob_channel, sim_channel):
    """
    **Vectorized statistics over the ORIGINAL predicted foreground voxels.**

    **Returns compact sparse assignment arrays as well, so cleaned semantic and**
    **instance masks can be rebuilt later without sorting all foreground voxels.**
    """
    flat_instances = instances.reshape(-1)
    original_flat = original_binary.reshape(-1)
    valid = original_flat & (flat_instances > 0)

    n_labels = int(instances.max()) + 1
    counts = np.zeros(n_labels, dtype=np.int64)
    prob_sums = np.zeros(n_labels, dtype=np.float64)
    sim_sums = np.zeros(n_labels, dtype=np.float64)

    if not np.any(valid):
        empty_indices = np.empty(0, dtype=np.uint32 if instances.size < 2 ** 32 else np.int64)
        empty_labels = np.empty(0, dtype=np.uint16 if n_labels <= 65536 else np.uint32)
        return counts, prob_sums, sim_sums, empty_indices, empty_labels

    valid_indices = np.flatnonzero(valid)
    valid_labels = flat_instances[valid].astype(np.int64, copy=False)

    counts = np.bincount(
        valid_labels,
        minlength=n_labels
    ).astype(np.int64, copy=False)

    prob_sums = np.bincount(
        valid_labels,
        weights=prob_channel.reshape(-1)[valid],
        minlength=n_labels
    )

    sim_sums = np.bincount(
        valid_labels,
        weights=sim_channel.reshape(-1)[valid],
        minlength=n_labels
    )

    # Compact sparse assignment representation. No argsort / np.unique needed.
    if instances.size < 2 ** 32:
        valid_indices = valid_indices.astype(np.uint32, copy=False)
    if n_labels <= 65536:
        valid_labels = valid_labels.astype(np.uint16, copy=False)
    else:
        valid_labels = valid_labels.astype(np.uint32, copy=False)

    return counts, prob_sums, sim_sums, valid_indices, valid_labels


def extract_centers_per_class(prob_maps, sim_maps, ordered_names, seg_mask,
                              min_voxels, merge_instances=None,
                              postprocess=None):
    all_peaks = []
    solid_masks = {}
    instance_assignments = {}

    columns = ['_instance_id', 'z', 'y', 'x', 'n_voxels', 'mean_prob',
               'sum_prob', 'mean_sim', 'sum_sim', 'class']

    for k, name in enumerate(ordered_names):
        cls_label = k + 1
        original_binary = (seg_mask == cls_label)
        class_cfg = postprocess.get(name, {}) if postprocess else {}

        if not isinstance(class_cfg, dict):
            raise ValueError(f'postprocess.{name} must be a mapping.')

        if not original_binary.any():
            print(f'  {name}: no voxels')
            all_peaks.append(pd.DataFrame(columns=columns))
            continue

        radius = float(class_cfg.get('solid_proxy_radius', 0))
        binary = make_closed_solid_proxy(original_binary, radius)
        solid_masks[name] = binary

        min_distance = max(1, int(class_cfg.get('center_min_distance', 1)))

        # EDT is computed on the processed SOLID proxy, not on the raw/hollow mask.
        dist = fast_edt_3d(binary)

        coords = peak_local_max(
            dist,
            labels=binary,
            min_distance=min_distance,
            exclude_border=False
        )

        markers = np.zeros_like(binary, dtype=np.int32)
        if len(coords):
            markers[tuple(coords.T)] = np.arange(
                1, len(coords) + 1, dtype=np.int32)

        n_instances = len(coords)
        if n_instances > 0:
            instances = watershed(
                -dist,
                markers=markers,
                mask=binary
            ).astype(np.int32, copy=False)
        else:
            instances = markers

        if n_instances == 0:
            print(f'  {name}: no instances found')
            all_peaks.append(pd.DataFrame(columns=columns))
            continue

        n_before_merge = int(instances.max())
        n_merges = 0

        if isinstance(merge_instances, dict) and name in merge_instances:
            merge_cfg = merge_instances[name]
            max_center_distance = float(merge_cfg.get('max_center_distance', 0))
            max_single_voxels = float(merge_cfg.get('max_single_voxels', 0))
            instances, n_merges = merge_watershed_instances(
                instances,
                max_center_distance,
                max_single_voxels
            )

        if n_merges > 0:
            print(
                f'  {name}: {n_before_merge} watershed regions -> '
                f'{int(instances.max())} after merging'
            )

        # Geometric centers come from the solid watershed regions.
        _, centers = get_label_geometry(instances)

        # Filtering statistics are measured only over original predicted voxels.
        counts, prob_sums, sim_sums, valid_indices, valid_labels = \
            get_instance_statistics(
                instances,
                original_binary,
                prob_maps[k],
                sim_maps[k]
            )

        instance_assignments[name] = (valid_indices, valid_labels)

        instance_ids = np.flatnonzero(counts > min_voxels)
        instance_ids = instance_ids[instance_ids > 0]

        if len(instance_ids):
            finite = np.isfinite(centers[instance_ids, 0])
            instance_ids = instance_ids[finite]

        if len(instance_ids) == 0:
            df = pd.DataFrame(columns=columns)
        else:
            n_vox = counts[instance_ids].astype(np.int64, copy=False)
            df = pd.DataFrame({
                '_instance_id': instance_ids.astype(np.int64, copy=False),
                'z': centers[instance_ids, 0].astype(np.float64, copy=False),
                'y': centers[instance_ids, 1].astype(np.float64, copy=False),
                'x': centers[instance_ids, 2].astype(np.float64, copy=False),
                'n_voxels': n_vox,
                'mean_prob': prob_sums[instance_ids] / n_vox,
                'sum_prob': prob_sums[instance_ids],
                'mean_sim': sim_sums[instance_ids] / n_vox,
                'sum_sim': sim_sums[instance_ids],
                'class': name
            }, columns=columns)

        all_peaks.append(df)
        print(f'  {name}: {len(df)} particles (min_distance={min_distance})')

    return all_peaks, solid_masks, instance_assignments


def postprocess_instances(df, postprocess):
    if not postprocess or len(df) == 0:
        return df.copy()
    filtered = []
    for name, class_df in df.groupby('class', sort=False):
        class_df = class_df.copy()
        class_cfg = postprocess.get(name, {})
        if not class_cfg:
            filtered.append(class_df)
            continue
        n_before = len(class_df)
        n_filters = 0
        for rule, threshold in class_cfg.items():
            if rule in ('solid_proxy_radius', 'center_min_distance'):
                continue
            if rule.startswith('min_'):
                feature = rule[4:]
                comparison = 'min'
            elif rule.startswith('max_'):
                feature = rule[4:]
                comparison = 'max'
            else:
                raise ValueError(f'Unknown postprocessing option "{rule}" for class "{name}". '
                                 f'Use solid_proxy_radius, center_min_distance, '
                                 f'min_<feature>, or max_<feature>.')
            if feature not in class_df.columns:
                raise ValueError(f'Unknown postprocessing feature "{feature}" in "{rule}" '
                                 f'for class "{name}". Available features: {list(df.columns)}')
            threshold = float(threshold)
            if comparison == 'min':
                class_df = class_df[class_df[feature] >= threshold]
            else:
                class_df = class_df[class_df[feature] <= threshold]
            n_filters += 1
        if n_filters == 0:
            filtered.append(class_df)
            continue
        print(f'  {name}: {n_before} → {len(class_df)} after postprocessing')
        filtered.append(class_df)
    if not filtered:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(filtered, ignore_index=True)


def build_output_masks(seg_mask, kept_instances, instance_assignments,
                       ordered_names):
    """
    **Rebuild cleaned semantic and globally numbered instance masks with LUTs.**

    **instance_assignments stores only sparse original-foreground voxel indices**
    **and their watershed labels, so no full per-class instance volumes and no**
    **voxel sorting are required.**
    """
    cleaned = np.zeros_like(seg_mask)
    instance_mask = np.zeros(seg_mask.shape, dtype=np.int32)

    if len(kept_instances) == 0:
        return cleaned, instance_mask

    if 'instance_id' not in kept_instances.columns:
        raise ValueError(
            'kept_instances must contain instance_id so STAR/CSV IDs match '
            'the H5 instance_mask.')

    class_labels = {name: k + 1 for k, name in enumerate(ordered_names)}

    cleaned_flat = cleaned.reshape(-1)
    instance_flat = instance_mask.reshape(-1)

    for name, class_df in kept_instances.groupby('class', sort=False):
        assignment = instance_assignments.get(name)
        if assignment is None or len(class_df) == 0:
            continue

        voxel_indices, watershed_labels = assignment
        if len(voxel_indices) == 0:
            continue

        local_ids = class_df['_instance_id'].to_numpy(dtype=np.int64)
        final_ids = class_df['instance_id'].to_numpy(dtype=np.int32)

        max_label = int(watershed_labels.max()) if len(watershed_labels) else 0
        if len(local_ids):
            max_label = max(max_label, int(local_ids.max()))

        final_lut = np.zeros(max_label + 1, dtype=np.int32)
        final_lut[local_ids] = final_ids

        mapped = final_lut[watershed_labels.astype(np.int64, copy=False)]
        keep = mapped > 0
        if not np.any(keep):
            continue

        selected_indices = voxel_indices[keep].astype(np.int64, copy=False)
        cleaned_flat[selected_indices] = class_labels[name]
        instance_flat[selected_indices] = mapped[keep]

    return cleaned, instance_mask


def centers_for_output(df):
    return df.drop(columns=['_instance_id'], errors='ignore')


def centers_for_star(df, rename_map):
    star_df = centers_for_output(df).rename(
        columns=rename_map, errors='ignore')
    if 'rlnMicrographName' in star_df.columns:
        star_df['rlnTomoName'] = star_df['rlnMicrographName']
    return star_df


def _main(config_file_path, filename=None):
    distributed, rank, world_size, local_rank, device = init_distributed()

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg['trained_model']
    backbone = load_backbone_model(checkpoint_path, device)
    decoder = load_decoder_model(checkpoint_path, device=device)
    backbone.eval()
    decoder.eval()
    temperature = load_temperature(checkpoint_path)

    all_class_names = cfg.get('class_names') or load_class_names(checkpoint_path)
    use_distance_head = bool(getattr(decoder, 'use_distance_head', False))

    predict_classes = cfg.get('predict_classes')
    if predict_classes is None:
        class_names = list(all_class_names)
    else:
        unknown = [name for name in predict_classes if name not in all_class_names]
        if unknown:
            raise ValueError(f'Unknown predict_classes: {unknown}. Available classes: {all_class_names}')
        class_names = [name for name in all_class_names if name in predict_classes]
        if not class_names:
            raise ValueError('predict_classes did not select any classes.')

    save_h5 = str(cfg.get('save_h5', 'both')).lower()
    if save_h5 not in ('both', 'probabilities', 'similarities', 'none'):
        raise ValueError('save_h5 must be "both", "probabilities", "similarities", or "none".')

    data_cfg = cfg['parameters']['data']
    patch_size = data_cfg['patch_size']
    file_ext = cfg['file_extension']
    batch_size = int(cfg['hyper_parameters']['batch_size'])
    threshold = float(cfg.get('prob_threshold', 0.5))
    min_voxels = float(cfg.get('min_voxels', 0))
    reuse = bool(cfg.get('reuse_predictions', True))
    k_per_class = int(cfg.get('k_per_class', 1))
    inference_workers = int(cfg.get('inference_workers', 0))
    save_individual_centers = bool(cfg.get('save_individual_centers', False))
    # Default: save only the final joined all-tomogram CSV/STAR files.
    # Set true to additionally keep per-tomogram all_centers CSV/STAR files.
    save_per_tomogram_centers = bool(
        cfg.get('save_per_tomogram_centers', False))
    mask_folder = cfg.get('mask_folder') or None
    mask_dataset_key = cfg.get('mask_dataset_key', 'labels')
    postprocess = cfg.get('postprocess') or {}
    merge_instances = cfg.get('merge_instances')
    filtering_enabled = any(isinstance(class_cfg, dict) and
                            any(rule.startswith(('min_', 'max_')) for rule in class_cfg)
                            for class_cfg in postprocess.values())

    prediction_folder = cfg['prediction_folder']
    output_folder = cfg.get('output_folder', prediction_folder)
    temp_root = cfg.get('temp_dir', cfg.get('distributed_tmp_dir', prediction_folder))
    os.makedirs(prediction_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    requested_nodes, requested_gpus = read_requested_topology(cfg)
    node_info = build_node_process_group(distributed, rank, world_size)
    expected_world_size = requested_nodes * requested_gpus
    actual_world_size = world_size if distributed else 1
    if actual_world_size != expected_world_size:
        raise RuntimeError(f'Distributed topology mismatch: config requests {requested_nodes} node(s) x '
                           f'{requested_gpus} GPU(s) per node = {expected_world_size} process(es), '
                           f'but WORLD_SIZE is {actual_world_size}.')
    if node_info['n_nodes'] != requested_nodes:
        raise RuntimeError(f'Distributed topology mismatch: config requests {requested_nodes} physical node(s), '
                           f'but ranks span {node_info["n_nodes"]}: {node_info["all_hosts"]}.')
    if node_info['group_size'] != requested_gpus:
        raise RuntimeError(f'Distributed topology mismatch on {node_info["hostname"]}: config requests '
                           f'{requested_gpus} GPU process(es) per node, but found {node_info["group_size"]}.')

    shared_tmp_dir = make_run_temp_dir(temp_root, distributed, rank)
    used_custom_prototypes = bool(cfg.get('prototypes_file') or cfg.get('support_files'))
    use_seg_head_probs = bool(decoder.use_dual_head) and not used_custom_prototypes

    if rank == 0:
        if decoder.use_dual_head and used_custom_prototypes:
            print('NOTE: checkpoint has a dual segmentation head, but custom prototypes were given '
                  '(prototypes_file/support_files) -- using similarity-based probabilities.')
        print(f'Classes: {class_names}')
        print(f'Temperature: {temperature}')
        print(f'H5 output: {save_h5}')
        print('H5 compression: disabled')
        print(f'Dual-head probabilities: {"ENABLED" if use_seg_head_probs else "disabled"}')
        print(f'Distance prediction: {"ENABLED" if use_distance_head else "disabled"}')
        print(
            'Per-tomogram center CSV/STAR files: '
            f'{"ENABLED" if save_per_tomogram_centers else "disabled (joined files only)"}')
        print(f'Requested topology: {requested_nodes} node(s) x {requested_gpus} GPU(s) per node')
        print(f'Temporary prediction storage root: {temp_root}')

    if cfg.get('prototypes_file') and cfg.get('support_files'):
        raise ValueError('Specify only one of "prototypes_file" or "support_files", not both.')

    prototypes = None
    if cfg.get('prototypes_file'):
        if rank == 0:
            print('\nLoading pre-computed prototypes...')
            prototypes = load_prototypes_file(cfg['prototypes_file'])
            prototypes = {name: tensor.detach().cpu() for name, tensor in prototypes.items()}
        if distributed:
            obj = [prototypes if rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            prototypes = obj[0]
    elif cfg.get('support_files'):
        if rank == 0:
            print('\nBuilding prototypes from support examples...')
            prototypes = build_prototypes_from_support(
                backbone, decoder, cfg['support_files'], patch_size, data_cfg, device,
                k_per_class=k_per_class)
            prototypes = {name: tensor.detach().cpu() for name, tensor in prototypes.items()}
        if distributed:
            obj = [prototypes if rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            prototypes = obj[0]

    if rank == 0:
        print('\nLoading trained prototype bank for background channel...')
    bank = load_prototype_bank(checkpoint_path, device)
    if bank is None:
        raise RuntimeError('No prototype bank in the checkpoint.')
    bg_prototypes = get_background_prototypes(bank)

    if prototypes is None:
        prototypes = prototypes_dict_from_bank(bank, all_class_names)
    if not prototypes:
        raise RuntimeError('No usable prototypes for any class -- check support_files/class_id '
                           'or the checkpoint\'s prototype bank.')

    missing = [name for name in class_names if name not in prototypes]
    if missing:
        raise ValueError(f'No prototypes available for requested classes: {missing}')
    prototypes = {name: prototypes[name] for name in class_names}

    if rank == 0:
        print(f'Prototypes ready for: {list(prototypes.keys())}')

    tomo_transforms = build_tomo_transforms(data_cfg)
    patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0),
                           overlap=(0, 0.5, 0.5, 0.5))
    rename_map = {'z': 'rlnCoordinateZ', 'y': 'rlnCoordinateY', 'x': 'rlnCoordinateX',
                  'n_voxels': 'rlnNVoxels', 'mean_prob': 'rlnMeanProb', 'sum_prob': 'rlnSumProb',
                  'mean_sim': 'rlnMeanSim', 'sum_sim': 'rlnSumSim',
                  'class': 'rlnClassLabel', 'tomo': 'rlnMicrographName',
                  'instance_id': 'rlnInstanceId'}

    data_folder = cfg['data_folder']
    files = cfg.get('test_files') or [f for f in os.listdir(data_folder)
                                      if os.path.isfile(os.path.join(data_folder, f)) and f.endswith(file_ext)]
    files = [filename] if filename else sorted(files)
    n_files = len(files)

    if rank == 0:
        print(f'\n{n_files} test tomogram(s)')
        print('Node groups:')
        for node_id, host in enumerate(node_info['all_hosts']):
            print(f'  node {node_id}: {host}')

    if n_files == 0:
        if distributed:
            dist.destroy_process_group()
        return

    jobs = []
    for file_idx in range(node_info['node_id'], n_files, node_info['n_nodes']):
        jobs.append({'file_idx': file_idx, 'group': node_info['group'], 'group_rank': node_info['group_rank'],
                     'group_size': node_info['group_size'], 'leader_rank': node_info['leader_rank'],
                     'group_ranks': node_info['group_ranks']})

    if node_info['group_rank'] == 0:
        assigned_names = [os.path.basename(files[job['file_idx']]) for job in jobs]
        print(f'Node {node_info["node_id"]} ({node_info["hostname"]}) uses ranks {node_info["group_ranks"]} / '
              f'{node_info["group_size"]} GPU(s)')
        print(f'  assigned tomograms: {assigned_names if assigned_names else "none"}')
        print(f'  local temp dir: {shared_tmp_dir}')

    def postprocess_and_write(tomo_name, h5_out, ordered_names, seg_mask,
                              prototype_probs, prototype_sims, distance_maps, roi, reused):
        if roi is not None:
            seg_mask[~roi] = 0
            prototype_probs = apply_roi_mask(prototype_probs, roi)
            prototype_sims[:, ~roi] = -1
            distance_maps = apply_roi_mask(distance_maps, roi)

        if distance_maps is not None:
            positive_distance = distance_maps > 0
            prototype_probs[~positive_distance] = 0.0
            for k in range(len(ordered_names)):
                seg_mask[(seg_mask == k + 1) & (~positive_distance[k])] = 0

        if not reused and save_h5 != 'none':
            with h5py.File(h5_out, 'w') as hf:
                for k, name in enumerate(ordered_names):
                    if save_h5 in ('both', 'similarities'):
                        h5_create_dataset(hf, f'sim_{name}', prototype_sims[k])
                    if save_h5 in ('both', 'probabilities'):
                        h5_create_dataset(hf, f'prob_{name}', prototype_probs[k])
                    if distance_maps is not None:
                        h5_create_dataset(hf, f'distance_{name}', distance_maps[k])
                h5_create_dataset(hf, 'seg_mask_raw', seg_mask.astype(np.int16))
                hf.attrs['class_names'] = ordered_names
                hf.attrs['seg_mode'] = 'dual_head' if use_seg_head_probs else 'prototype'
                hf.attrs['temperature'] = temperature
                hf.attrs['similarity_type'] = 'raw_cosine'
                if distance_maps is not None:
                    hf.attrs['distance_type'] = 'signed_tanh'
            print(f'  Saved prediction maps: {os.path.basename(h5_out)}  ({save_h5})')

        print('  Extracting centers...')
        peaks_per_class, solid_masks, instance_assignments = extract_centers_per_class(
            prototype_probs, prototype_sims, ordered_names, seg_mask, min_voxels,
            merge_instances=merge_instances, postprocess=postprocess)

        if save_h5 != 'none':
            with h5py.File(h5_out, 'a') as hf:
                for name, solid_mask in solid_masks.items():
                    key = f'solid_{name}'
                    if key in hf:
                        del hf[key]
                    h5_create_dataset(hf, key, solid_mask.astype(np.uint8))

        tomo_combined = []
        for name, peaks in zip(ordered_names, peaks_per_class):
            if len(peaks) == 0:
                continue
            peaks['tomo'] = tomo_name
            tomo_combined.append(peaks)

        raw_csv = os.path.join(
            output_folder, f'{tomo_name}_all_centers.csv')
        raw_star = os.path.join(
            output_folder, f'{tomo_name}_all_centers.star')
        post_csv = os.path.join(
            output_folder, f'{tomo_name}_all_centers_postprocessed.csv')
        post_star = os.path.join(
            output_folder, f'{tomo_name}_all_centers_postprocessed.star')

        # Even when per-tomogram outputs are disabled, rank 0 needs a small
        # temporary CSV from each tomogram in order to build the joined files
        # across nodes. These hidden staging files are removed at the end.
        raw_join_csv = (
            raw_csv if save_per_tomogram_centers else
            os.path.join(output_folder, f'.{tomo_name}_all_centers.join.csv')
        )
        post_join_csv = (
            post_csv if save_per_tomogram_centers else
            os.path.join(
                output_folder,
                f'.{tomo_name}_all_centers_postprocessed.join.csv')
        )

        # Remove stale files from previous runs. When per-tomogram output is
        # disabled this also removes old visible files, leaving only the joined
        # all-tomogram outputs after the run.
        stale_paths = {
            raw_join_csv, post_join_csv,
            raw_csv, raw_star, post_csv, post_star,
        }
        for path in stale_paths:
            if os.path.exists(path):
                os.remove(path)

        kept_instances = pd.DataFrame(
            columns=['_instance_id', 'instance_id', 'class'])

        if tomo_combined:
            combined = pd.concat(tomo_combined, ignore_index=True)
            combined['tomo'] = tomo_name

            # Stable per-tomogram IDs are assigned once, before filtering.
            # Retained particles keep these IDs, so rlnInstanceId in the
            # STAR/CSV is exactly the label used in the final H5 instance_mask.
            # Gaps after filtering are intentional and harmless.
            combined = combined.reset_index(drop=True)
            combined['instance_id'] = np.arange(
                1, len(combined) + 1, dtype=np.int32)

            output_combined = centers_for_output(combined)
            output_combined.to_csv(raw_join_csv, index=False)

            if save_per_tomogram_centers:
                starfile.write(
                    centers_for_star(output_combined, rename_map),
                    raw_star,
                    overwrite=True)

            if save_individual_centers:
                for name, class_df in combined.groupby(
                        'class', sort=False):
                    output_class = centers_for_output(class_df)
                    starfile.write(
                        centers_for_star(output_class, rename_map),
                        os.path.join(
                            output_folder,
                            f'{tomo_name}_{name}_centers.star'),
                        overwrite=True)

            print(f'  {len(combined)} total particles')
            kept_instances = combined

            if filtering_enabled:
                print('  Applying class-specific postprocessing...')
                postprocessed = postprocess_instances(
                    combined, postprocess)
                output_postprocessed = centers_for_output(
                    postprocessed)
                output_postprocessed.to_csv(
                    post_join_csv, index=False)

                if (save_per_tomogram_centers and
                        len(output_postprocessed)):
                    starfile.write(
                        centers_for_star(
                            output_postprocessed, rename_map),
                        post_star,
                        overwrite=True)

                print(
                    f'  {len(postprocessed)} particles '
                    f'after postprocessing')
                kept_instances = postprocessed
        else:
            print('  No particles found in this tomogram.')

        if save_h5 != 'none':
            cleaned_seg_mask, final_instance_mask = build_output_masks(
                seg_mask, kept_instances, instance_assignments, ordered_names)
            with h5py.File(h5_out, 'a') as hf:
                if 'seg_mask_raw' not in hf:
                    h5_create_dataset(hf, 'seg_mask_raw', seg_mask.astype(np.int16))
                if 'seg_mask' in hf:
                    del hf['seg_mask']
                h5_create_dataset(hf, 'seg_mask', cleaned_seg_mask.astype(np.int16))
                if 'instance_mask' in hf:
                    del hf['instance_mask']
                h5_create_dataset(hf, 'instance_mask', final_instance_mask)
            print(f'  Saved cleaned seg_mask and instance_mask: {len(kept_instances)} retained instances')

    for job in jobs:
        current_file = os.path.join(data_folder, files[job['file_idx']])
        tomo_name = os.path.basename(current_file).split(file_ext)[0]
        h5_out = os.path.join(prediction_folder, f'{tomo_name}_preds.h5')
        leader = job['group_rank'] == 0

        reusable = None
        if leader and save_h5 != 'none' and reuse:
            reusable = load_reusable_prediction(h5_out, use_distance_head)
            if reusable is not None and list(reusable[0]) != list(class_names):
                reusable = None

        reused = group_broadcast_bool(reusable is not None, job['group'], job['leader_rank'],
                                      job['group_rank'], job['group_size'])
        if reused:
            if leader:
                print(f'\n{"=" * 55}')
                print(f'Tomogram: {tomo_name}  (reusing saved prediction maps)')
                ordered_names, seg_mask, prototype_probs, prototype_sims, distance_maps = reusable
                original_size = list(seg_mask.shape)
                roi = load_roi_mask(mask_folder, tomo_name, mask_dataset_key, original_size)
                postprocess_and_write(tomo_name, h5_out, ordered_names, seg_mask, prototype_probs,
                                      prototype_sims, distance_maps, roi, reused=True)
            if job['group_size'] > 1:
                dist.barrier(group=job['group'])
            continue

        sample = tomo_transforms({'image': current_file, 'file_name': current_file})
        img = sample['image']
        original_size = list(img[0].shape)
        roi = load_roi_mask(mask_folder, tomo_name, mask_dataset_key, original_size)
        padded_size = [padded_size_for_even_tiling(original_size[i], patch_size[i]) for i in range(3)]
        img = SpatialPad(spatial_size=padded_size, method='end', mode='edge')(img)
        input_size = list(img[0].shape)

        prediction_roi = None
        if roi is not None:
            prediction_roi = np.zeros(input_size, dtype=bool)
            prediction_roi[:original_size[0], :original_size[1], :original_size[2]] = roi

        if leader:
            print(f'\n{"=" * 55}')
            print(f'Tomogram: {tomo_name}  size: {original_size}')
            print(f'  Extracting predictions on {job["group_size"]} GPU(s)...')

        sim_maps_raw, probs_all_raw, distance_maps_raw, ordered_names, sim_stride, shared_paths = \
            extract_predictions(
                img, backbone, decoder, temperature, prototypes, bg_prototypes, class_names,
                patch_size, input_size, batch_size, patch_iter, device, tomo_name=tomo_name,
                roi_mask=prediction_roi, all_class_names=all_class_names,
                use_seg_head_probs=use_seg_head_probs, process_group=job['group'],
                group_rank=job['group_rank'], group_size=job['group_size'],
                group_leader_rank=job['leader_rank'], shared_tmp_dir=shared_tmp_dir,
                inference_workers=inference_workers)

        if not leader:
            if job['group_size'] > 1:
                dist.barrier(group=job['group'])
            continue

        sim_maps = upsample_sim_maps(sim_maps_raw, sim_stride, original_size)
        probs_all = upsample_sim_maps(probs_all_raw, sim_stride, original_size)
        distance_maps = upsample_sim_maps(distance_maps_raw, sim_stride, original_size)
        prototype_probs = probs_all[1:]
        prototype_sims = sim_maps[1:]
        seg_mask = extract_seg_mask(probs_all, threshold=threshold)

        postprocess_and_write(tomo_name, h5_out, ordered_names, seg_mask, prototype_probs,
                              prototype_sims, distance_maps, roi, reused=False)

        close_memmap(sim_maps_raw)
        close_memmap(probs_all_raw)
        close_memmap(distance_maps_raw)
        if job['group_size'] > 1:
            dist.barrier(group=job['group'])
            cleanup_shared_prediction_files(shared_paths)

    if distributed:
        dist.barrier()

    if node_info['group_rank'] == 0:
        try:
            os.rmdir(shared_tmp_dir)
        except OSError:
            pass

    if distributed:
        dist.barrier()

    if rank == 0:
        combined_frames, post_frames = [], []
        for file in files:
            tomo_name = os.path.basename(file).split(file_ext)[0]

            raw_csv = os.path.join(
                output_folder, f'{tomo_name}_all_centers.csv')
            raw_join_csv = (
                raw_csv if save_per_tomogram_centers else
                os.path.join(
                    output_folder,
                    f'.{tomo_name}_all_centers.join.csv')
            )
            if os.path.exists(raw_join_csv):
                df = pd.read_csv(raw_join_csv)
                if len(df):
                    combined_frames.append(df)

            if filtering_enabled:
                post_csv = os.path.join(
                    output_folder,
                    f'{tomo_name}_all_centers_postprocessed.csv')
                post_join_csv = (
                    post_csv if save_per_tomogram_centers else
                    os.path.join(
                        output_folder,
                        f'.{tomo_name}_all_centers_postprocessed.join.csv')
                )
                if os.path.exists(post_join_csv):
                    df = pd.read_csv(post_join_csv)
                    if len(df):
                        post_frames.append(df)

        out_all = os.path.join(output_folder, 'all_tomograms_centers.csv')
        out_all_star = os.path.join(output_folder, 'all_tomograms_centers.star')
        if combined_frames:
            final = pd.concat(combined_frames, ignore_index=True)
            final.to_csv(out_all, index=False)
            starfile.write(centers_for_star(final, rename_map), out_all_star, overwrite=True)
            print(f'\nDone. {len(final)} particles across {n_files} tomogram(s) -> {out_all}')
        else:
            for path in (out_all, out_all_star):
                if os.path.exists(path):
                    os.remove(path)
            print('\nNo particles found in any tomogram.')

        if filtering_enabled:
            out_post = os.path.join(output_folder, 'all_tomograms_centers_postprocessed.csv')
            out_post_star = os.path.join(output_folder, 'all_tomograms_centers_postprocessed.star')
            if post_frames:
                final_post = pd.concat(post_frames, ignore_index=True)
                final_post.to_csv(out_post, index=False)
                starfile.write(centers_for_star(final_post, rename_map), out_post_star, overwrite=True)
                print(f'Postprocessed: {len(final_post)} particles -> {out_post}')
            else:
                for path in (out_post, out_post_star):
                    if os.path.exists(path):
                        os.remove(path)

        if not save_per_tomogram_centers:
            for file in files:
                tomo_name = os.path.basename(file).split(file_ext)[0]
                staging_paths = (
                    os.path.join(
                        output_folder,
                        f'.{tomo_name}_all_centers.join.csv'),
                    os.path.join(
                        output_folder,
                        f'.{tomo_name}_all_centers_postprocessed.join.csv'),
                )
                for path in staging_paths:
                    if os.path.exists(path):
                        os.remove(path)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def maybe_launch_distributed_from_config(config_file_path, filename=None):
    """Auto-launch local workers for one node; multi-node jobs are launched externally."""
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        return False

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)
    nodes, gpus_per_node = read_requested_topology(cfg)

    if nodes > 1:
        raise RuntimeError(f'The config requests parameters.nodes={nodes} with '
                           f'parameters.gpu_devices={gpus_per_node} GPUs per node. Multi-node jobs must be '
                           f'launched across the allocated nodes with scheduler/torchrun '
                           f'(expected WORLD_SIZE={nodes * gpus_per_node}).')
    if gpus_per_node <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError(f'parameters.gpu_devices={gpus_per_node}, but CUDA is not available.')
    if gpus_per_node > torch.cuda.device_count():
        raise RuntimeError(f'parameters.gpu_devices={gpus_per_node}, but only '
                           f'{torch.cuda.device_count()} CUDA device(s) are visible.')

    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--max_restarts=0',
           f'--nproc_per_node={gpus_per_node}', os.path.abspath(__file__),
           '--config_file', config_file_path]
    if filename is not None:
        cmd.extend(['--filename', filename])
    print(f'Launching distributed prototype inference on 1 node with {gpus_per_node} GPU worker(s)...')
    launch_env = os.environ.copy()
    launch_env.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run(cmd, check=True, env=launch_env)
    return True


def main(config_file_path, filename=None):
    if not maybe_launch_distributed_from_config(config_file_path, filename):
        _main(config_file_path, filename)


if __name__ == '__main__':
    parser = parser_helper('Prototype refinement: zero/few-shot prediction')
    args = parser.parse_args()
    main(args.config_file, getattr(args, 'filename', None))
