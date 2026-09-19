import os
import sys
import csv
import uuid
import glob
import yaml
import h5py
import socket
import subprocess
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from skimage.measure import regionprops_table
from monai.transforms import (
    Compose,
    SpatialPad,
    LoadImaged,
    EnsureTyped,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, PatchIter
from cryosiam.apps.dense_simsiam_instance import load_backbone_model, load_prediction_model, instance_segmentation


def combine_instance_region_csvs(prediction_folder, output_csv="all_instance_regions.csv"):
    csv_files = sorted(glob.glob(os.path.join(prediction_folder, "*_instance_regions.csv")))

    if len(csv_files) == 0:
        print("No instance region CSV files found.")
        return None

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        base = os.path.basename(csv_file)
        tomo_name = base.replace("_instance_regions.csv", "")
        df["tomo"] = tomo_name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(prediction_folder, output_csv)
    merged.to_csv(out_path, index=False)
    print(f"Saved merged CSV to {out_path}")
    return merged


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


def patch_grid_count(input_size, patch_size, overlap=0.5):
    counts = []
    for size, patch in zip(input_size, patch_size):
        stride = max(1, int(round(patch * (1.0 - overlap))))
        if size <= patch:
            counts.append(1)
        else:
            counts.append(int(np.ceil((size - patch) / stride)) + 1)
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
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')

    return distributed, rank, world_size, local_rank, device


def build_node_process_group(distributed, rank, world_size):
    """Build one process group per physical node so one tomogram never spans nodes."""
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

    groups = {}
    for host in host_order:
        groups[host] = dist.new_group(ranks=host_to_ranks[host])

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
    """Create the same run-specific name inside each node-local temp directory."""
    if distributed:
        run_id = [uuid.uuid4().hex[:12] if rank == 0 else None]
        dist.broadcast_object_list(run_id, src=0)
        run_id = run_id[0]
    else:
        run_id = uuid.uuid4().hex[:12]

    path = os.path.join(temp_root, f'cryosiam_instance_{run_id}')
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def shared_prediction_path(tmp_dir, tomo_name, leader_rank):
    return os.path.join(
        tmp_dir, f'.{safe_name(tomo_name)}.instance_group{leader_rank}.preds.f32')


def open_shared_prediction(path, input_size, create=False):
    mode = 'w+' if create else 'r+'
    predictions = np.memmap(
        path, dtype=np.float32, mode=mode, shape=(3, *input_size))
    if create:
        predictions.flush()
    return predictions


def close_memmap(arr):
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


def cleanup_shared_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def reusable_prediction(path):
    if not os.path.isfile(path):
        return False

    try:
        with h5py.File(path, 'r') as f:
            return all(key in f for key in ('foreground', 'distances', 'boundaries'))
    except OSError:
        return False


def load_reusable_prediction(path):
    with h5py.File(path, 'r') as f:
        return f['foreground'][()], f['distances'][()], f['boundaries'][()]


def broadcast_bool(value, process_group, leader_rank, group_rank, group_size):
    if group_size == 1:
        return bool(value)

    obj = [bool(value) if group_rank == 0 else None]
    dist.broadcast_object_list(obj, src=leader_rank, group=process_group)
    return bool(obj[0])


def patch_slices(c_batch, input_size, patch_size):
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


def distributed_patch_prediction(img, backbone, prediction_model, patch_iter, patch_size, input_size, batch_size,
                                 device, tomo_name, process_group=None, group_rank=0, group_size=1, leader_rank=0,
                                 shared_tmp_dir=None, inference_workers=0):
    """Predict one tomogram while sharding its patches over all GPUs on one node."""
    distributed_tomo = group_size > 1
    shared_path = None

    if distributed_tomo:
        os.makedirs(shared_tmp_dir, exist_ok=True)
        shared_path = shared_prediction_path(
            shared_tmp_dir, tomo_name, leader_rank)

        if group_rank == 0:
            cleanup_shared_file(shared_path)
            predictions = open_shared_prediction(
                shared_path, input_size, create=True)
        else:
            predictions = None

        dist.barrier(group=process_group)

        if group_rank != 0:
            predictions = open_shared_prediction(
                shared_path, input_size, create=False)
    else:
        predictions = np.zeros((3, *input_size), dtype=np.float32)

    patch_dataset = ShardedPatchDataset(
        img, patch_iter, shard_rank=group_rank, num_shards=group_size)

    loader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        num_workers=int(inference_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(inference_workers > 0)
    )

    progress_total = local_patch_count(
        input_size, patch_size, group_rank, group_size, inference_workers)
    progress = tqdm(
        total=progress_total,
        desc=f'  {tomo_name} | leader GPU shard',
        unit='patch',
        leave=False,
        disable=(group_rank != 0)
    )

    with torch.inference_mode():
        for item in loader:
            patches = item[0].to(device, non_blocking=True)
            coords = item[1].numpy().astype(int)

            if group_rank == 0:
                progress.update(patches.shape[0])

            z, _ = backbone.forward_predict(patches)
            foreground_pred, distance_pred, boundaries_pred = prediction_model(z)
            foreground_pred = torch.sigmoid(foreground_pred).float().cpu().numpy()
            distance_pred = distance_pred.float().cpu().numpy()
            boundaries_pred = torch.sigmoid(boundaries_pred).float().cpu().numpy()

            for batch_i in range(patches.shape[0]):
                c_batch = coords[batch_i][1:]

                if any(
                        c_batch[d][0] >= input_size[d] - patch_size[d] // 4
                        for d in range(len(input_size))):
                    continue

                slices, slices2 = patch_slices(
                    c_batch, input_size, patch_size)
                predictions[(0,) + slices] = \
                    foreground_pred[batch_i][0][slices2]
                predictions[(1,) + slices] = \
                    distance_pred[batch_i][0][slices2]
                predictions[(2,) + slices] = \
                    boundaries_pred[batch_i][0][slices2]

    if group_rank == 0:
        progress.close()

    if isinstance(predictions, np.memmap):
        predictions.flush()

    if distributed_tomo:
        dist.barrier(group=process_group)

        if group_rank != 0:
            close_memmap(predictions)
            return None, shared_path

    return predictions, shared_path


def read_requested_topology(cfg):
    """Read the requested physical node count and GPUs per node."""
    parameters = cfg.get('parameters', {})
    nodes_value = parameters.get('nodes', 1)
    gpu_devices_value = parameters.get('gpu_devices', 1)

    try:
        n_nodes = int(nodes_value)
    except (TypeError, ValueError):
        raise ValueError(
            f'parameters.nodes must be an integer number of physical nodes, got {nodes_value!r}.')

    try:
        gpus_per_node = int(gpu_devices_value)
    except (TypeError, ValueError):
        raise ValueError(
            f'parameters.gpu_devices must be an integer number of GPUs per node, got {gpu_devices_value!r}.')

    if n_nodes < 1:
        raise ValueError(
            f'parameters.nodes must be >= 1, got {n_nodes}.')
    if gpus_per_node < 1:
        raise ValueError(
            f'parameters.gpu_devices must be >= 1, got {gpus_per_node}.')

    return n_nodes, gpus_per_node


def maybe_launch_distributed_from_config(config_file_path):
    """Auto-launch workers on one node; multi-node jobs must be launched externally."""
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        return False

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    n_nodes, gpus_per_node = read_requested_topology(cfg)

    if n_nodes > 1:
        raise RuntimeError(
            f'The config requests parameters.nodes={n_nodes} with parameters.gpu_devices={gpus_per_node} GPUs '
            'per node. Multi-node jobs must first be launched across the allocated nodes with your scheduler/torchrun '
            f'so that RANK, LOCAL_RANK and WORLD_SIZE are set. The expected WORLD_SIZE is {n_nodes * gpus_per_node}.')

    if gpus_per_node <= 1:
        return False

    if not torch.cuda.is_available():
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but CUDA is not available.')

    visible_gpus = torch.cuda.device_count()
    if gpus_per_node > visible_gpus:
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but only {visible_gpus} CUDA device(s) are visible. '
            'Check the allocation and CUDA_VISIBLE_DEVICES.')

    cmd = [
        sys.executable,
        '-m', 'torch.distributed.run',
        '--standalone',
        '--max_restarts=0',
        f'--nproc_per_node={gpus_per_node}',
        os.path.abspath(__file__),
        *sys.argv[1:]
    ]

    print(
        f'Launching distributed instance inference on 1 node with '
        f'{gpus_per_node} GPU worker(s)...')

    launch_env = os.environ.copy()
    launch_env.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run(cmd, check=True, env=launch_env)
    return True


def main(config_file_path, filename=None):
    distributed, rank, world_size, local_rank, device = init_distributed()

    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if 'trained_model' in cfg and cfg['trained_model'] is not None:
        checkpoint_path = cfg['trained_model']
    else:
        checkpoint_path = os.path.join(
            cfg['log_dir'], 'model', 'model_best.ckpt')

    backbone = load_backbone_model(checkpoint_path, device)
    prediction_model = load_prediction_model(checkpoint_path, device)
    backbone.eval()
    prediction_model.eval()

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    net_config = checkpoint['hyper_parameters']['config']

    test_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    reuse_predictions = bool(cfg.get('reuse_predictions', True))
    semantic_predictions = cfg.get('semantic_predictions') or None
    semantic_labels = cfg.get('semantic_foreground_labels') or None
    mask_folder = cfg.get('mask_folder') or None
    patch_size = list(net_config['parameters']['data']['patch_size'])
    spatial_dims = int(net_config['parameters']['network']['spatial_dims'])
    batch_size = int(cfg['hyper_parameters']['batch_size'])
    inference_workers = int(cfg.get('inference_workers', 0))
    temp_root = cfg.get('temp_dir', prediction_folder)

    os.makedirs(prediction_folder, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    requested_nodes, requested_gpus_per_node = read_requested_topology(cfg)
    node_info = build_node_process_group(
        distributed, rank, world_size)

    expected_world_size = requested_nodes * requested_gpus_per_node
    actual_world_size = world_size if distributed else 1

    if actual_world_size != expected_world_size:
        raise RuntimeError(
            f'Distributed topology mismatch: config requests {requested_nodes} node(s) x '
            f'{requested_gpus_per_node} GPU(s) per node = {expected_world_size} process(es), '
            f'but WORLD_SIZE is {actual_world_size}. Check the scheduler/torchrun launch.')

    if node_info['n_nodes'] != requested_nodes:
        raise RuntimeError(
            f'Distributed topology mismatch: config requests parameters.nodes={requested_nodes}, but the launched '
            f'ranks span {node_info["n_nodes"]} physical hostname(s): {node_info["all_hosts"]}.')

    if node_info['group_size'] != requested_gpus_per_node:
        raise RuntimeError(
            f'Distributed topology mismatch on host {node_info["hostname"]}: config requests '
            f'{requested_gpus_per_node} GPU process(es) per node, but this node has {node_info["group_size"]} ranks.')

    shared_tmp_dir = make_run_temp_dir(
        temp_root, distributed, rank)

    files = cfg.get('test_files')
    if files is None:
        files = [
            x for x in os.listdir(test_folder)
            if os.path.isfile(os.path.join(test_folder, x))
               and x.endswith(cfg['file_extension'])
        ]

    if filename:
        files = [filename]

    files = sorted(files)

    if rank == 0:
        print(f'Using {world_size if distributed else 1} GPU process(es)')
        print(f'Test tomograms: {len(files)}')
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

    transforms = Compose([
        LoadImaged(keys='image', reader=reader),
        EnsureChannelFirstd(keys='image'),
        NumpyToTensord(keys='image'),
        ScaleIntensityRanged(
            keys='image',
            a_min=cfg['parameters']['data']['min'],
            a_max=cfg['parameters']['data']['max'],
            b_min=0, b_max=1, clip=True
        ),
        NormalizeIntensityd(
            keys='image',
            subtrahend=cfg['parameters']['data']['mean'],
            divisor=cfg['parameters']['data']['std']
        ),
        EnsureTyped(keys='image', data_type='tensor')
    ])

    if spatial_dims == 2:
        patch_iter = PatchIter(
            patch_size=tuple(patch_size),
            start_pos=(0, 0),
            overlap=(0, 0.5, 0.5)
        )
    else:
        patch_iter = PatchIter(
            patch_size=tuple(patch_size),
            start_pos=(0, 0, 0),
            overlap=(0, 0.5, 0.5, 0.5)
        )

    pad_transform = SpatialPad(
        spatial_size=patch_size, method='end', mode='constant')

    jobs = []
    for file_idx in range(
            node_info['node_id'], len(files), node_info['n_nodes']):
        jobs.append({
            'file_idx': file_idx,
            'group': node_info['group'],
            'group_rank': node_info['group_rank'],
            'group_size': node_info['group_size'],
            'leader_rank': node_info['leader_rank']
        })

    if node_info['group_rank'] == 0:
        assigned_names = [
            os.path.basename(files[job['file_idx']])
            for job in jobs
        ]
        print(
            f'Node {node_info["node_id"]} ({node_info["hostname"]}) '
            f'uses ranks {node_info["group_ranks"]} / '
            f'{node_info["group_size"]} GPU(s)')
        print(
            f'  assigned tomograms: '
            f'{assigned_names if assigned_names else "none"}')
        print(f'  local temp dir: {shared_tmp_dir}')

    for job in jobs:
        file = files[job['file_idx']]
        current_file = os.path.join(test_folder, file)
        root = os.path.basename(file).split(cfg['file_extension'])[0]
        out_file = os.path.join(
            prediction_folder, os.path.basename(file))
        raw_preds_h5 = os.path.join(
            prediction_folder, f'{root}_preds.h5')
        leader = job['group_rank'] == 0

        can_reuse = (
                leader and reuse_predictions and
                reusable_prediction(raw_preds_h5)
        )
        can_reuse = broadcast_bool(can_reuse,
                                   job['group'],
                                   job['leader_rank'],
                                   job['group_rank'],
                                   job['group_size']
                                   )

        shared_path = None
        predictions = None
        shared_handle = None

        if can_reuse:
            if leader:
                print(
                    f'Loading saved predictions for file '
                    f'{current_file}')
                foreground_out, distances_out, boundaries_out = \
                    load_reusable_prediction(raw_preds_h5)
                original_size = list(foreground_out.shape)
            else:
                if job['group_size'] > 1:
                    dist.barrier(group=job['group'])
                continue
        else:
            sample = transforms({
                'image': current_file,
                'file_name': current_file
            })

            original_size = list(sample['image'][0].shape)
            img = pad_transform(sample['image'])
            input_size = list(img[0].shape)

            if leader:
                print(
                    f'Running prediction for file {current_file}')
                print(
                    f'  size={original_size}, padded={input_size}, '
                    f'GPUs={job["group_size"]}')

            predictions, shared_path = \
                distributed_patch_prediction(img=img,
                                             backbone=backbone,
                                             prediction_model=prediction_model,
                                             patch_iter=patch_iter,
                                             patch_size=patch_size,
                                             input_size=input_size,
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
                if job['group_size'] > 1:
                    dist.barrier(group=job['group'])
                continue

            shared_handle = (predictions if isinstance(predictions, np.memmap) else None)

            crop = tuple(slice(0, n) for n in original_size)
            foreground_out = predictions[(0,) + crop]
            distances_out = predictions[(1,) + crop]
            boundaries_out = predictions[(2,) + crop]

            if mask_folder:
                print(f'Masking out of file {current_file}')
                mask_name = f'{root}_preds.h5'
                with h5py.File(os.path.join(mask_folder, mask_name), 'r') as f:
                    mask = f['labels'][()].astype(np.int8)

                if list(mask.shape) != list(original_size):
                    mask = resize(mask, original_size, mode='constant', preserve_range=True).astype(np.int8)

                foreground_out = foreground_out * mask
                distances_out = distances_out * mask
                boundaries_out = boundaries_out * mask

            if cfg['save_raw_predictions']:
                with h5py.File(raw_preds_h5, 'w') as f:
                    f.create_dataset('foreground', data=np.asarray(foreground_out))
                    f.create_dataset('distances', data=np.asarray(distances_out))
                    f.create_dataset('boundaries', data=np.asarray(boundaries_out))

        if semantic_predictions:
            semantic_file = os.path.join(semantic_predictions, os.path.basename(file))
            with h5py.File(semantic_file.split(cfg['file_extension'])[0] + '_preds.h5', 'r') as f:
                foreground_out = f['labels'][()]

            foreground_out = np.isin(foreground_out, semantic_labels).astype(np.float32)

        min_dist = cfg['parameters']['network']['min_center_distance']
        max_dist = cfg['parameters']['network']['max_center_distance']
        postprocessing = cfg['parameters']['network'].get(
            'postprocessing', True)

        print(
            f'Running instance segmentation for file '
            f'{current_file}')

        instance_labels = instance_segmentation(foreground_out,
                                                distances_out,
                                                boundaries_out,
                                                threshold_min=min_dist,
                                                threshold_max=max_dist,
                                                boundary_bias=cfg['parameters']['network']['boundary_bias'],
                                                assignment_threshold=cfg['parameters']['network'][
                                                    'threshold_foreground'],
                                                distance_type=cfg['parameters']['network']['distance_type'],
                                                postprocessing=postprocessing)

        print(f'Saving predictions for file {current_file}')
        instance_h5 = os.path.join(
            prediction_folder,
            f'{root}_instance_preds.h5'
        )
        with h5py.File(instance_h5, 'w') as f:
            f.create_dataset('instances', data=instance_labels)

        regions = regionprops_table(instance_labels, properties=['label', 'area', 'bbox', 'centroid'])
        regions_file = os.path.join(
            prediction_folder,
            f'{root}_instance_regions.csv'
        )
        with open(regions_file, 'w') as f:
            w = csv.writer(f)
            w.writerow(list(regions.keys()))
            for ind in range(
                    regions['label'].shape[0]):
                w.writerow([
                    regions[key][ind]
                    for key in regions.keys()
                ])

        if shared_handle is not None:
            close_memmap(shared_handle)

        if job['group_size'] > 1:
            dist.barrier(group=job['group'])
            cleanup_shared_file(shared_path)

    if distributed:
        dist.barrier()

    if rank == 0:
        combine_instance_region_csvs(
            prediction_folder,
            "all_instance_regions.csv"
        )

    if distributed:
        dist.barrier()

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
