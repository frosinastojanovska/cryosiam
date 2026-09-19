import os
import sys
import csv
import uuid
import yaml
import h5py
import socket
import subprocess
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    SpatialPad,
    EnsureChannelFirstd,
    EnsureTyped
)

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, PatchIter, MrcWriter
from cryosiam.transforms import ScaleIntensityd, InvertIntensityd
from cryosiam.apps.dense_simsiam_regression import load_backbone_model, load_prediction_model


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
    """Create one distributed process group per physical node."""
    hostname = socket.gethostname()
    if not distributed:
        return {'hostname': hostname, 'node_id': 0, 'n_nodes': 1, 'group': None, 'group_rank': 0,
                'group_size': 1, 'leader_rank': 0, 'group_ranks': [0], 'all_hosts': [hostname]}

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
    return {'hostname': hostname, 'node_id': host_order.index(hostname), 'n_nodes': len(host_order),
            'group': groups[hostname], 'group_rank': group_ranks.index(rank), 'group_size': len(group_ranks),
            'leader_rank': group_ranks[0], 'group_ranks': group_ranks, 'all_hosts': host_order}


def make_run_temp_dir(temp_root, distributed, rank):
    """Create the same run-specific directory name in each node-local temp root."""
    if distributed:
        run_id = [uuid.uuid4().hex[:12] if rank == 0 else None]
        dist.broadcast_object_list(run_id, src=0)
        run_id = run_id[0]
    else:
        run_id = uuid.uuid4().hex[:12]

    path = os.path.join(temp_root, f'cryosiam_regression_{run_id}')
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def shared_prediction_path(tmp_dir, tomo_name, leader_rank):
    return os.path.join(tmp_dir, f'.{safe_name(tomo_name)}.regression_group{leader_rank}.preds.f32')


def open_shared_prediction(path, num_output_channels, input_size, create=False):
    mode = 'w+' if create else 'r+'
    preds = np.memmap(path, dtype=np.float32, mode=mode, shape=(num_output_channels, *input_size))
    if create:
        preds.flush()
    return preds


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


def distributed_patch_prediction(img, backbone, prediction_model, patch_iter, patch_size, input_size,
                                 num_output_channels, batch_size, device, tomo_name, process_group=None,
                                 group_rank=0, group_size=1, leader_rank=0, shared_tmp_dir=None,
                                 inference_workers=0):
    """Predict one tomogram while sharding its patches over all GPUs on one node."""
    distributed_tomo = group_size > 1
    shared_path = None

    if distributed_tomo:
        os.makedirs(shared_tmp_dir, exist_ok=True)
        shared_path = shared_prediction_path(shared_tmp_dir, tomo_name, leader_rank)
        if group_rank == 0:
            cleanup_shared_file(shared_path)
            preds_out = open_shared_prediction(shared_path, num_output_channels, input_size, create=True)
        else:
            preds_out = None

        dist.barrier(group=process_group)
        if group_rank != 0:
            preds_out = open_shared_prediction(shared_path, num_output_channels, input_size, create=False)
    else:
        preds_out = np.zeros((num_output_channels, *input_size), dtype=np.float32)

    patch_dataset = ShardedPatchDataset(img, patch_iter, shard_rank=group_rank, num_shards=group_size)
    loader = DataLoader(patch_dataset, batch_size=batch_size, num_workers=int(inference_workers),
                        pin_memory=torch.cuda.is_available(), persistent_workers=bool(inference_workers > 0))

    progress_total = local_patch_count(input_size, patch_size, group_rank, group_size, inference_workers)
    progress = tqdm(total=progress_total, desc=f'  {tomo_name} | leader GPU shard', unit='patch',
                    leave=False, disable=(group_rank != 0))

    with torch.inference_mode():
        for item in loader:
            patches = item[0].to(device, non_blocking=True)
            coords = item[1].numpy().astype(int)
            if group_rank == 0:
                progress.update(patches.shape[0])

            z, _ = backbone.forward_predict(patches)
            out = prediction_model(z).float().cpu().numpy()

            for batch_i in range(patches.shape[0]):
                c_batch = coords[batch_i][1:]
                if any(c_batch[d][0] >= input_size[d] - patch_size[d] // 4 for d in range(len(input_size))):
                    continue

                slices, slices2 = patch_slices(c_batch, input_size, patch_size)
                preds_out[(slice(None),) + slices] = out[batch_i][(slice(None),) + slices2]

    if group_rank == 0:
        progress.close()

    if isinstance(preds_out, np.memmap):
        preds_out.flush()

    if distributed_tomo:
        dist.barrier(group=process_group)
        if group_rank != 0:
            close_memmap(preds_out)
            return None, shared_path

    return preds_out, shared_path


def read_requested_topology(cfg):
    """Read physical node count and GPUs per node from the YAML configuration."""
    parameters = cfg.get('parameters', {})
    nodes_value = parameters.get('nodes', 1)
    gpu_devices_value = parameters.get('gpu_devices', 1)

    try:
        n_nodes = int(nodes_value)
    except (TypeError, ValueError):
        raise ValueError(f'parameters.nodes must be an integer number of physical nodes, got {nodes_value!r}.')

    try:
        gpus_per_node = int(gpu_devices_value)
    except (TypeError, ValueError):
        raise ValueError(
            f'parameters.gpu_devices must be an integer number of GPUs per node, got {gpu_devices_value!r}.')

    if n_nodes < 1:
        raise ValueError(f'parameters.nodes must be >= 1, got {n_nodes}.')
    if gpus_per_node < 1:
        raise ValueError(f'parameters.gpu_devices must be >= 1, got {gpus_per_node}.')
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
            f'The config requests parameters.nodes={n_nodes} with parameters.gpu_devices={gpus_per_node} GPUs per node. '
            'Multi-node jobs must first be launched across the allocated nodes with your scheduler/torchrun so that '
            f'RANK, LOCAL_RANK and WORLD_SIZE are set. The expected WORLD_SIZE is {n_nodes * gpus_per_node}.')

    if gpus_per_node <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError(f'parameters.gpu_devices={gpus_per_node}, but CUDA is not available.')

    visible_gpus = torch.cuda.device_count()
    if gpus_per_node > visible_gpus:
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but only {visible_gpus} CUDA device(s) are visible. '
            'Check the allocation and CUDA_VISIBLE_DEVICES.')

    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--max_restarts=0',
           f'--nproc_per_node={gpus_per_node}', os.path.abspath(__file__), *sys.argv[1:]]

    print(f'Launching distributed regression inference on 1 node with {gpus_per_node} GPU worker(s)...')
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

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    net_config = checkpoint['hyper_parameters']['config']

    test_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    num_output_channels = int(net_config['parameters']['network']['n_output_channels'])
    patch_size = list(cfg['parameters']['data']['patch_size'])
    spatial_dims = int(net_config['parameters']['network']['spatial_dims'])
    batch_size = int(cfg['hyper_parameters']['batch_size'])
    inference_workers = int(cfg.get('inference_workers', 0))
    temp_root = cfg.get('temp_dir', prediction_folder)

    os.makedirs(prediction_folder, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    requested_nodes, requested_gpus_per_node = read_requested_topology(cfg)
    node_info = build_node_process_group(distributed, rank, world_size)
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

    shared_tmp_dir = make_run_temp_dir(temp_root, distributed, rank)

    if filename:
        files = [filename]
    else:
        files = cfg.get('test_files')
        if files is None:
            files = [x for x in os.listdir(test_folder)
                     if os.path.isfile(os.path.join(test_folder, x)) and x.endswith(cfg['file_extension'])]
    files = sorted(files)

    if rank == 0:
        print(f'Using {world_size if distributed else 1} GPU process(es)')
        print(f'Test tomograms: {len(files)}')
        print(f'Requested topology: {requested_nodes} node(s) x {requested_gpus_per_node} GPU(s) per node = '
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
        InvertIntensityd(keys='image'),
        ScaleIntensityd(keys='image'),
        ScaleIntensityRanged(keys='image', a_min=cfg['parameters']['data']['min'],
                             a_max=cfg['parameters']['data']['max'], b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys='image', subtrahend=cfg['parameters']['data']['mean'],
                            divisor=cfg['parameters']['data']['std']),
        EnsureTyped(keys='image', data_type='tensor')
    ])

    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0), overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0), overlap=(0, 0.5, 0.5, 0.5))
    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='constant')

    jobs = []
    for file_idx in range(node_info['node_id'], len(files), node_info['n_nodes']):
        jobs.append({'file_idx': file_idx, 'group': node_info['group'], 'group_rank': node_info['group_rank'],
                     'group_size': node_info['group_size'], 'leader_rank': node_info['leader_rank']})

    if node_info['group_rank'] == 0:
        assigned_names = [os.path.basename(files[job['file_idx']]) for job in jobs]
        print(f'Node {node_info["node_id"]} ({node_info["hostname"]}) uses ranks {node_info["group_ranks"]} / '
              f'{node_info["group_size"]} GPU(s)')
        print(f'  assigned tomograms: {assigned_names if assigned_names else "none"}')
        print(f'  local temp dir: {shared_tmp_dir}')

    for job in jobs:
        file = files[job['file_idx']]
        current_file = os.path.join(test_folder, file)
        root = os.path.basename(file).split(cfg['file_extension'])[0]
        out_file = os.path.join(prediction_folder, os.path.basename(file))
        leader = job['group_rank'] == 0

        sample = transforms({'image': current_file, 'file_name': current_file})
        original_size = list(sample['image'][0].shape)
        img = pad_transform(sample['image'])
        input_size = list(img[0].shape)

        if leader:
            print(f'Running prediction for file {current_file}')
            print(f'  size={original_size}, padded={input_size}, GPUs={job["group_size"]}')

        preds_out, shared_path = distributed_patch_prediction(
            img=img, backbone=backbone, prediction_model=prediction_model, patch_iter=patch_iter,
            patch_size=patch_size, input_size=input_size, num_output_channels=num_output_channels,
            batch_size=batch_size, device=device, tomo_name=root, process_group=job['group'],
            group_rank=job['group_rank'], group_size=job['group_size'], leader_rank=job['leader_rank'],
            shared_tmp_dir=shared_tmp_dir, inference_workers=inference_workers)

        if not leader:
            if job['group_size'] > 1:
                dist.barrier(group=job['group'])
            continue

        shared_handle = preds_out if isinstance(preds_out, np.memmap) else None
        crop = tuple(slice(0, n) for n in original_size)
        preds_out = preds_out[(slice(None),) + crop]

        if cfg['scale_prediction']:
            preds_out = (preds_out - preds_out.min()) / (preds_out.max() - preds_out.min())

        if cfg['save_raw_predictions']:
            with h5py.File(os.path.join(prediction_folder, f'{root}_preds.h5'), 'w') as f:
                f.create_dataset('preds', data=np.asarray(preds_out))

        print(f'Saving predictions for file {current_file}')
        voxel_size = reader.read(current_file).voxel_size
        writer = MrcWriter(output_dtype=np.float32, overwrite=True)
        writer.set_metadata({'voxel_size': voxel_size})
        writer.set_data_array(np.asarray(preds_out[0]), channel_dim=None)
        writer.write(out_file.split(cfg['file_extension'])[0] + f'{cfg["file_extension"]}')

        if shared_handle is not None:
            close_memmap(shared_handle)
        elif isinstance(preds_out, np.memmap):
            close_memmap(preds_out)

        if job['group_size'] > 1:
            dist.barrier(group=job['group'])
            cleanup_shared_file(shared_path)

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
    parser = parser_helper("Run CryoSiam denoising prediction")
    args = parser.parse_args()
    if not maybe_launch_distributed_from_config(args.config_file):
        main(args.config_file, getattr(args, 'filename', None))
