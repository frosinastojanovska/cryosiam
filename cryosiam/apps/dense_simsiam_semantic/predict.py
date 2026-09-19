import os
import sys
import uuid
import yaml
import h5py
import socket
import subprocess
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
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
                f'LOCAL_RANK={local_rank}, but only '
                f'{torch.cuda.device_count()} CUDA device(s) are visible.'
            )

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
    Build one process group per physical node.

    A tomogram is never split across nodes. All GPU processes on one
    physical node cooperate on the same tomogram at a time.
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
                'all_hosts': [hostname]
                }

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

    # All ranks must create all groups in the same order.
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
            'all_hosts': host_order
            }


def make_run_temp_dir(temp_root, distributed, rank):
    """
    Create the same run-specific directory name in each node-local
    temporary root.
    """

    if distributed:
        run_id = [uuid.uuid4().hex[:12] if rank == 0 else None]
        dist.broadcast_object_list(run_id, src=0)
        run_id = run_id[0]
    else:
        run_id = uuid.uuid4().hex[:12]

    path = os.path.join(temp_root, f'cryosiam_semantic_{run_id}')
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def shared_prediction_paths(tmp_dir, tomo_name, leader_rank):
    base = os.path.join(tmp_dir, f'.{safe_name(tomo_name)}.semantic_group{leader_rank}')
    return {'probs': base + '.probs.f32',
            'distances': base + '.distances.f32'}


def open_shared_arrays(paths, num_classes, input_size,
                       create=False):
    mode = 'w+' if create else 'r+'
    shape = num_classes, *input_size
    probs = np.memmap(paths['probs'], dtype=np.float32, mode=mode, shape=shape)
    distances = np.memmap(paths['distances'], dtype=np.float32, mode=mode, shape=shape)
    if create:
        probs.flush()
        distances.flush()
    return probs, distances


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


def cleanup_shared_files(paths):
    if not paths:
        return

    for path in paths.values():
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


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


def broadcast_bool(value, process_group,
                   leader_rank, group_rank, group_size):
    if group_size == 1:
        return bool(value)

    obj = [bool(value) if group_rank == 0 else None]
    dist.broadcast_object_list(obj, src=leader_rank, group=process_group)
    return bool(obj[0])


def patch_slices(c_batch, input_size, patch_size):
    slices = tuple(slice(c[0], c[1] - p // 4) if c[0] == 0 else slice(c[0] + p // 4, c[1]) \
        if c[1] >= s else slice(c[0] + p // 4, c[1] - p // 4)
                   for c, s, p in zip(c_batch, input_size, patch_size))

    slices2 = tuple(slice(0, 3 * p // 4) if c[0] == 0 else slice(p // 4, p - (c[1] - s)) \
        if c[1] >= s else slice(p // 4, 3 * p // 4)
                    for c, s, p in zip(c_batch, input_size, patch_size))

    return slices, slices2


def distributed_patch_prediction(img, backbone, prediction_model, patch_iter, patch_size,
                                 input_size, num_classes, batch_size, device, tomo_name, process_group=None,
                                 group_rank=0, group_size=1, leader_rank=0, shared_tmp_dir=None, inference_workers=0):
    """
    Predict one tomogram while sharding its patches across all GPUs
    assigned to the same physical node.
    """

    distributed_tomo = group_size > 1
    paths = None

    if distributed_tomo:
        os.makedirs(shared_tmp_dir, exist_ok=True)
        paths = shared_prediction_paths(shared_tmp_dir, tomo_name, leader_rank)
        if group_rank == 0:
            cleanup_shared_files(paths)
            probs_out, distances_out = open_shared_arrays(paths, num_classes, input_size, create=True)
        else:
            probs_out = None
            distances_out = None

        dist.barrier(group=process_group)
        if group_rank != 0:
            probs_out, distances_out = open_shared_arrays(paths, num_classes, input_size, create=False)
    else:
        probs_out = np.zeros((num_classes, *input_size), dtype=np.float32)
        distances_out = np.zeros((num_classes, *input_size), dtype=np.float32)

    patch_dataset = ShardedPatchDataset(img, patch_iter, shard_rank=group_rank, num_shards=group_size)

    loader = DataLoader(patch_dataset, batch_size=batch_size, num_workers=int(inference_workers),
                        pin_memory=torch.cuda.is_available(), persistent_workers=bool(inference_workers > 0))

    progress_total = local_patch_count(input_size, patch_size, group_rank, group_size, inference_workers)

    progress = tqdm(total=progress_total, desc=(f'  {tomo_name} | leader GPU shard'),
                    unit='patch', leave=False, disable=(group_rank != 0))

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
            out = (out.float().cpu().numpy())
            d_out = (d_out.float().cpu().numpy())
            for batch_i in range(patches.shape[0]):
                c_batch = coords[batch_i][1:]
                if any(c_batch[d][0] >= input_size[d] - patch_size[d] // 4 for d in range(len(input_size))):
                    continue
                slices, slices2 = patch_slices(c_batch, input_size, patch_size)
                probs_out[(slice(None),) + slices] = out[batch_i][(slice(None),) + slices2]
                distances_out[(slice(None),) + slices] = d_out[batch_i][(slice(None),) + slices2]

    if group_rank == 0:
        progress.close()

    if isinstance(probs_out, np.memmap):
        probs_out.flush()
        distances_out.flush()

    if distributed_tomo:
        dist.barrier(group=process_group)
        if group_rank != 0:
            close_memmap(probs_out)
            close_memmap(distances_out)
            return None, None, paths

    return probs_out, distances_out, paths


def labels_from_probabilities(probs_out, threshold, num_classes):
    if num_classes == 1:
        return (probs_out[0] > threshold).astype(np.uint8)
    thresholded = np.array(probs_out, dtype=np.float32, copy=True)
    thresholded[thresholded < threshold] = 0
    return np.argmax(thresholded, axis=0).astype(np.uint8)


def read_requested_topology(cfg):
    """Read physical nodes and GPUs per node from the YAML."""

    parameters = cfg.get('parameters', {})
    nodes_value = parameters.get('nodes', 1)
    gpu_devices_value = parameters.get('gpu_devices', 1)
    try:
        n_nodes = int(nodes_value)
    except (
            TypeError,
            ValueError):
        raise ValueError(
            'parameters.nodes must be '
            'an integer number of '
            'physical nodes, got '
            f'{nodes_value!r}.'
        )

    try:
        gpus_per_node = int(gpu_devices_value)
    except (TypeError, ValueError):
        raise ValueError(
            'parameters.gpu_devices '
            'must be an integer number '
            'of GPUs per node, got '
            f'{gpu_devices_value!r}.'
        )

    if n_nodes < 1:
        raise ValueError(f'parameters.nodes must be >= 1, got {n_nodes}.')

    if gpus_per_node < 1:
        raise ValueError(f'parameters.gpu_devices must be >= 1, got {gpus_per_node}.')

    return n_nodes, gpus_per_node


def maybe_launch_distributed_from_config(config_file_path):
    """
    Auto-launch local workers for one node. Multi-node jobs must
    already be launched by the scheduler/torchrun.
    """

    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        return False

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    n_nodes, gpus_per_node = read_requested_topology(cfg)

    if n_nodes > 1:
        raise RuntimeError('The config requests '
                           f'parameters.nodes={n_nodes} '
                           'with '
                           'parameters.gpu_devices='
                           f'{gpus_per_node} GPUs per '
                           'node. Multi-node jobs must '
                           'first be launched across '
                           'the allocated nodes with '
                           'your scheduler/torchrun so '
                           'that RANK, LOCAL_RANK and '
                           'WORLD_SIZE are set. The '
                           'expected WORLD_SIZE is '
                           f'{n_nodes * gpus_per_node}.')

    if gpus_per_node <= 1:
        return False

    if not torch.cuda.is_available():
        raise RuntimeError('parameters.gpu_devices='
                           f'{gpus_per_node}, but CUDA '
                           'is not available.')

    visible_gpus = torch.cuda.device_count()

    if gpus_per_node > visible_gpus:
        raise RuntimeError('parameters.gpu_devices='
                           f'{gpus_per_node}, but only '
                           f'{visible_gpus} CUDA '
                           'device(s) are visible. '
                           'Check the allocation and '
                           'CUDA_VISIBLE_DEVICES.')

    cmd = [sys.executable,
           '-m',
           'torch.distributed.run',
           '--standalone',
           '--max_restarts=0',
           f'--nproc_per_node='
           f'{gpus_per_node}',
           os.path.abspath(__file__),
           *sys.argv[1:]]

    print('Launching distributed '
          'semantic inference on 1 node '
          f'with {gpus_per_node} GPU '
          'worker(s)...')

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
    reuse_predictions = bool(cfg.get('reuse_predictions', False))
    mask_folder = cfg.get('mask_folder') or None
    num_classes = int(net_config['parameters']['network']['out_channels'])
    threshold = float(cfg['parameters']['network']['threshold'])
    patch_size = list(net_config['parameters']['data']['patch_size'])
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

    if (actual_world_size != expected_world_size):
        raise RuntimeError('Distributed topology mismatch: config requests {requested_nodes} node(s) '
                           f'x {requested_gpus_per_node} GPU(s) per node = {expected_world_size} '
                           'process(es), but WORLD_SIZE is {actual_world_size}. Check the scheduler/'
                           'torchrun launch.')

    if node_info['n_nodes'] != requested_nodes:
        raise RuntimeError('Distributed topology mismatch: config requests parameters.nodes='
                           f'{requested_nodes}, but the launched ranks span {node_info["n_nodes"]} '
                           'physical hostname(s): {node_info["all_hosts"]}.')

    if (node_info['group_size'] != requested_gpus_per_node):
        raise RuntimeError('Distributed topology mismatch on host '
                           f'{node_info["hostname"]}: config requests {requested_gpus_per_node} '
                           'GPU process(es) per node, but this node has {node_info["group_size"]} distributed rank(s).')

    shared_tmp_dir = make_run_temp_dir(temp_root, distributed, rank)
    if filename:
        files = [filename]
    else:
        files = cfg.get('test_files')
        if files is None:
            files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x))
                     and x.endswith(cfg['file_extension'])]

    files = sorted(files)

    if rank == 0:
        print(f'Using {world_size if distributed else 1} GPU process(es)')
        print(f'Test tomograms: {len(files)}')
        print(f'Requested topology: {requested_nodes} node(s)x {requested_gpus_per_node} '
              f'GPU(s) per node = {expected_world_size} total GPU process(es)')
        print(f'Physical nodes detected: {node_info["n_nodes"]}')
        print('Node groups:')

        for node_id, host in enumerate(node_info['all_hosts']):
            print(f'  node {node_id}: '                f'{host}')
        print(f'Temporary prediction storage root: {temp_root}')

    if not files:
        if distributed:
            dist.destroy_process_group()

        return

    reader = MrcReader(read_in_mem=True)

    writer = MrcWriter()

    transforms = Compose([
        LoadImaged(keys='image', reader=reader),
        EnsureChannelFirstd(keys='image'),
        NumpyToTensord(keys='image'),
        ScaleIntensityRanged(keys='image',
                             a_min=cfg['parameters']['data']['min'],
                             a_max=cfg['parameters']['data']['max'],
                             b_min=0,
                             b_max=1,
                             clip=True),
        NormalizeIntensityd(keys='image',
                            subtrahend=cfg['parameters']['data']['mean'],
                            divisor=cfg['parameters']['data']['std']),
        EnsureTyped(keys='image', data_type='tensor')])

    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='constant')
    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size),
                               start_pos=(0, 0),
                               overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size),
                               start_pos=(0, 0, 0),
                               overlap=(0, 0.5, 0.5, 0.5))

    jobs = []

    # One tomogram per physical node group.
    for file_idx in range(node_info['node_id'], len(files), node_info['n_nodes']):
        jobs.append({'file_idx': file_idx,
                     'group': node_info['group'],
                     'group_rank': node_info['group_rank'],
                     'group_size': node_info['group_size'],
                     'leader_rank': node_info['leader_rank']
                     })

    if node_info['group_rank'] == 0:
        assigned_names = [os.path.basename(files[job['file_idx']]) for job in jobs]

        print(f'Node {node_info["node_id"]} ({node_info["hostname"]}) uses ranks {node_info["group_ranks"]} '
              f'/ {node_info["group_size"]} GPU(s)')

        print(f'  assigned tomograms: {assigned_names if assigned_names else "none"}')
        print(f'  local temp dir: {shared_tmp_dir}')

    for job in jobs:
        file = files[job['file_idx']]

        current_file = os.path.join(test_folder, file)
        root = os.path.basename(file).split(cfg['file_extension'])[0]
        out_file = os.path.join(prediction_folder, os.path.basename(file))
        preds_h5 = os.path.join(prediction_folder, f'{root}_preds.h5')

        leader = job['group_rank'] == 0
        can_reuse = leader and reuse_predictions and reusable_prediction(preds_h5)
        can_reuse = broadcast_bool(can_reuse,
                                   job['group'],
                                   job['leader_rank'],
                                   job['group_rank'],
                                   job['group_size'])

        shared_paths = None
        probs_out = None
        distances_out = None
        shared_probs_handle = None
        shared_distances_handle = None

        if can_reuse:
            if leader:
                print(f'Loading saved predictions for file {current_file}')

                labels_out, probs_out, distances_out = load_reusable_prediction(preds_h5)
                original_size = list(labels_out.shape)
            else:
                if job['group_size'] > 1:
                    dist.barrier(group=job['group'])

                continue
        else:
            sample = transforms({'image': current_file,
                                 'file_name': current_file})

            original_size = list(sample['image'][0].shape)
            img = pad_transform(sample['image'])
            input_size = list(img[0].shape)

            if leader:
                print('Running prediction for file {current_file}')
                print(f'  size={original_size}, padded={input_size}, GPUs{job["group_size"]}')

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
                if job['group_size'] > 1:
                    dist.barrier(group=job['group'])
                continue

            shared_probs_handle = probs_out if isinstance(probs_out, np.memmap) else None

            shared_distances_handle = distances_out if isinstance(distances_out, np.memmap) else None
            crop = tuple(slice(0, n) for n in original_size)
            probs_out = probs_out[(slice(None),) + crop]
            distances_out = distances_out[(slice(None),) + crop]
            labels_out = labels_from_probabilities(probs_out, threshold, num_classes)

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

        print(f'Saving predictions for file {current_file}')

        if cfg.get('save_internal_files', False):
            with h5py.File(preds_h5, 'w') as f:
                f.create_dataset('labels', data=np.asarray(labels_out))

                f.create_dataset('probs', data=np.asarray(probs_out))

                f.create_dataset('distances', data=np.asarray(distances_out))
        else:
            if cfg.get('save_original_file_extension', False):
                writer.set_data_array(labels_out.astype(np.uint8), channel_dim=None)

                writer.write(out_file)
            else:
                with h5py.File(preds_h5, 'w') as f:
                    f.create_dataset('labels', data=labels_out.astype(np.uint8))

        if shared_probs_handle is not None:
            close_memmap(shared_probs_handle)
        elif isinstance(probs_out, np.memmap):
            close_memmap(probs_out)

        if shared_distances_handle is not None:
            close_memmap(shared_distances_handle)
        elif isinstance(distances_out, np.memmap):
            close_memmap(distances_out)

        if job['group_size'] > 1:
            dist.barrier(group=job['group'])
            cleanup_shared_files(shared_paths)

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
