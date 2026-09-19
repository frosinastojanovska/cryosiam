import os
import sys
import csv
import uuid
import h5py
import yaml
import socket
import starfile
import subprocess
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from skimage.morphology import convex_hull_image

from torch.utils.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    SpatialPad,
    EnsureType,
    LoadImaged,
    EnsureTyped,
    CenterSpatialCrop,
    EnsureChannelFirst,
    NormalizeIntensityd,
    ScaleIntensityRanged
)

from cryosiam.data import MrcReader
from cryosiam.utils import parser_helper
from cryosiam.apps.simsiam_prediction import load_backbone


class ParticlePatchDataset(Dataset):
    def __init__(self, image, instances_mask, regions, indices, transform, masking=0, cube_box=False):
        self.image = image
        self.instances_mask = instances_mask
        self.regions = regions
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.masking = int(masking)
        self.cube_box = bool(cube_box)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        label = int(self.regions['label'][i])
        slices = (slice(int(self.regions['bbox-0'][i]), int(self.regions['bbox-3'][i])),
                  slice(int(self.regions['bbox-1'][i]), int(self.regions['bbox-4'][i])),
                  slice(int(self.regions['bbox-2'][i]), int(self.regions['bbox-5'][i])))
        sub_mask = self.instances_mask[slices] == label
        patch = self.image[slices].copy()

        if self.masking == 0 and self.cube_box:
            max_extent = max(s.stop - s.start for s in slices)
            box_size = next(b for b in [16, 32, 64] if b >= max_extent)
            centered_slices = tuple(slice((s.start + s.stop) // 2 - box_size // 2,
                                          (s.start + s.stop) // 2 + box_size // 2) for s in slices)
            patch = self.image[centered_slices].copy()
        elif self.masking == 1:
            hull_mask = convex_hull_image(sub_mask)
            patch[hull_mask == 0] = 0
        elif self.masking == 2:
            patch[sub_mask == 0] = 0

        return self.transform(patch), i


def load_regions(regions_file_name, instances_mask):
    if not os.path.exists(regions_file_name):
        return regionprops_table(instances_mask, properties=['label', 'area', 'bbox', 'centroid'])

    regions = collections.defaultdict(list)
    with open(regions_file_name, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for key, value in row.items():
                regions[key].append(int(value) if 'label' in key or 'bbox' in key else float(value))
    return {key: np.asarray(value) for key, value in regions.items()}


def filter_regions(regions, instances_mask, min_particle_size=None, max_particle_size=None, masking=0):
    regions = {key: np.asarray(value) for key, value in regions.items()}
    regions['hull_area'] = np.zeros(len(regions['label']), dtype=np.int64)
    keep = []

    for i in range(len(regions['label'])):
        if min_particle_size and regions['area'][i] < min_particle_size:
            continue
        if max_particle_size and regions['area'][i] > max_particle_size:
            continue
        if masking == 1:
            slices = (slice(int(regions['bbox-0'][i]), int(regions['bbox-3'][i])),
                      slice(int(regions['bbox-1'][i]), int(regions['bbox-4'][i])),
                      slice(int(regions['bbox-2'][i]), int(regions['bbox-5'][i])))
            sub_mask = instances_mask[slices] == int(regions['label'][i])
            regions['hull_area'][i] = np.sum(convex_hull_image(sub_mask))
        keep.append(i)

    return {key: value[keep] for key, value in regions.items()}


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
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
    return distributed, rank, world_size, local_rank, device


def build_node_process_group(distributed, rank, world_size):
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

    groups = {host: dist.new_group(ranks=host_to_ranks[host]) for host in host_order}
    group_ranks = host_to_ranks[hostname]
    return {'hostname': hostname, 'node_id': host_order.index(hostname), 'n_nodes': len(host_order),
            'group': groups[hostname], 'group_rank': group_ranks.index(rank), 'group_size': len(group_ranks),
            'leader_rank': group_ranks[0], 'group_ranks': group_ranks, 'all_hosts': host_order}


def make_run_temp_dir(temp_root, distributed, rank):
    run_id = [uuid.uuid4().hex[:12] if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(run_id, src=0)
    elif run_id[0] is None:
        run_id[0] = uuid.uuid4().hex[:12]
    path = os.path.join(temp_root, f'cryosiam_simsiam_{run_id[0]}')
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def shared_paths(tmp_dir, tomo_name, leader_rank):
    base = os.path.join(tmp_dir, f'.{safe_name(tomo_name)}.group{leader_rank}')
    return {'image': base + '.image.f32', 'mask': base + '.mask.i32', 'embeddings': base + '.embeddings.f32'}


def write_memmap(path, array, dtype):
    out = np.memmap(path, dtype=dtype, mode='w+', shape=array.shape)
    out[...] = array
    out.flush()
    del out


def open_memmap(path, shape, dtype, mode='r'):
    return np.memmap(path, dtype=dtype, mode=mode, shape=tuple(shape))


def close_memmap(array):
    if array is None:
        return
    try:
        array.flush()
    except Exception:
        pass
    base, seen = array, set()
    while getattr(base, 'base', None) is not None and id(base) not in seen:
        seen.add(id(base))
        base = base.base
    if isinstance(base, np.memmap):
        try:
            base._mmap.close()
        except Exception:
            pass


def cleanup_shared_files(paths):
    for path in paths.values():
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def broadcast_object(value, group, leader_rank, group_rank, group_size):
    if group_size == 1:
        return value
    obj = [value if group_rank == 0 else None]
    dist.broadcast_object_list(obj, src=leader_rank, group=group)
    return obj[0]


def read_requested_topology(cfg):
    parameters = cfg.get('parameters', {})
    nodes = int(parameters.get('nodes', 1))
    gpus_per_node = int(parameters.get('gpu_devices', 1))
    if nodes < 1 or gpus_per_node < 1:
        raise ValueError('parameters.nodes and parameters.gpu_devices must both be >= 1.')
    return nodes, gpus_per_node


def maybe_launch_distributed_from_config(config_file_path):
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        return False

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)
    nodes, gpus_per_node = read_requested_topology(cfg)

    if nodes > 1:
        raise RuntimeError(f'The config requests {nodes} nodes x {gpus_per_node} GPUs per node. Multi-node jobs must '
                           f'be launched by the scheduler/torchrun with WORLD_SIZE={nodes * gpus_per_node}.')
    if gpus_per_node <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError(f'parameters.gpu_devices={gpus_per_node}, but CUDA is not available.')
    if gpus_per_node > torch.cuda.device_count():
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but only {torch.cuda.device_count()} GPUs are visible.')

    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--max_restarts=0',
           f'--nproc_per_node={gpus_per_node}', os.path.abspath(__file__), *sys.argv[1:]]
    print(f'Launching distributed SimSiam embedding inference on 1 node with {gpus_per_node} GPU worker(s)...')
    launch_env = os.environ.copy()
    launch_env.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run(cmd, check=True, env=launch_env)
    return True


def write_outputs(out_file, cfg, embeddings, regions, dim):
    n_particles = len(regions['label'])
    with h5py.File(out_file + '_embeds.h5', 'w') as f:
        dataset = f.create_dataset('embeddings', shape=(dim, n_particles), dtype=np.float32)
        for start in range(0, n_particles, 4096):
            stop = min(start + 4096, n_particles)
            dataset[:, start:stop] = embeddings[start:stop].T

    with h5py.File(out_file + '_instance_labels.h5', 'w') as f:
        f.create_dataset('labels', data=np.asarray(regions['label'], dtype=np.int64))

    with open(out_file + '_instance_regions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(list(regions.keys()))
        for i in range(n_particles):
            writer.writerow([regions[key][i] for key in regions.keys()])

    regions_df = pd.DataFrame(regions)
    regions_df.drop(columns=['hull_area'], inplace=True)
    regions_df.rename(columns={'centroid-0': 'rlnCoordinateZ', 'centroid-1': 'rlnCoordinateY',
                               'centroid-2': 'rlnCoordinateX', 'bbox-0': 'rlnBbox-0', 'bbox-1': 'rlnBbox-1',
                               'bbox-2': 'rlnBbox-2', 'bbox-3': 'rlnBbox-3', 'bbox-4': 'rlnBbox-4',
                               'bbox-5': 'rlnBbox-5', 'label': 'rlnLabel', 'area': 'rlnArea'}, inplace=True)
    starfile.write(regions_df, out_file + '_instance_regions.star', overwrite=True)


def main(config_file_path, filename=None):
    distributed, rank, world_size, local_rank, device = init_distributed()

    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    checkpoint_path = cfg['trained_model'] if cfg.get('trained_model') is not None \
        else os.path.join(cfg['log_dir'], 'model', 'last.ckpt')
    contrastive = bool(cfg.get('contrastive', False))
    net, dim = load_backbone(checkpoint_path, contrastive=contrastive, device=device)
    net.eval()

    test_folder = cfg['data_folder']
    instances_mask_folder = cfg['instances_mask_folder']
    prediction_folder = cfg['prediction_folder']
    batch_size = int(cfg['hyper_parameters']['batch_size'])
    inference_workers = int(cfg.get('inference_workers', 0))
    temp_root = cfg.get('temp_dir', prediction_folder)
    os.makedirs(prediction_folder, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    requested_nodes, requested_gpus = read_requested_topology(cfg)
    node_info = build_node_process_group(distributed, rank, world_size)
    expected_world_size = requested_nodes * requested_gpus
    if world_size != expected_world_size:
        raise RuntimeError(f'Expected WORLD_SIZE={expected_world_size}, got {world_size}.')
    if node_info['n_nodes'] != requested_nodes:
        raise RuntimeError(f'Expected {requested_nodes} physical node(s), found {node_info["n_nodes"]}.')
    if node_info['group_size'] != requested_gpus:
        raise RuntimeError(f'Expected {requested_gpus} GPU process(es) on {node_info["hostname"]}, '
                           f'found {node_info["group_size"]}.')

    shared_tmp_dir = make_run_temp_dir(temp_root, distributed, rank)
    files = cfg.get('test_files')
    if files is None:
        files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x))]
    if cfg.get('train_files') is not None:
        files += cfg['train_files']
    if filename:
        files = [filename]
    files = sorted(files)

    reader = MrcReader(read_in_mem=True)
    transforms = Compose([
        LoadImaged(keys=['image'], reader=reader),
        ScaleIntensityRanged(keys=['image'], a_min=cfg['parameters']['data']['min'],
                             a_max=cfg['parameters']['data']['max'], b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys='image', subtrahend=cfg['parameters']['data']['mean'],
                            divisor=cfg['parameters']['data']['std']),
        EnsureTyped(keys=['image'], data_type='numpy')
    ])
    patch_transforms = Compose([EnsureChannelFirst(channel_dim='no_channel'),
                                SpatialPad(cfg['parameters']['data']['patch_size']),
                                CenterSpatialCrop(roi_size=cfg['parameters']['data']['patch_size']),
                                EnsureType(data_type='tensor')])

    jobs = range(node_info['node_id'], len(files), node_info['n_nodes'])
    if rank == 0:
        print(f'Using {world_size} GPU process(es): {requested_nodes} node(s) x {requested_gpus} GPU(s) per node')
    if node_info['group_rank'] == 0:
        print(f'Node {node_info["node_id"]} ({node_info["hostname"]}) assigned: '
              f'{[os.path.basename(files[i]) for i in jobs]}')
        jobs = range(node_info['node_id'], len(files), node_info['n_nodes'])

    for file_idx in jobs:
        file = files[file_idx]
        current_file = os.path.join(test_folder, file)
        root = os.path.basename(file).split(cfg['file_extension'])[0]
        out_file = os.path.join(prediction_folder, root)
        paths = shared_paths(shared_tmp_dir, root, node_info['leader_rank'])
        leader = node_info['group_rank'] == 0

        skip = leader and os.path.exists(out_file + '_embeds.h5')
        skip = broadcast_object(skip, node_info['group'], node_info['leader_rank'],
                                node_info['group_rank'], node_info['group_size'])
        if skip:
            if leader:
                print('Skipping', out_file)
            continue

        if 'instances' in cfg:
            suffix = f'_min-{cfg["instances"]["min_center_distance"]}_max-{cfg["instances"]["max_center_distance"]}'
        else:
            suffix = ''
        instances_file = os.path.join(instances_mask_folder, root + f'_instance_preds{suffix}.h5')
        regions_file = os.path.join(instances_mask_folder, root + f'_instance_regions{suffix}.csv')

        exists = leader and os.path.exists(instances_file)
        exists = broadcast_object(exists, node_info['group'], node_info['leader_rank'],
                                  node_info['group_rank'], node_info['group_size'])
        if not exists:
            if leader:
                print('Missing instance prediction, skipping', instances_file)
            continue

        metadata = None
        if leader:
            print(f'Running prediction for file {current_file}')
            sample = transforms({'image': current_file})
            image = np.asarray(sample['image'], dtype=np.float32)
            with h5py.File(instances_file, 'r') as f:
                mask = f['instances'][()]
            mask = expand_labels(mask, distance=cfg['expand_labels']).astype(np.int32, copy=False)
            regions = load_regions(regions_file, mask)
            regions = filter_regions(regions, mask, cfg['min_particle_size'], cfg['max_particle_size'],
                                     cfg['masking_type'])
            cleanup_shared_files(paths)
            write_memmap(paths['image'], image, np.float32)
            write_memmap(paths['mask'], mask, np.int32)
            embeddings = np.memmap(paths['embeddings'], dtype=np.float32, mode='w+', shape=(len(regions['label']), dim))
            embeddings.flush()
            close_memmap(embeddings)
            metadata = {'image_shape': image.shape, 'mask_shape': mask.shape, 'regions': regions,
                        'n_particles': len(regions['label'])}

        metadata = broadcast_object(metadata, node_info['group'], node_info['leader_rank'],
                                    node_info['group_rank'], node_info['group_size'])
        if node_info['group_size'] > 1:
            dist.barrier(group=node_info['group'])

        image = open_memmap(paths['image'], metadata['image_shape'], np.float32)
        mask = open_memmap(paths['mask'], metadata['mask_shape'], np.int32)
        embeddings = open_memmap(paths['embeddings'], (metadata['n_particles'], dim), np.float32, mode='r+')
        regions = metadata['regions']
        indices = np.arange(node_info['group_rank'], metadata['n_particles'], node_info['group_size'], dtype=np.int64)

        dataset = ParticlePatchDataset(image, mask, regions, indices, patch_transforms,
                                       masking=cfg['masking_type'], cube_box=cfg.get('cube_box', False))
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=inference_workers,
                            pin_memory=torch.cuda.is_available(), persistent_workers=bool(inference_workers > 0))
        progress = tqdm(total=len(indices), desc=f'  {root} | leader GPU shard', unit='particle',
                        leave=False, disable=not leader)

        with torch.inference_mode():
            for patches, particle_indices in loader:
                patches = patches.to(device, non_blocking=True)
                if contrastive:
                    _, out = net.forward_one(patches)
                else:
                    out = net.encoder(patches)
                embeddings[particle_indices.numpy()] = out.float().cpu().numpy()
                if leader:
                    progress.update(len(particle_indices))

        if leader:
            progress.close()
        embeddings.flush()
        if node_info['group_size'] > 1:
            dist.barrier(group=node_info['group'])

        if leader:
            print(f'Saving predictions for file {current_file}')
            write_outputs(out_file, cfg, embeddings, regions, dim)

        close_memmap(embeddings)
        close_memmap(image)
        close_memmap(mask)
        if node_info['group_size'] > 1:
            dist.barrier(group=node_info['group'])
        if leader:
            cleanup_shared_files(paths)

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
        main(args.config_file, args.filename)
