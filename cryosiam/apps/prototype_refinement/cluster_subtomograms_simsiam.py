import os
import re
import sys
import json
import h5py
import uuid
import yaml
import torch
import shutil
import base64
import mrcfile
import argparse
import hashlib
import starfile
import subprocess
import numpy as np
import pandas as pd
import plotly.io as pio
from pathlib import Path
import torch.nn.functional as F
import torch.distributed as dist
import plotly.graph_objects as go
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from skimage.segmentation import expand_labels
from skimage.morphology import convex_hull_image

from cryosiam.apps.simsiam_prediction import load_backbone


def read_requested_topology(cfg):
    parameters = cfg.get('parameters', {})
    nodes = int(parameters.get('nodes', 1))
    configured_gpus = int(parameters.get('gpu_devices', 1))

    # Existing simsiam_prediction configs commonly use gpu_devices: 0
    # for the ordinary single-GPU/default case.
    gpus_per_node = 1 if configured_gpus == 0 else configured_gpus

    if nodes < 1 or gpus_per_node < 1:
        raise ValueError(
            'parameters.nodes must be >= 1 and '
            'parameters.gpu_devices must be >= 0.')
    return nodes, gpus_per_node


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f'LOCAL_RANK={local_rank}, but only '
                f'{torch.cuda.device_count()} CUDA device(s) are visible.')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    distributed = world_size > 1
    if distributed:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://')

    return distributed, rank, world_size, local_rank, device


def maybe_launch_distributed_from_config(config_file_path):
    # If torchrun / the scheduler already launched us, do not launch again.
    if ('LOCAL_RANK' in os.environ or
            int(os.environ.get('WORLD_SIZE', '1')) > 1):
        return False

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    nodes, gpus_per_node = read_requested_topology(cfg)

    if nodes > 1:
        raise RuntimeError(
            f'The config requests {nodes} nodes x {gpus_per_node} GPUs per node. '
            'Multi-node jobs must be launched by the scheduler/torchrun with '
            f'WORLD_SIZE={nodes * gpus_per_node}.')

    if gpus_per_node <= 1:
        return False

    if not torch.cuda.is_available():
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but CUDA is not available.')

    if gpus_per_node > torch.cuda.device_count():
        raise RuntimeError(
            f'parameters.gpu_devices={gpus_per_node}, but only '
            f'{torch.cuda.device_count()} GPU(s) are visible.')

    cmd = [
        sys.executable,
        '-m', 'torch.distributed.run',
        '--rdzv_backend=static',
        '--master_addr=127.0.0.1',
        '--master_port=29500',
        '--max_restarts=0',
        f'--nproc_per_node={gpus_per_node}',
        os.path.abspath(__file__),
        '--config_file',
        os.path.abspath(config_file_path),
    ]

    print(
        f'Launching distributed SimSiam clustering inference on 1 node with '
        f'{gpus_per_node} GPU worker(s)...')

    launch_env = os.environ.copy()
    launch_env.setdefault('OMP_NUM_THREADS', '1')
    launch_env.setdefault('OPENBLAS_NUM_THREADS', '1')
    launch_env.setdefault('MKL_NUM_THREADS', '1')
    subprocess.run(cmd, check=True, env=launch_env)
    return True


def make_run_temp_dir(output_folder, distributed, rank):
    run_id = [uuid.uuid4().hex[:12] if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(run_id, src=0)
    path = Path(output_folder) / f'.simsiam_cluster_tmp_{run_id[0]}'
    path.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    return path


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_size(value):
    if isinstance(value, (int, float)):
        size = (int(value),) * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        size = tuple(int(v) for v in value)
    else:
        raise ValueError('Size must be an integer or [z, y, x].')
    if any(v < 4 for v in size):
        raise ValueError('Size values must be >= 4.')
    return size


def orientation_file_from_config(cfg, cluster_cfg):
    explicit = cluster_cfg.get('orientations_file')
    if explicit:
        return Path(explicit)

    orient_cfg = cfg.get('initial_orientations') or {}
    if orient_cfg.get('output'):
        return Path(orient_cfg['output'])

    output_folder = Path(
        cfg.get('output_folder', cfg.get('prediction_folder', '.')))
    output_file = orient_cfg.get('output_file', 'initial_orientations.csv')
    return output_folder / output_file


def _read_star_table(path):
    data = starfile.read(path)
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, dict):
        if 'particles' in data and isinstance(data['particles'], pd.DataFrame):
            return data['particles'].copy()

        tables = [value for value in data.values()
                  if isinstance(value, pd.DataFrame)]
        if len(tables) == 1:
            return tables[0].copy()

        raise ValueError(
            f'Could not identify the particle table in STAR file {path}. '
            f'Available blocks: {list(data.keys())}')

    raise ValueError(
        f'Unsupported STAR structure returned by starfile for {path}: '
        f'{type(data).__name__}')


def _recover_original_centers_from_instance_masks(
        df, prediction_folder, mask_dataset='instance_mask'):
    """
    Recover the original particle centroids directly from the prediction H5
    instance labels.

    This is used for STAR input because the orientation STAR stores the refined
    coordinates used for averaging, while SimSiam clustering should use the
    original unaligned particle crop.
    """
    df = df.copy()

    original_x = np.full(len(df), np.nan, dtype=np.float64)
    original_y = np.full(len(df), np.nan, dtype=np.float64)
    original_z = np.full(len(df), np.nan, dtype=np.float64)

    for tomo_name, indices in df.groupby('tomo', sort=False).groups.items():
        h5_path = prediction_path(prediction_folder, tomo_name)

        with h5py.File(h5_path, 'r') as hf:
            if mask_dataset not in hf:
                raise KeyError(
                    f'{h5_path}: missing dataset "{mask_dataset}". '
                    'STAR input requires the prediction instance labels so the '
                    'original particle centers can be recovered.')

            labels = hf[mask_dataset][()]

        if labels.ndim != 3:
            raise ValueError(
                f'{h5_path}:{mask_dataset} must be 3D, got {labels.shape}.')

        # Work only on foreground voxels. This is much cheaper than making one
        # full-volume boolean mask per particle.
        z, y, x = np.nonzero(labels)
        if len(z) == 0:
            raise RuntimeError(
                f'{h5_path}:{mask_dataset} contains no labeled instances.')

        ids = labels[z, y, x].astype(np.int64, copy=False)
        max_id = int(ids.max())

        counts = np.bincount(ids, minlength=max_id + 1)
        sum_z = np.bincount(ids, weights=z, minlength=max_id + 1)
        sum_y = np.bincount(ids, weights=y, minlength=max_id + 1)
        sum_x = np.bincount(ids, weights=x, minlength=max_id + 1)

        for idx in indices:
            instance_id = int(df.at[idx, 'instance_id'])
            if (instance_id < 1 or instance_id >= len(counts) or
                    counts[instance_id] == 0):
                raise ValueError(
                    f'Instance {instance_id} from the STAR file was not found '
                    f'in {h5_path}:{mask_dataset}.')

            n = float(counts[instance_id])
            original_z[idx] = sum_z[instance_id] / n
            original_y[idx] = sum_y[instance_id] / n
            original_x[idx] = sum_x[instance_id] / n

    df['center_x'] = original_x
    df['center_y'] = original_y
    df['center_z'] = original_z
    return df


def read_orientation_table(
        path, prediction_folder, class_names=None,
        use_refined_centers_for_average=True,
        mask_dataset='instance_mask'):
    """
    Read either the initial-orientation CSV or STAR file.

    Internal convention after loading:
      center_x/y/z
          Original particle center. Used for unaligned SimSiam embedding
          extraction, so clustering is independent of orientation refinement.

      average_center_x/y/z
          Refined center when requested/available. Used only for aligned raw
          cluster averages.

      r00..r22
          Particle -> reference rotation matrix.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(path)

        if 'class_name' not in df.columns and 'class' in df.columns:
            df = df.rename(columns={'class': 'class_name'})

        required = {
            'tomo', 'instance_id', 'class_name',
            'center_x', 'center_y', 'center_z',
            'r00', 'r01', 'r02',
            'r10', 'r11', 'r12',
            'r20', 'r21', 'r22',
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f'Orientation CSV is missing required columns: '
                f'{sorted(missing)}')

        df = df.copy()

        # Current orientation CSVs store the refined center in center_* and the
        # pre-alignment center in original_center_*. Older CSVs may only have
        # center_* or may have explicit refined_center_* fields.
        refined_x = (
            df['refined_center_x'].astype(float)
            if 'refined_center_x' in df.columns
            else df['center_x'].astype(float))
        refined_y = (
            df['refined_center_y'].astype(float)
            if 'refined_center_y' in df.columns
            else df['center_y'].astype(float))
        refined_z = (
            df['refined_center_z'].astype(float)
            if 'refined_center_z' in df.columns
            else df['center_z'].astype(float))

        if all(c in df.columns for c in
               ('original_center_x', 'original_center_y',
                'original_center_z')):
            original_x = df['original_center_x'].astype(float)
            original_y = df['original_center_y'].astype(float)
            original_z = df['original_center_z'].astype(float)
        else:
            original_x = df['center_x'].astype(float)
            original_y = df['center_y'].astype(float)
            original_z = df['center_z'].astype(float)

        # Embeddings always use the original, unaligned center.
        df['center_x'] = original_x
        df['center_y'] = original_y
        df['center_z'] = original_z

        if use_refined_centers_for_average:
            df['average_center_x'] = refined_x
            df['average_center_y'] = refined_y
            df['average_center_z'] = refined_z
        else:
            df['average_center_x'] = original_x
            df['average_center_y'] = original_y
            df['average_center_z'] = original_z

    elif suffix == '.star':
        star_df = _read_star_table(path)

        tomo_col = (
            'rlnTomoName'
            if 'rlnTomoName' in star_df.columns
            else 'rlnMicrographName'
            if 'rlnMicrographName' in star_df.columns
            else None)

        required_star = {
            'rlnInstanceId',
            'rlnClassLabel',
            'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
            'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
        }
        missing = required_star.difference(star_df.columns)
        if tomo_col is None:
            missing.add('rlnTomoName/rlnMicrographName')
        if missing:
            raise ValueError(
                f'Orientation STAR is missing required columns: '
                f'{sorted(missing)}')

        df = pd.DataFrame({
            'tomo': star_df[tomo_col].astype(str),
            'instance_id': star_df['rlnInstanceId'].astype(np.int64),
            'class_name': star_df['rlnClassLabel'].astype(str),
        })

        refined_x = star_df['rlnCoordinateX'].astype(float).to_numpy()
        refined_y = star_df['rlnCoordinateY'].astype(float).to_numpy()
        refined_z = star_df['rlnCoordinateZ'].astype(float).to_numpy()

        # save_relion_star() stores RELION angles for the inverse
        # (reference -> particle) rotation. Reconstruct that rotation and
        # transpose it back to the script's particle -> reference convention.
        angles = star_df[
            ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
        ].to_numpy(dtype=np.float64)

        R_reference_to_particle = Rotation.from_euler(
            'ZYZ', angles, degrees=True).as_matrix()
        R_particle_to_reference = np.transpose(
            R_reference_to_particle, (0, 2, 1))

        for r in range(3):
            for c in range(3):
                df[f'r{r}{c}'] = R_particle_to_reference[:, r, c]

        # The STAR coordinates are the refined coordinates. Recover the
        # original instance centroids from H5 for the orientation-independent
        # SimSiam embedding crop.
        df = _recover_original_centers_from_instance_masks(
            df,
            prediction_folder=prediction_folder,
            mask_dataset=mask_dataset)

        if use_refined_centers_for_average:
            df['average_center_x'] = refined_x
            df['average_center_y'] = refined_y
            df['average_center_z'] = refined_z
        else:
            df['average_center_x'] = df['center_x'].astype(float)
            df['average_center_y'] = df['center_y'].astype(float)
            df['average_center_z'] = df['center_z'].astype(float)

    else:
        raise ValueError(
            f'Unsupported orientation file "{path}". Use .csv or .star.')

    if class_names is not None:
        df = df[
            df['class_name'].astype(str).isin(class_names)
        ].copy()

    if len(df) == 0:
        raise RuntimeError('No particles remain after class filtering.')

    df = df.reset_index(drop=True)
    df['_particle_index'] = np.arange(len(df), dtype=np.int64)
    return df


def load_params(config_file):
    cfg = read_yaml(config_file)
    cluster_cfg = cfg.get('cluster_subtomograms') or {}
    if not isinstance(cluster_cfg, dict):
        raise ValueError('cluster_subtomograms must be a mapping.')

    if 'data_folder' not in cfg:
        raise KeyError('Config is missing top-level data_folder.')
    if 'prediction_folder' not in cfg:
        raise KeyError('Config is missing top-level prediction_folder.')

    model_path = cluster_cfg.get('simsiam_model')
    if not model_path:
        raise ValueError('Set cluster_subtomograms.simsiam_model.')
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    orientation_path = orientation_file_from_config(cfg, cluster_cfg)
    if not orientation_path.is_file():
        raise FileNotFoundError(orientation_path)

    class_names = cluster_cfg.get('class_names')
    class_name = cluster_cfg.get('class_name')
    if class_names is not None and class_name is not None:
        raise ValueError('Use only one of class_name or class_names.')
    if class_names is None and class_name is not None:
        class_names = [str(class_name)]
    elif isinstance(class_names, str):
        class_names = [class_names]
    elif class_names is not None:
        class_names = [str(v) for v in class_names]

    mask_mode = str(cluster_cfg.get('mask_mode', 'convex_hull')).lower()
    if mask_mode not in ('convex_hull', 'strict'):
        raise ValueError('mask_mode must be "convex_hull" or "strict".')

    orient_cfg = cfg.get('initial_orientations') or {}
    average_crop_size = cluster_cfg.get(
        'average_crop_size', orient_cfg.get('registration_crop_size', 64))

    output_folder = Path(cluster_cfg.get(
        'output_folder',
        Path(cfg.get('output_folder', cfg['prediction_folder'])) /
        'cluster_subtomograms'))

    params = {
        'simsiam_model': model_path,
        'orientations_file': orientation_path,
        'data_folder': Path(cfg['data_folder']),
        'prediction_folder': Path(cfg['prediction_folder']),
        'file_extension': str(cfg.get('file_extension', '.mrc')),
        'output_folder': output_folder,
        'class_names': class_names,
        'contrastive': bool(cluster_cfg.get(
            'contrastive', cfg.get('contrastive', True))),
        'embedding_batch_size': int(cluster_cfg.get('embedding_batch_size', 32)),
        'reuse_saved_embeddings': bool(
            cluster_cfg.get('reuse_saved_embeddings', True)),
        'save_embeddings_cache': bool(
            cluster_cfg.get('save_embeddings_cache', True)),
        'force_recompute_embeddings': bool(
            cluster_cfg.get('force_recompute_embeddings', False)),
        'use_mask': bool(cluster_cfg.get('use_mask', False)),
        'mask_mode': mask_mode,
        'mask_dataset': str(cluster_cfg.get('mask_dataset', 'instance_mask')),
        'mask_expand': int(cluster_cfg.get('mask_expand', 0)),
        'skip_border_particles': bool(cluster_cfg.get('skip_border_particles', True)),
        'use_refined_centers_for_average': bool(
            cluster_cfg.get('use_refined_centers_for_average', True)),
        'use_pca': bool(cluster_cfg.get('use_pca', False)),
        'pca_components': int(cluster_cfg.get('pca_components', 16)),
        'clustering_method': str(
            cluster_cfg.get('clustering_method', 'kmeans')).lower(),
        'n_clusters': int(cluster_cfg.get('n_clusters', 4)),
        'spectral_n_neighbors': int(
            cluster_cfg.get('spectral_n_neighbors', 15)),
        'random_state': int(cluster_cfg.get('random_state', 10)),
        'umap_neighbors': int(cluster_cfg.get('umap_neighbors', 15)),
        'umap_min_dist': float(cluster_cfg.get('umap_min_dist', 0.1)),
        'umap_metric': str(cluster_cfg.get('umap_metric', 'cosine')),
        'save_raw_averages': bool(cluster_cfg.get('save_raw_averages', True)),
        'save_individual_cluster_files': bool(
            cluster_cfg.get(
                'save_individual_cluster_files',
                cluster_cfg.get('save_cluster_star_files', False))),
        'show_particle_crops_in_umap': bool(
            cluster_cfg.get('show_particle_crops_in_umap', False)),
        'particle_crop_preview_mode': str(
            cluster_cfg.get('particle_crop_preview_mode', 'aligned')).lower(),
        'particle_crop_preview_size': normalize_size(
            cluster_cfg.get('particle_crop_preview_size', average_crop_size)),
        'average_crop_size': normalize_size(average_crop_size),
        'normalize_raw_for_average': bool(
            cluster_cfg.get('normalize_raw_for_average', True)),
        'apply_soft_mask_to_average': bool(
            cluster_cfg.get('apply_soft_mask_to_average', False)),
        'average_mask_radius_fraction': float(
            cluster_cfg.get('average_mask_radius_fraction', 0.46)),
        'average_mask_edge_fraction': float(
            cluster_cfg.get('average_mask_edge_fraction', 0.06)),
        'max_particles': cluster_cfg.get('max_particles'),
    }

    if params['embedding_batch_size'] < 1:
        raise ValueError('embedding_batch_size must be >= 1.')
    if params['clustering_method'] not in ('kmeans', 'spectral'):
        raise ValueError(
            'clustering_method must be "kmeans" or "spectral".')
    if params['n_clusters'] < 2:
        raise ValueError('n_clusters must be >= 2.')
    if params['spectral_n_neighbors'] < 2:
        raise ValueError('spectral_n_neighbors must be >= 2.')
    if params['use_pca'] and params['pca_components'] < 2:
        raise ValueError('pca_components must be >= 2.')
    if params['mask_expand'] < 0:
        raise ValueError('mask_expand must be >= 0.')
    if params['particle_crop_preview_mode'] not in ('aligned', 'raw'):
        raise ValueError(
            'particle_crop_preview_mode must be "aligned" or "raw".')
    if params['max_particles'] is not None:
        params['max_particles'] = int(params['max_particles'])
        if params['max_particles'] < 1:
            raise ValueError('max_particles must be >= 1.')

    return cfg, params


def _checkpoint_data_cfg(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', {})

    for key in ('config', 'backbone_config', 'model_config'):
        candidate = hparams.get(key)
        if (isinstance(candidate, dict) and
                isinstance(candidate.get('parameters'), dict) and
                isinstance(candidate['parameters'].get('data'), dict)):
            return dict(candidate['parameters']['data'])
    return None


def resolve_simsiam_data_cfg(cfg, checkpoint_path):
    data_cfg = _checkpoint_data_cfg(checkpoint_path)
    if data_cfg is None:
        parameters = cfg.get('parameters') or {}
        if isinstance(parameters.get('data'), dict):
            data_cfg = dict(parameters['data'])

    if data_cfg is None:
        raise RuntimeError(
            'Could not obtain SimSiam parameters.data from the checkpoint and '
            'the current YAML has no parameters.data fallback.')

    required = ('patch_size', 'min', 'max', 'mean', 'std')
    missing = [key for key in required if key not in data_cfg]
    if missing:
        raise RuntimeError(
            f'SimSiam data configuration is missing: {missing}')

    data_cfg['patch_size'] = normalize_size(data_cfg['patch_size'])
    return data_cfg


def load_simsiam(checkpoint_path, contrastive, device):
    # Same loader used by cryosiam.apps.simsiam_prediction.
    net, dim = load_backbone(str(checkpoint_path), contrastive=contrastive, device=device)
    net.eval()
    return net, int(dim)


def preprocess_patch(patch, data_cfg):
    # Equivalent to the ordinary prediction transforms:
    # ScaleIntensityRanged(..., b_min=0, b_max=1, clip=True)
    # NormalizeIntensityd(subtrahend=mean, divisor=std)
    x = np.asarray(patch, dtype=np.float32)
    a_min = float(data_cfg['min'])
    a_max = float(data_cfg['max'])
    mean = float(data_cfg['mean'])
    std = float(data_cfg['std'])

    if a_max <= a_min:
        raise ValueError('Invalid SimSiam min/max intensity range.')
    if abs(std) < 1e-8:
        raise ValueError('Invalid SimSiam normalization std.')

    x = np.clip(x, a_min, a_max)
    x = (x - a_min) / (a_max - a_min)
    x = (x - mean) / std
    return x.astype(np.float32, copy=False)


def tomogram_path(data_folder, tomo_name, extension):
    folder = Path(data_folder)
    tomo = str(tomo_name)
    stem = Path(tomo).stem
    candidates = [
        folder / tomo,
        folder / f'{tomo}{extension}',
        folder / f'{stem}{extension}',
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'Could not find tomogram {tomo_name}. Tried {[p.name for p in candidates]}')


def prediction_path(prediction_folder, tomo_name):
    folder = Path(prediction_folder)
    tomo = str(tomo_name)
    stem = Path(tomo).stem
    candidates = [
        folder / f'{tomo}_preds.h5',
        folder / f'{tomo}.h5',
        folder / f'{stem}_preds.h5',
        folder / f'{stem}.h5',
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'Could not find prediction H5 for {tomo_name}. '
        f'Tried {[p.name for p in candidates]}')


def extract_centered_patch(volume, center_zyx, shape, fill_value=0.0):
    shape = np.asarray(shape, dtype=np.int64)
    center = np.asarray(center_zyx, dtype=np.float64)
    start = np.floor(
        center - (shape.astype(np.float64) - 1.0) / 2.0).astype(np.int64)
    stop = start + shape

    patch = np.full(tuple(shape), float(fill_value), dtype=np.float32)
    volume_shape = np.asarray(volume.shape, dtype=np.int64)
    src_start = np.maximum(start, 0)
    src_stop = np.minimum(stop, volume_shape)

    if np.all(src_stop > src_start):
        src_slc = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_stop))
        dst_start = src_start - start
        dst_stop = dst_start + (src_stop - src_start)
        dst_slc = tuple(slice(int(a), int(b)) for a, b in zip(dst_start, dst_stop))
        patch[dst_slc] = np.asarray(volume[src_slc], dtype=np.float32)

    fully_inside = bool(np.all(start >= 0) and np.all(stop <= volume_shape))
    return patch, fully_inside


def instance_mask_for_patch(label_patch, instance_id, mode, expand_distance):
    labels = np.asarray(label_patch)
    if expand_distance > 0:
        labels = expand_labels(labels, distance=int(expand_distance))

    strict_mask = labels == int(instance_id)
    if not np.any(strict_mask):
        return None

    if mode == 'strict':
        return strict_mask

    try:
        hull = convex_hull_image(strict_mask)
        if not np.any(hull):
            raise RuntimeError('empty convex hull')
        return hull.astype(bool)
    except Exception as exc:
        print(
            f'WARNING: convex hull failed for instance {instance_id}: {exc}. '
            'Using strict mask instead.')
        return strict_mask


def extract_simsiam_input(row, params, data_cfg, tomo_cache, h5_cache):
    tomo = str(row['tomo'])
    center = np.array(
        [row['center_z'], row['center_y'], row['center_x']], dtype=np.float64)
    patch_size = data_cfg['patch_size']

    if tomo not in tomo_cache:
        path = tomogram_path(params['data_folder'], tomo, params['file_extension'])
        tomo_cache[tomo] = mrcfile.mmap(path, permissive=True, mode='r')

    raw_patch, inside = extract_centered_patch(
        tomo_cache[tomo].data, center, patch_size, fill_value=0.0)
    if params['skip_border_particles'] and not inside:
        return None

    # The original SimSiam prediction scales/normalizes the tomogram first and
    # only then applies convex-hull/strict masking to the extracted particle.
    patch = preprocess_patch(raw_patch, data_cfg)

    if params['use_mask']:
        if tomo not in h5_cache:
            path = prediction_path(params['prediction_folder'], tomo)
            h5_cache[tomo] = h5py.File(path, 'r')

        hf = h5_cache[tomo]
        key = params['mask_dataset']
        if key not in hf:
            raise KeyError(f'{hf.filename}: missing dataset "{key}".')

        label_patch, mask_inside = extract_centered_patch(
            hf[key], center, patch_size, fill_value=0)
        if params['skip_border_particles'] and not mask_inside:
            return None

        particle_mask = instance_mask_for_patch(
            label_patch,
            int(row['instance_id']),
            params['mask_mode'],
            params['mask_expand'])
        if particle_mask is None:
            print(
                f'WARNING: instance {row["instance_id"]} was not found in the '
                f'local {params["mask_dataset"]} crop for {tomo}.')
            return None

        patch = patch.copy()
        patch[~particle_mask] = 0.0

    return patch


def close_caches(tomo_cache, h5_cache):
    for obj in tomo_cache.values():
        try:
            obj.close()
        except Exception:
            pass
    for obj in h5_cache.values():
        try:
            obj.close()
        except Exception:
            pass


@torch.no_grad()
def extract_embeddings(df, net, dim, params, data_cfg, device):
    batch_size = int(params['embedding_batch_size'])
    embedding_chunks = []
    kept_rows = []
    batch = []
    batch_rows = []
    tomo_cache = {}
    h5_cache = {}

    def flush():
        if not batch:
            return

        array = np.stack(batch, axis=0).astype(np.float32, copy=False)
        image = torch.from_numpy(array[:, None]).to(
            device=device, dtype=torch.float32, non_blocking=True)

        if params['contrastive']:
            _, out = net.forward_one(image)
        else:
            # This is exactly the representation used by the ordinary
            # simsiam_embeddings_predict implementation.
            out = net.encoder(image)

        out = out.detach().float()
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)

        # KMeans is then driven by embedding direction rather than arbitrary
        # embedding magnitude.
        out = F.normalize(out, dim=1)
        embedding_chunks.append(out.cpu().numpy().astype(np.float32, copy=False))
        kept_rows.extend(batch_rows)
        batch.clear()
        batch_rows.clear()

    try:
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if i == 1 or i % 100 == 0 or i == total:
                print(f'  SimSiam embedding extraction: {i}/{total}')

            try:
                patch = extract_simsiam_input(
                    row, params, data_cfg, tomo_cache, h5_cache)
            except Exception as exc:
                print(
                    f'WARNING: skipping {row["tomo"]} instance '
                    f'{row["instance_id"]}: {exc}')
                continue

            if patch is None:
                continue

            batch.append(patch)
            batch_rows.append(row)
            if len(batch) >= batch_size:
                flush()

        flush()
    finally:
        close_caches(tomo_cache, h5_cache)

    if not embedding_chunks:
        raise RuntimeError('No SimSiam embeddings were extracted.')

    X = np.concatenate(embedding_chunks, axis=0)
    metadata = pd.DataFrame(kept_rows).reset_index(drop=True)

    if len(metadata) != len(X):
        raise RuntimeError('Embedding count does not match particle metadata.')

    if X.shape[1] != dim:
        print(
            f'  NOTE: load_backbone reported dim={dim}, encoder returned '
            f'{X.shape[1]}; using the returned dimensionality.')

    print(f'  embedding matrix: {X.shape[0]} x {X.shape[1]}')
    return X, metadata


def prepare_cluster_space(embeddings, params):
    if not params['use_pca']:
        print('  PCA before KMeans: disabled')
        return embeddings, None

    n_components = min(
        int(params['pca_components']),
        embeddings.shape[0] - 1,
        embeddings.shape[1])
    if n_components < 2:
        raise RuntimeError('Not enough samples/features for PCA.')

    pca = PCA(
        n_components=n_components,
        svd_solver='randomized',
        random_state=int(params['random_state']))
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    print(
        f'  PCA: {embeddings.shape[1]} -> {n_components}; '
        f'explained variance={pca.explained_variance_ratio_.sum():.4f}')
    return reduced, pca


def cluster_particles(features, params):
    method = params['clustering_method']
    n_clusters = int(params['n_clusters'])

    if n_clusters > len(features):
        raise ValueError(
            f'n_clusters={n_clusters}, but only '
            f'{len(features)} particles are available.')

    if method == 'kmeans':
        model = KMeans(
            n_clusters=n_clusters,
            random_state=int(params['random_state']),
            n_init=10)
        labels = model.fit_predict(features).astype(np.int32)

        print(f'  KMeans clusters: {n_clusters}')

    elif method == 'spectral':
        if len(features) < 3:
            raise ValueError(
                'Spectral clustering requires at least 3 particles.')

        n_neighbors = min(
            int(params['spectral_n_neighbors']),
            len(features) - 1)

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            assign_labels='kmeans',
            n_init=10,
            random_state=int(params['random_state']))

        labels = model.fit_predict(features).astype(np.int32)

        print(
            f'  Spectral clusters: {n_clusters} '
            f'(nearest-neighbor graph, n_neighbors={n_neighbors})')

    else:
        raise ValueError(
            f'Unsupported clustering method: {method}')

    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(
            f'    cluster {int(label)}: '
            f'{int(count)} particle(s)')

    return labels, model


def compute_umap(features, params):
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            'UMAP requires umap-learn. Install it with: pip install umap-learn') from exc

    n_neighbors = min(
        int(params['umap_neighbors']), max(2, len(features) - 1))
    reducer = umap.UMAP(
        n_components=2,
        metric=params['umap_metric'],
        n_neighbors=n_neighbors,
        min_dist=float(params['umap_min_dist']),
        random_state=int(params['random_state']))
    return reducer.fit_transform(features).astype(np.float32)


def rotation_from_row(row):
    return np.array([
        [row['r00'], row['r01'], row['r02']],
        [row['r10'], row['r11'], row['r12']],
        [row['r20'], row['r21'], row['r22']],
    ], dtype=np.float64)


def output_center(shape):
    return (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0


def xyz_rotation_to_zyx(R_xyz):
    P = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)
    return P @ np.asarray(R_xyz, dtype=np.float64) @ P


def aligned_subtomogram(volume, center_zyx, crop_size, R_particle_to_reference,
                        fill_value=0.0, order=1):
    R_zyx = xyz_rotation_to_zyx(R_particle_to_reference)
    matrix = R_zyx.T
    out_center = output_center(crop_size)
    center_zyx = np.asarray(center_zyx, dtype=np.float64)
    offset = center_zyx - matrix @ out_center

    return affine_transform(
        volume,
        matrix=matrix,
        offset=offset,
        output_shape=tuple(int(v) for v in crop_size),
        order=int(order),
        mode='constant',
        cval=float(fill_value),
        prefilter=False,
    ).astype(np.float32, copy=False)


def spherical_mask(shape, radius_fraction=0.46, edge_fraction=0.06):
    shape = np.asarray(shape, dtype=np.int64)
    center = output_center(shape)
    z, y, x = np.indices(tuple(shape), dtype=np.float32)
    dz = (z - center[0]) / max(1.0, 0.5 * float(shape[0] - 1))
    dy = (y - center[1]) / max(1.0, 0.5 * float(shape[1] - 1))
    dx = (x - center[2]) / max(1.0, 0.5 * float(shape[2] - 1))
    r = np.sqrt(dx * dx + dy * dy + dz * dz)

    radius = float(radius_fraction)
    edge = max(float(edge_fraction), 0.0)
    if edge <= 0:
        return (r <= radius).astype(np.float32)

    mask = np.ones_like(r, dtype=np.float32)
    mask[r >= radius + edge] = 0.0
    transition = (r > radius) & (r < radius + edge)
    mask[transition] = 0.5 * (
            1.0 + np.cos(np.pi * (r[transition] - radius) / edge))
    return mask


def normalize_particle(volume, mask):
    core = mask > 0.5
    if not np.any(core):
        return volume
    values = volume[core]
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return ((volume - mean) / std).astype(np.float32, copy=False)


def write_mrc(path, volume, voxel_size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(volume, dtype=np.float32))
        if voxel_size is not None:
            try:
                mrc.voxel_size = voxel_size
            except Exception:
                pass


def save_average_preview(path, volume, title):
    import matplotlib.pyplot as plt

    arr = np.asarray(volume, dtype=np.float32)
    zc, yc, xc = [s // 2 for s in arr.shape]
    images = [
        ('XY', arr[zc]),
        ('XZ', arr[:, yc, :]),
        ('YZ', arr[:, :, xc]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    for ax, (name, image) in zip(axes, images):
        finite = np.isfinite(image)
        if np.any(finite):
            lo, hi = np.percentile(image[finite], [1.0, 99.0])
        else:
            lo, hi = 0.0, 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        ax.imshow(image, cmap='gray', origin='lower', vmin=lo, vmax=hi)
        ax.set_title(name, fontsize=9)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def create_cluster_averages(df, params, class_dir):
    crop_size = params['average_crop_size']
    norm_mask = spherical_mask(
        crop_size,
        params['average_mask_radius_fraction'],
        params['average_mask_edge_fraction'])

    average_dir = Path(class_dir) / 'cluster_averages'
    average_dir.mkdir(parents=True, exist_ok=True)

    sums = {}
    counts = {}
    voxel_size = None

    by_tomo = {}
    for _, row in df.iterrows():
        by_tomo.setdefault(str(row['tomo']), []).append(row)

    for tomo_name, rows in by_tomo.items():
        path = tomogram_path(params['data_folder'], tomo_name, params['file_extension'])
        print(f'    averaging {tomo_name}: {len(rows)} particle(s)')
        with mrcfile.mmap(path, permissive=True, mode='r') as mrc:
            volume = mrc.data
            if voxel_size is None:
                try:
                    voxel_size = mrc.voxel_size.copy()
                except Exception:
                    voxel_size = None

            fill_value = 0.0 if params['skip_border_particles'] else float(np.median(volume))

            for row in rows:
                label = int(row['cluster'])
                center = np.array([
                    row['average_center_z'],
                    row['average_center_y'],
                    row['average_center_x']],
                    dtype=np.float64)
                R = rotation_from_row(row)

                particle = aligned_subtomogram(
                    volume, center, crop_size, R,
                    fill_value=fill_value, order=1)

                if params['normalize_raw_for_average']:
                    particle = normalize_particle(particle, norm_mask)
                if params['apply_soft_mask_to_average']:
                    particle = particle * norm_mask

                if label not in sums:
                    sums[label] = np.zeros(crop_size, dtype=np.float64)
                    counts[label] = 0
                sums[label] += particle.astype(np.float64, copy=False)
                counts[label] += 1

    outputs = {}
    for label in sorted(sums):
        if counts[label] == 0:
            continue
        average = (sums[label] / float(counts[label])).astype(np.float32)
        name = f'cluster_{label:02d}'
        mrc_path = average_dir / f'{name}_average.mrc'
        png_path = average_dir / f'{name}_average.png'
        write_mrc(mrc_path, average, voxel_size=voxel_size)
        save_average_preview(
            png_path, average, f'{name} average (n={counts[label]})')
        outputs[label] = {
            'mrc': mrc_path,
            'png': png_path,
            'count': int(counts[label]),
        }
    return outputs


def cluster_rotation_to_relion_angles(row):
    """
    Convert the internal particle->reference matrix back to the RELION
    reference->particle intrinsic ZYZ convention used by save_relion_star().
    """
    R_particle_to_reference = rotation_from_row(row)
    R_reference_to_particle = R_particle_to_reference.T

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Gimbal lock detected.*',
            category=UserWarning)
        rot, tilt, psi = Rotation.from_matrix(
            R_reference_to_particle).as_euler(
            'ZYZ', degrees=True)

    return float(rot), float(tilt), float(psi)


def cluster_metadata_to_star(df):
    rows = []

    for _, row in df.iterrows():
        rot, tilt, psi = cluster_rotation_to_relion_angles(row)

        # The standard STAR coordinates follow the orientation output:
        # use the refined center for downstream averaging/alignment when it
        # exists, while keeping the original prediction center explicitly.
        center_x = float(
            row['average_center_x']
            if 'average_center_x' in row.index
            else row['center_x'])
        center_y = float(
            row['average_center_y']
            if 'average_center_y' in row.index
            else row['center_y'])
        center_z = float(
            row['average_center_z']
            if 'average_center_z' in row.index
            else row['center_z'])

        star_row = {
            'rlnTomoName': str(row['tomo']),
            'rlnMicrographName': str(row['tomo']),
            'rlnCoordinateX': center_x,
            'rlnCoordinateY': center_y,
            'rlnCoordinateZ': center_z,
            'rlnAngleRot': rot,
            'rlnAngleTilt': tilt,
            'rlnAnglePsi': psi,
            'rlnInstanceId': int(row['instance_id']),
            'rlnClassLabel': str(row['class_name']),
            'rlnClusterLabel': int(row['cluster']),
            'rlnOriginalCoordinateX': float(row['center_x']),
            'rlnOriginalCoordinateY': float(row['center_y']),
            'rlnOriginalCoordinateZ': float(row['center_z']),
        }

        # Preserve useful metadata when present.
        optional_columns = {
            'n_voxels': 'rlnNVoxels',
            'mean_sim': 'rlnMeanSim',
            'match_score': 'rlnMatchScore',
            'sim_refine_score': 'rlnSimRefineScore',
            'umap_1': 'rlnUmap1',
            'umap_2': 'rlnUmap2',
            'reference_id': 'rlnReferenceId',
            'master_reference_id': 'rlnMasterReferenceId',
        }

        for source, target in optional_columns.items():
            if source not in row.index:
                continue
            value = row[source]
            if pd.isna(value):
                continue

            if source in ('reference_id', 'master_reference_id'):
                star_row[target] = str(value)
            elif source == 'n_voxels':
                star_row[target] = int(value)
            else:
                star_row[target] = float(value)

        rows.append(star_row)

    return pd.DataFrame(rows)


def save_cluster_files(metadata, class_dir, save_individual=False):
    """
    Always save the full clustering table as both CSV and STAR.

    If save_individual=True, also save one CSV/STAR pair per KMeans cluster.
    """
    class_dir = Path(class_dir)

    joined_csv = class_dir / 'subtomogram_clusters.csv'
    joined_star = class_dir / 'subtomogram_clusters.star'

    metadata.to_csv(joined_csv, index=False)
    starfile.write(
        cluster_metadata_to_star(metadata),
        joined_star,
        overwrite=True)

    individual_files = []
    if save_individual:
        cluster_file_dir = class_dir / 'cluster_files'
        cluster_file_dir.mkdir(parents=True, exist_ok=True)

        for cluster_label in sorted(
                int(v) for v in metadata['cluster'].unique()):
            subset = metadata[
                metadata['cluster'] == cluster_label
            ].copy()

            csv_path = (
                cluster_file_dir /
                f'cluster_{cluster_label:02d}.csv')
            star_path = (
                cluster_file_dir /
                f'cluster_{cluster_label:02d}.star')

            subset.to_csv(csv_path, index=False)
            starfile.write(
                cluster_metadata_to_star(subset),
                star_path,
                overwrite=True)

            individual_files.append({
                'cluster': cluster_label,
                'csv': csv_path,
                'star': star_path,
            })

    return joined_csv, joined_star, individual_files


def save_particle_preview(path, volume, title):
    import matplotlib.pyplot as plt

    arr = np.asarray(volume, dtype=np.float32)
    zc, yc, xc = [size // 2 for size in arr.shape]
    images = [
        ('XY', arr[zc]),
        ('XZ', arr[:, yc, :]),
        ('YZ', arr[:, :, xc]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.7, 2.0))
    for ax, (name, image) in zip(axes, images):
        finite = np.isfinite(image)
        if np.any(finite):
            lo, hi = np.percentile(image[finite], [1.0, 99.0])
        else:
            lo, hi = 0.0, 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        ax.imshow(image, cmap='gray', origin='lower', vmin=lo, vmax=hi)
        ax.set_title(name, fontsize=7)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def create_particle_crop_previews(df, params, data_cfg, class_dir):
    mode = params['particle_crop_preview_mode']
    crop_size = params['particle_crop_preview_size']
    preview_dir = Path(class_dir) / 'particle_previews'
    preview_dir.mkdir(parents=True, exist_ok=True)

    norm_mask = spherical_mask(
        crop_size,
        params['average_mask_radius_fraction'],
        params['average_mask_edge_fraction'])

    tomo_cache = {}
    h5_cache = {}
    relpaths = []

    try:
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if i == 1 or i % 200 == 0 or i == total:
                print(f'  Particle crop previews: {i}/{total}')

            tomo = str(row['tomo'])
            if tomo not in tomo_cache:
                path = tomogram_path(
                    params['data_folder'], tomo, params['file_extension'])
                tomo_cache[tomo] = mrcfile.mmap(
                    path, permissive=True, mode='r')

            volume = tomo_cache[tomo].data
            preview_name = (
                f'{i - 1:06d}_instance_{int(row["instance_id"])}_'
                f'cluster_{int(row["cluster"]):02d}.png')
            preview_path = preview_dir / preview_name

            if mode == 'raw':
                center = np.array([
                    row['center_z'], row['center_y'], row['center_x']],
                    dtype=np.float64)
                particle, _ = extract_centered_patch(
                    volume, center, crop_size, fill_value=0.0)
                particle = preprocess_patch(particle, data_cfg)

                if params['use_mask']:
                    if tomo not in h5_cache:
                        path = prediction_path(
                            params['prediction_folder'], tomo)
                        h5_cache[tomo] = h5py.File(path, 'r')

                    hf = h5_cache[tomo]
                    key = params['mask_dataset']
                    if key not in hf:
                        raise KeyError(
                            f'{hf.filename}: missing dataset "{key}".')

                    label_patch, _ = extract_centered_patch(
                        hf[key], center, crop_size, fill_value=0)
                    particle_mask = instance_mask_for_patch(
                        label_patch,
                        int(row['instance_id']),
                        params['mask_mode'],
                        params['mask_expand'])
                    if particle_mask is not None:
                        particle = particle.copy()
                        particle[~particle_mask] = 0.0

            else:
                center = np.array([
                    row['average_center_z'],
                    row['average_center_y'],
                    row['average_center_x']],
                    dtype=np.float64)
                R = rotation_from_row(row)
                particle = aligned_subtomogram(
                    volume, center, crop_size, R,
                    fill_value=0.0, order=1)

                if params['normalize_raw_for_average']:
                    particle = normalize_particle(particle, norm_mask)
                if params['apply_soft_mask_to_average']:
                    particle = particle * norm_mask

            save_particle_preview(
                preview_path,
                particle,
                f'{tomo} | instance {int(row["instance_id"])} | '
                f'cluster {int(row["cluster"])}')

            relpaths.append(str(preview_path.relative_to(class_dir)))

    finally:
        close_caches(tomo_cache, h5_cache)

    result = df.copy()
    result['particle_preview_relpath'] = relpaths
    result['particle_preview_mode'] = mode
    return result


def png_data_uri(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return 'data:image/png;base64,' + encoded


def make_interactive_umap(df, averages, output_path, title):
    df = df.copy().reset_index(drop=True)
    df['_html_row_index'] = np.arange(len(df), dtype=np.int64)

    colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
    ]
    hover_cols = [
        c for c in (
            'tomo', 'instance_id', 'particle_preview_mode',
            'mean_sim', 'match_score',
            'sim_refine_score', 'reference_id')
        if c in df.columns
    ]

    fig = go.Figure()
    labels = sorted(int(v) for v in df['cluster'].unique())

    for i, label in enumerate(labels):
        subset = df[df['cluster'] == label]
        custom = []
        for _, row in subset.iterrows():
            values = [str(label)]
            preview = (
                str(row['particle_preview_relpath'])
                if 'particle_preview_relpath' in row.index and
                pd.notna(row['particle_preview_relpath'])
                else '')
            values.append(preview)
            values.append(int(row['_html_row_index']))

            for col in hover_cols:
                value = row[col]
                if isinstance(value, float) and not np.isfinite(value):
                    value = ''
                values.append(str(value))
            custom.append(values)

        hover = [
            f'<b>cluster {label}</b>',
            'UMAP 1: %{x:.3f}',
            'UMAP 2: %{y:.3f}',
        ]
        for j, col in enumerate(hover_cols, start=3):
            hover.append(f'{col}: %{{customdata[{j}]}}')

        fig.add_trace(go.Scattergl(x=subset['umap_1'],
                                   y=subset['umap_2'],
                                   mode='markers',
                                   name=f'cluster {label}',
                                   marker=dict(
                                       size=6,
                                       opacity=0.72,
                                       color=colors[i % len(colors)]),
                                   customdata=custom,
                                   hovertemplate='<br>'.join(hover) + '<extra></extra>'))

    fig.update_layout(
        title=title,
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        template='plotly_white',
        legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'),
        margin=dict(l=55, r=20, t=60, b=50))

    preview_data = {}
    for label in labels:
        subset = df[df['cluster'] == label]
        info = {
            'name': f'cluster {label}',
            'count': int(len(subset)),
            'image': None,
            'mrc': None,
        }
        if label in averages:
            info['image'] = png_data_uri(averages[label]['png'])
            info['mrc'] = str(averages[label]['mrc'])
            info['average_count'] = int(averages[label]['count'])
        preview_data[str(label)] = info

    export_csv_df = df.drop(
        columns=[
            '_html_row_index',
            'particle_preview_relpath',
            'particle_preview_mode',
        ],
        errors='ignore')
    export_star_df = cluster_metadata_to_star(df)

    csv_columns = list(export_csv_df.columns)
    star_columns = list(export_star_df.columns)
    csv_rows_json = export_csv_df.to_json(orient='records')
    star_rows_json = export_star_df.to_json(orient='records')

    plot_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={
            'displaylogo': False,
            'responsive': True,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
        })
    match = re.search(r'id="([^"]+)" class="plotly-graph-div"', plot_html)
    if not match:
        raise RuntimeError('Could not determine Plotly graph div id.')
    plot_id = match.group(1)

    default_label = max(
        labels, key=lambda label: int((df['cluster'] == label).sum()))

    html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
body { margin: 0; font-family: Arial, sans-serif; background: white; color: #222; }
#layout { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px; padding: 14px; align-items: start; }
#plot-panel { min-width: 0; }
#preview-panel { border: 1px solid #ddd; border-radius: 8px; padding: 12px; position: sticky; top: 12px; background: white; }
#preview-title { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
#preview-meta { color: #666; font-size: 13px; margin-bottom: 10px; }
#preview-image { width: 100%; height: auto; display: none; border-radius: 4px; border: 1px solid #eee; }
#preview-mrc { font-size: 11px; color: #777; margin-top: 8px; overflow-wrap: anywhere; }
#particle-preview-title { font-size: 15px; font-weight: 600; margin-top: 16px; margin-bottom: 4px; }
#particle-preview-meta { color: #666; font-size: 13px; margin-bottom: 8px; }
#particle-preview-image { width: 100%; height: auto; display: none; border-radius: 4px; border: 1px solid #eee; }
#selection-title { font-size: 15px; font-weight: 600; margin-top: 18px; margin-bottom: 4px; }
#selection-meta { color: #666; font-size: 13px; margin-bottom: 8px; }
#selection-filename { width: 100%; box-sizing: border-box; border: 1px solid #bbb; border-radius: 5px; padding: 7px 9px; margin-bottom: 8px; font-size: 12px; }
#selection-actions { display: flex; flex-wrap: wrap; gap: 6px; }
#selection-actions button { border: 1px solid #bbb; background: #fff; border-radius: 5px; padding: 7px 9px; cursor: pointer; font-size: 12px; }
#selection-actions button:hover:not(:disabled) { background: #f5f5f5; }
#selection-actions button:disabled { opacity: 0.45; cursor: default; }
#selection-help { font-size: 11px; color: #777; margin-top: 7px; line-height: 1.35; }
#preview-help { font-size: 12px; color: #777; margin-top: 10px; line-height: 1.4; }
@media (max-width: 900px) { #layout { grid-template-columns: 1fr; } #preview-panel { position: static; } }
</style>
</head>
<body>
<div id="layout">
<div id="plot-panel">__PLOT__</div>
<aside id="preview-panel">
<div id="preview-title">Cluster average</div>
<div id="preview-meta">Hover over a particle to preview its cluster.</div>
<img id="preview-image" alt="Cluster average preview">
<div id="preview-mrc"></div>
<div id="particle-preview-title">Particle crop</div>
<div id="particle-preview-meta">Hover over a point to show its crop.</div>
<img id="particle-preview-image" alt="Particle crop preview">
<div id="selection-title">Selected particles</div>
<div id="selection-meta">0 points selected</div>
<input id="selection-filename" type="text" value="selected_particles" aria-label="Output file name">
<div id="selection-actions">
<button id="download-selected-csv" type="button" disabled>Save CSV</button>
<button id="download-selected-star" type="button" disabled>Save STAR</button>
<button id="clear-selection" type="button" disabled>Clear</button>
</div>
<div id="selection-help">Use the box-select or lasso-select tool in the Plotly toolbar, then save the selected particles.</div>
<div id="preview-help">Hover updates both previews. Click a point to pin it. Plotly legend items can be toggled normally.</div>
</aside>
</div>
<script>
(function() {
const previews = __PREVIEWS__;
const graph = document.getElementById(__PLOT_ID__);
const titleEl = document.getElementById('preview-title');
const metaEl = document.getElementById('preview-meta');
const imageEl = document.getElementById('preview-image');
const mrcEl = document.getElementById('preview-mrc');
const particleMetaEl = document.getElementById('particle-preview-meta');
const particleImageEl = document.getElementById('particle-preview-image');
const selectionMetaEl = document.getElementById('selection-meta');
const filenameInput = document.getElementById('selection-filename');
const downloadCsvButton = document.getElementById('download-selected-csv');
const downloadStarButton = document.getElementById('download-selected-star');
const clearSelectionButton = document.getElementById('clear-selection');

const csvColumns = __CSV_COLUMNS__;
const starColumns = __STAR_COLUMNS__;
const csvRows = __CSV_ROWS__;
const starRows = __STAR_ROWS__;

const selectedRows = new Set();
let pinned = null;
function showCluster(key) {
  if (key === null || key === undefined || !previews[key]) return;
  const item = previews[key];
  titleEl.textContent = item.name;
  const avgCount = item.average_count !== undefined ? ' · average n = ' + item.average_count : '';
  metaEl.textContent = 'UMAP points n = ' + item.count + avgCount;
  if (item.image) { imageEl.src = item.image; imageEl.style.display = 'block'; }
  else { imageEl.removeAttribute('src'); imageEl.style.display = 'none'; }
  mrcEl.textContent = item.mrc ? item.mrc : 'No average saved for this cluster.';
}
function showParticle(cd) {
  if (!cd || cd.length < 2 || !cd[1]) {
    particleMetaEl.textContent = 'No particle crop saved for this point.';
    particleImageEl.removeAttribute('src');
    particleImageEl.style.display = 'none';
    return;
  }
  const tomo = cd.length > 3 ? cd[3] : '';
  const instanceId = cd.length > 4 ? cd[4] : '';
  const mode = cd.length > 5 ? cd[5] : '';
  particleMetaEl.textContent =
    'tomo=' + tomo + ' · instance_id=' + instanceId +
    (mode ? ' · mode=' + mode : '');
  particleImageEl.src = cd[1];
  particleImageEl.style.display = 'block';
}
function updateSelectionState() {
  const n = selectedRows.size;
  selectionMetaEl.textContent =
    n + (n === 1 ? ' point selected' : ' points selected');
  downloadCsvButton.disabled = n === 0;
  downloadStarButton.disabled = n === 0;
  clearSelectionButton.disabled = n === 0;
}

function csvCell(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  if (/[",\\n\\r]/.test(text)) {
    return '"' + text.replace(/"/g, '""') + '"';
  }
  return text;
}

function starCell(value) {
  if (value === null || value === undefined) return '?';
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '?';
  }

  const text = String(value);
  if (text.length === 0) return "''";
  if (!/[\\s'"]/.test(text)) return text;
  if (!text.includes("'")) return "'" + text + "'";
  if (!text.includes('"')) return '"' + text + '"';
  return "'" + text.replace(/'/g, '') + "'";
}

function selectedIndices() {
  return Array.from(selectedRows).sort(function(a, b) { return a - b; });
}

function selectedCsvText() {
  const indices = selectedIndices();
  const lines = [csvColumns.map(csvCell).join(',')];
  indices.forEach(function(index) {
    const row = csvRows[index];
    lines.push(csvColumns.map(function(column) {
      return csvCell(row[column]);
    }).join(','));
  });
  return lines.join('\\n') + '\\n';
}

function selectedStarText() {
  const indices = selectedIndices();
  let text = 'data_particles\\n\\nloop_\\n';
  starColumns.forEach(function(column, index) {
    text += '_' + column + ' #' + (index + 1) + '\\n';
  });
  indices.forEach(function(index) {
    const row = starRows[index];
    text += starColumns.map(function(column) {
      return starCell(row[column]);
    }).join(' ') + '\\n';
  });
  return text;
}

function downloadText(filename, text, mimeType) {
  const blob = new Blob([text], {type: mimeType});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(function() {
    URL.revokeObjectURL(url);
  }, 1000);
}

function selectedFilenameStem() {
  let value = filenameInput.value.trim();

  value = value.replace(/\\.(csv|star)$/i, '');
  value = value.replace(/[\\/:*?"<>|]+/g, '_');
  value = value.replace(/\\s+/g, '_');
  value = value.replace(/^_+|_+$/g, '');

  if (!value) value = 'selected_particles';
  filenameInput.value = value;
  return value;
}

downloadCsvButton.addEventListener('click', function() {
  if (selectedRows.size === 0) return;
  downloadText(
    selectedFilenameStem() + '.csv',
    selectedCsvText(),
    'text/csv;charset=utf-8');
});

downloadStarButton.addEventListener('click', function() {
  if (selectedRows.size === 0) return;
  downloadText(
    selectedFilenameStem() + '.star',
    selectedStarText(),
    'text/plain;charset=utf-8');
});

clearSelectionButton.addEventListener('click', function() {
  selectedRows.clear();
  Plotly.restyle(graph, {selectedpoints: null});
  Plotly.relayout(graph, {selections: []});
  updateSelectionState();
});

graph.on('plotly_selected', function(ev) {
  selectedRows.clear();

  if (ev && ev.points) {
    ev.points.forEach(function(point) {
      const cd = point.customdata;
      if (!cd || cd.length < 3) return;
      const rowIndex = Number(cd[2]);
      if (Number.isInteger(rowIndex) &&
          rowIndex >= 0 &&
          rowIndex < csvRows.length) {
        selectedRows.add(rowIndex);
      }
    });
  }

  updateSelectionState();
});

graph.on('plotly_deselect', function() {
  selectedRows.clear();
  updateSelectionState();
});

graph.on('plotly_hover', function(ev) {
  if (!ev.points || !ev.points.length) return;
  const cd = ev.points[0].customdata;
  if (cd && cd.length) {
    showCluster(String(cd[0]));
    showParticle(cd);
  }
});
graph.on('plotly_unhover', function() {
  if (pinned !== null) {
    showCluster(String(pinned[0]));
    showParticle(pinned);
  }
});
graph.on('plotly_click', function(ev) {
  if (!ev.points || !ev.points.length) return;
  const cd = ev.points[0].customdata;
  if (cd && cd.length) {
    pinned = cd;
    showCluster(String(cd[0]));
    showParticle(cd);
  }
});
showCluster(__DEFAULT__);
updateSelectionState();
})();
</script>
</body>
</html>'''

    html = html.replace('__TITLE__', str(title))
    html = html.replace('__PLOT__', plot_html)
    html = html.replace('__PREVIEWS__', json.dumps(preview_data))
    html = html.replace('__PLOT_ID__', json.dumps(plot_id))
    html = html.replace('__DEFAULT__', json.dumps(str(default_label)))
    html = html.replace('__CSV_COLUMNS__', json.dumps(csv_columns))
    html = html.replace('__STAR_COLUMNS__', json.dumps(star_columns))
    html = html.replace('__CSV_ROWS__', csv_rows_json)
    html = html.replace('__STAR_ROWS__', star_rows_json)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def embedding_cache_paths(class_dir):
    return {
        'npz': Path(class_dir) / 'simsiam_embeddings.npz',
        'csv': Path(class_dir) / 'simsiam_embedding_metadata.csv',
        'json': Path(class_dir) / 'simsiam_embedding_cache.json',
    }


def particle_table_hash(df):
    hash_cols = [
        col for col in (
            'tomo', 'instance_id', 'class_name',
            'center_x', 'center_y', 'center_z',
            'average_center_x', 'average_center_y', 'average_center_z',
            'r00', 'r01', 'r02',
            'r10', 'r11', 'r12',
            'r20', 'r21', 'r22',
            '_particle_index',
        )
        if col in df.columns
    ]
    if not hash_cols:
        return None

    hashed = pd.util.hash_pandas_object(
        df[hash_cols], index=False).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def build_embedding_cache_manifest(class_name, class_df, params, data_cfg):
    orientation_path = Path(params['orientations_file'])
    model_path = Path(params['simsiam_model'])

    manifest = {
        'class_name': str(class_name),
        'n_particles_requested': int(len(class_df)),
        'particle_table_hash': particle_table_hash(class_df),
        'orientations_file': str(orientation_path.resolve()),
        'orientations_mtime_ns': int(
            orientation_path.stat().st_mtime_ns),
        'simsiam_model': str(model_path.resolve()),
        'simsiam_model_mtime_ns': int(
            model_path.stat().st_mtime_ns),
        'prediction_folder': str(
            Path(params['prediction_folder']).resolve()),
        'data_folder': str(
            Path(params['data_folder']).resolve()),
        'contrastive': bool(params['contrastive']),
        'use_mask': bool(params['use_mask']),
        'mask_mode': str(params['mask_mode']),
        'mask_dataset': str(params['mask_dataset']),
        'mask_expand': int(params['mask_expand']),
        'skip_border_particles': bool(
            params['skip_border_particles']),
        'patch_size': [int(v) for v in data_cfg['patch_size']],
        'data_min': float(data_cfg['min']),
        'data_max': float(data_cfg['max']),
        'data_mean': float(data_cfg['mean']),
        'data_std': float(data_cfg['std']),
        'max_particles': (
            None if params['max_particles'] is None
            else int(params['max_particles'])),
    }
    return manifest


def load_embedding_cache_manifest(cache_paths):
    path = cache_paths['json']
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def embedding_cache_is_valid(cache_paths, expected_manifest):
    if not (cache_paths['npz'].is_file() and
            cache_paths['csv'].is_file() and
            cache_paths['json'].is_file()):
        return False

    actual = load_embedding_cache_manifest(cache_paths)
    if actual is None:
        return False

    return actual == expected_manifest


def save_embedding_cache(cache_paths, embeddings, metadata, manifest):
    cache_paths['npz'].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_paths['npz'],
        embeddings=np.asarray(embeddings, dtype=np.float32))
    metadata.to_csv(cache_paths['csv'], index=False)
    cache_paths['json'].write_text(
        json.dumps(manifest, indent=2, sort_keys=True))


def load_embedding_cache(cache_paths):
    with np.load(cache_paths['npz']) as cached:
        embeddings = cached['embeddings'].astype(np.float32, copy=False)

    metadata = pd.read_csv(cache_paths['csv'])
    metadata = metadata.reset_index(drop=True)

    if len(metadata) != len(embeddings):
        raise RuntimeError(
            f'Cached SimSiam metadata has {len(metadata)} rows but '
            f'the cached embedding matrix has {len(embeddings)} rows.')

    return embeddings, metadata


def process_class(class_name, class_df, params, data_cfg, net, dim, device,
                  distributed=False, rank=0, world_size=1, tmp_dir=None):
    safe_name = str(class_name).replace('/', '_').replace(' ', '_')
    class_dir = params['output_folder'] / safe_name
    class_dir.mkdir(parents=True, exist_ok=True)

    if params['max_particles'] is not None:
        class_df = class_df.iloc[:params['max_particles']].copy()

    cache_paths = embedding_cache_paths(class_dir)
    expected_manifest = build_embedding_cache_manifest(
        class_name, class_df, params, data_cfg)

    use_cached_embeddings = False
    if rank == 0:
        use_cached_embeddings = (
                params['reuse_saved_embeddings'] and
                not params['force_recompute_embeddings'] and
                embedding_cache_is_valid(cache_paths, expected_manifest))

    if distributed:
        payload = [bool(use_cached_embeddings)]
        dist.broadcast_object_list(payload, src=0)
        use_cached_embeddings = bool(payload[0])

    if rank == 0:
        print(f'\nClass {class_name}: {len(class_df)} particle(s)')
        if use_cached_embeddings:
            print('  Using cached SimSiam embeddings.')
        elif params['force_recompute_embeddings']:
            print('  Recomputing SimSiam embeddings because force_recompute_embeddings=true.')
        else:
            print('  No valid embedding cache found; extracting SimSiam embeddings.')

    if use_cached_embeddings:
        if rank != 0:
            # Wait until rank 0 finishes clustering/averaging for this class.
            dist.barrier()
            return None

        embeddings, metadata = load_embedding_cache(cache_paths)
        print(
            f'  loaded cached embedding matrix: '
            f'{embeddings.shape[0]} x {embeddings.shape[1]}')

    else:
        # ------------------------------------------------------------------
        # Multi-GPU embedding extraction.
        # Every rank gets an interleaved particle shard and runs the frozen
        # SimSiam encoder on its LOCAL_RANK GPU. Only rank 0 subsequently runs
        # PCA/KMeans/UMAP and writes the final outputs.
        # ------------------------------------------------------------------
        if distributed:
            local_df = class_df.iloc[rank::world_size].copy()
            print(
                f'  [rank {rank}/{world_size}] GPU {device}: '
                f'{len(local_df)} particle(s)')

            if len(local_df) > 0:
                local_embeddings, local_metadata = extract_embeddings(
                    local_df, net, dim, params, data_cfg, device)
                local_indices = local_metadata['_particle_index'].to_numpy(
                    dtype=np.int64, copy=True)
            else:
                local_embeddings = np.empty((0, int(dim)), dtype=np.float32)
                local_indices = np.empty((0,), dtype=np.int64)

            shard_path = Path(tmp_dir) / f'{safe_name}.rank{rank:04d}.npz'
            np.savez(
                shard_path,
                embeddings=local_embeddings.astype(np.float32, copy=False),
                particle_indices=local_indices)

            # All embedding shards must be completely written before rank 0 merges.
            dist.barrier()

            if rank != 0:
                # Keep the worker alive until rank 0 has finished clustering,
                # averaging and writing the report for this class.
                dist.barrier()
                return None

            embedding_chunks = []
            index_chunks = []
            embedding_width = None

            for worker_rank in range(world_size):
                path = Path(tmp_dir) / f'{safe_name}.rank{worker_rank:04d}.npz'
                with np.load(path) as shard:
                    emb = shard['embeddings'].astype(np.float32, copy=False)
                    idx = shard['particle_indices'].astype(np.int64, copy=False)

                if len(emb) != len(idx):
                    raise RuntimeError(
                        f'Embedding shard {path} has {len(emb)} embeddings but '
                        f'{len(idx)} particle indices.')

                if emb.ndim != 2:
                    raise RuntimeError(
                        f'Embedding shard {path} has invalid shape {emb.shape}.')

                if len(emb) > 0:
                    if embedding_width is None:
                        embedding_width = emb.shape[1]
                    elif emb.shape[1] != embedding_width:
                        raise RuntimeError(
                            'SimSiam embedding dimensionality differs between GPU shards: '
                            f'{embedding_width} versus {emb.shape[1]}.')
                    embedding_chunks.append(emb)
                    index_chunks.append(idx)

                try:
                    path.unlink()
                except OSError:
                    pass

            if not embedding_chunks:
                raise RuntimeError('No SimSiam embeddings were extracted on any GPU.')

            embeddings = np.concatenate(embedding_chunks, axis=0)
            particle_indices = np.concatenate(index_chunks, axis=0)
            order = np.argsort(particle_indices, kind='stable')
            embeddings = embeddings[order]
            particle_indices = particle_indices[order]

            source_by_index = class_df.set_index('_particle_index', drop=False)
            missing = [
                int(i) for i in particle_indices
                if int(i) not in source_by_index.index]
            if missing:
                raise RuntimeError(
                    f'Could not map merged embedding indices back to particles: '
                    f'{missing[:5]}')

            metadata = source_by_index.loc[particle_indices].reset_index(drop=True)
            print(
                f'  merged embedding matrix from {world_size} GPU(s): '
                f'{embeddings.shape[0]} x {embeddings.shape[1]}')

        else:
            embeddings, metadata = extract_embeddings(
                class_df, net, dim, params, data_cfg, device)

        if params['save_embeddings_cache']:
            save_embedding_cache(
                cache_paths, embeddings, metadata, expected_manifest)
            print(f'  saved embedding cache: {cache_paths["npz"]}')

    # From here on only rank 0 executes in distributed mode. sklearn
    # clustering, UMAP and scipy-based raw averaging are CPU-side operations.
    cluster_features, pca = prepare_cluster_space(embeddings, params)
    labels, cluster_model = cluster_particles(
        cluster_features, params)

    print('  Computing UMAP for visualization...')
    umap_xy = compute_umap(cluster_features, params)

    metadata = metadata.copy()
    metadata['cluster'] = labels
    metadata['umap_1'] = umap_xy[:, 0]
    metadata['umap_2'] = umap_xy[:, 1]

    csv_path, joined_star_path, individual_cluster_files = save_cluster_files(
        metadata,
        class_dir,
        save_individual=params['save_individual_cluster_files'])

    print(f'  Cluster CSV:  {csv_path}')
    print(f'  Cluster STAR: {joined_star_path}')
    if params['save_individual_cluster_files']:
        print(
            f'  Individual cluster CSV/STAR pairs: '
            f'{len(individual_cluster_files)} -> {class_dir / "cluster_files"}')

    npz_data = {
        'embeddings': embeddings.astype(np.float32),
        'cluster_features': cluster_features.astype(np.float32),
        'cluster': labels.astype(np.int32),
        'umap': umap_xy.astype(np.float32),
        'clustering_method': np.asarray(params['clustering_method']),
    }

    if hasattr(cluster_model, 'cluster_centers_'):
        npz_data['kmeans_centers'] = (
            cluster_model.cluster_centers_.astype(np.float32))

    if pca is not None:
        npz_data['pca_explained_variance_ratio'] = (
            pca.explained_variance_ratio_.astype(np.float32))

    npz_path = class_dir / 'clustering_features.npz'
    np.savez_compressed(npz_path, **npz_data)

    averages = {}
    if params['save_raw_averages']:
        print('  Creating aligned raw subtomogram averages for the clusters...')
        averages = create_cluster_averages(metadata, params, class_dir)

    if params['show_particle_crops_in_umap']:
        print('  Creating particle crop previews for the UMAP...')
        metadata = create_particle_crop_previews(
            metadata, params, data_cfg, class_dir)

    html_path = class_dir / 'subtomogram_clusters_umap.html'
    method_title = (
        'KMeans'
        if params['clustering_method'] == 'kmeans'
        else 'Spectral')
    make_interactive_umap(
        metadata,
        averages,
        html_path,
        title=(
            f'{class_name}: SimSiam embedding '
            f'{method_title} clustering'))

    print(f'  NPZ:  {npz_path}')
    print(f'  HTML: {html_path}')

    if distributed:
        # Release the non-zero ranks waiting above.
        dist.barrier()

    return {'csv': csv_path,
            'star': joined_star_path,
            'cluster_files': individual_cluster_files,
            'npz': npz_path,
            'html': html_path}


def main(config_file):
    distributed, rank, world_size, local_rank, device = init_distributed()

    cfg, params = load_params(config_file)
    params['output_folder'].mkdir(parents=True, exist_ok=True)

    requested_nodes, requested_gpus = read_requested_topology(cfg)
    expected_world_size = requested_nodes * requested_gpus
    if world_size != expected_world_size:
        raise RuntimeError(
            f'Expected WORLD_SIZE={expected_world_size} from '
            f'parameters.nodes={requested_nodes} and '
            f'parameters.gpu_devices={requested_gpus}, got {world_size}.')

    data_cfg = resolve_simsiam_data_cfg(cfg, params['simsiam_model'])

    if rank == 0:
        print('SimSiam subtomogram clustering:')
        print(f'  model: {params["simsiam_model"]}')
        print(f'  orientations: {params["orientations_file"]}')
        print(
            f'  orientation format: '
            f'{params["orientations_file"].suffix.lower().lstrip(".").upper()}')
        print(f'  data folder: {params["data_folder"]}')
        print(f'  prediction folder: {params["prediction_folder"]}')
        print(
            f'  workers: {world_size} GPU process(es) '
            f'({requested_nodes} node(s) x {requested_gpus} GPU(s))')
        print(f'  patch size: {data_cfg["patch_size"]}')
        print(f'  contrastive SimSiam: {params["contrastive"]}')
        print(
            f'  embedding cache: reuse={params["reuse_saved_embeddings"]}, '
            f'save={params["save_embeddings_cache"]}, '
            f'force_recompute={params["force_recompute_embeddings"]}')
        if params['use_mask']:
            print(
                f'  masking: {params["mask_mode"]} '
                f'(dataset={params["mask_dataset"]}, expand={params["mask_expand"]})')
        else:
            print('  masking: disabled')
        print(
            f'  PCA before clustering: '
            f'{params["use_pca"]}')
        if params['use_pca']:
            print(
                f'  PCA components: '
                f'{params["pca_components"]}')
        print(
            f'  clustering method: '
            f'{params["clustering_method"]}')
        print(f'  clusters: {params["n_clusters"]}')
        if params['clustering_method'] == 'spectral':
            print(
                f'  spectral neighbors: '
                f'{params["spectral_n_neighbors"]}')
        print('  joined cluster CSV/STAR: always saved')
        print(
            f'  individual cluster CSV/STAR files: '
            f'{params["save_individual_cluster_files"]}')
        print(
            f'  particle crop previews: '
            f'{params["show_particle_crops_in_umap"]}')
        print(f'  output: {params["output_folder"]}')

    print(f'[rank {rank}] Loading frozen SimSiam model on {device}...')
    net, dim = load_simsiam(
        params['simsiam_model'], params['contrastive'], device)
    if rank == 0:
        print(f'  embedding dim: {dim}')

    df = read_orientation_table(params['orientations_file'],
                                prediction_folder=params['prediction_folder'],
                                class_names=params['class_names'],
                                use_refined_centers_for_average=params['use_refined_centers_for_average'],
                                mask_dataset=params['mask_dataset'])

    classes = sorted(df['class_name'].astype(str).unique())
    if rank == 0:
        print(f'  classes: {classes}')

    tmp_dir = make_run_temp_dir(
        params['output_folder'], distributed, rank)

    results = {}
    try:
        for class_name in classes:
            class_df = df[df['class_name'].astype(str) == class_name].copy()
            result = process_class(class_name, class_df, params, data_cfg, net, dim, device,
                                   distributed=distributed, rank=rank, world_size=world_size,
                                   tmp_dir=tmp_dir)
            if rank == 0:
                results[class_name] = result

        if rank == 0:
            print('\nFinished clustering.')
            for class_name, result in results.items():
                print(f'  {class_name}: {result["html"]}')

    finally:
        if distributed:
            dist.barrier()
        if rank == 0:
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

def run(config_file):
    """CLI/package entrypoint with optional multi-GPU launching."""
    if not maybe_launch_distributed_from_config(config_file):
        return main(config_file)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Cluster cryo-ET subtomograms using SimSiam embeddings, '
            'KMeans or spectral clustering, UMAP, and cluster averages.'))
    parser.add_argument(
        '--config_file',
        required=True,
        help='CryoSiam YAML config file.')
    args = parser.parse_args()
    run(args.config_file)
