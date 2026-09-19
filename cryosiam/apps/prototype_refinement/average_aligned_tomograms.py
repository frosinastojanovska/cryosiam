import argparse
import shlex
from pathlib import Path

import mrcfile
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

REQUIRED_STAR_COLUMNS = {
    'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
    'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
}


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _simple_star_table(path):
    lines = Path(path).read_text().splitlines()
    columns = []
    rows = []
    in_loop = False
    reading_rows = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'):
            if reading_rows and rows:
                break
            continue

        lower = line.lower()
        if lower == 'loop_':
            in_loop = True
            reading_rows = False
            columns = []
            rows = []
            continue

        if in_loop and line.startswith('_'):
            columns.append(line.split()[0].lstrip('_'))
            continue

        if in_loop and columns:
            if lower.startswith('data_') or lower == 'loop_' or line.startswith('_'):
                if rows:
                    break
                continue
            values = shlex.split(line)
            if len(values) < len(columns):
                continue
            rows.append(values[:len(columns)])
            reading_rows = True

    if not columns or not rows:
        raise ValueError(f'Could not find a STAR loop table in {path}')

    data = {name: [] for name in columns}
    for row in rows:
        for name, value in zip(columns, row):
            data[name].append(value)
    return data


def read_star(path):
    try:
        import starfile
        table = starfile.read(path)
        if isinstance(table, dict):
            tables = [value for value in table.values() if hasattr(value, 'columns')]
            if not tables:
                raise ValueError(f'No tabular block found in STAR file: {path}')
            table = tables[-1]
        data = {str(column): table[column].to_numpy() for column in table.columns}
    except ImportError:
        data = _simple_star_table(path)

    missing = REQUIRED_STAR_COLUMNS.difference(data)
    if missing:
        raise ValueError(f'STAR file is missing required columns: {sorted(missing)}')

    if 'rlnTomoName' in data:
        tomo_key = 'rlnTomoName'
    elif 'rlnMicrographName' in data:
        tomo_key = 'rlnMicrographName'
    else:
        raise ValueError('STAR file must contain rlnTomoName or rlnMicrographName.')

    n = len(data['rlnCoordinateX'])
    rows = []
    for i in range(n):
        row = {
            'tomo': str(data[tomo_key][i]),
            'x': float(data['rlnCoordinateX'][i]),
            'y': float(data['rlnCoordinateY'][i]),
            'z': float(data['rlnCoordinateZ'][i]),
            'rot': float(data['rlnAngleRot'][i]),
            'tilt': float(data['rlnAngleTilt'][i]),
            'psi': float(data['rlnAnglePsi'][i]),
        }
        if 'rlnClassLabel' in data:
            row['class_name'] = str(data['rlnClassLabel'][i])
        rows.append(row)
    return rows


def normalize_size(value):
    if isinstance(value, (int, float)):
        size = [int(value)] * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        size = [int(v) for v in value]
    else:
        raise ValueError('crop_size must be an integer or [z, y, x].')
    if any(v < 8 for v in size):
        raise ValueError('crop_size values must be >= 8.')
    return tuple(size)


def tomo_path(data_folder, tomo_name, extension):
    path = Path(data_folder) / f'{tomo_name}{extension}'
    if path.is_file():
        return path

    if tomo_name.endswith(extension):
        alternate = Path(data_folder) / tomo_name
        if alternate.is_file():
            return alternate

    raise FileNotFoundError(f'Could not find tomogram for {tomo_name}: expected {path}')


def star_angles_to_particle_to_reference(rot, tilt, psi):
    # The orientation scripts save RELION angles for reference -> particle.
    # To align the observed particle back into the reference frame, invert it.
    R_reference_to_particle = Rotation.from_euler(
        'ZYZ', [float(rot), float(tilt), float(psi)], degrees=True).as_matrix()
    return R_reference_to_particle.T


def xyz_rotation_to_zyx(R_xyz):
    P = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0]], dtype=np.float64)
    return P @ np.asarray(R_xyz, dtype=np.float64) @ P


def output_center(shape):
    return (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0


def transformed_output_corners(center_zyx, shape, R_particle_to_reference):
    shape = np.asarray(shape, dtype=np.int64)
    out_center = output_center(shape)
    R_zyx = xyz_rotation_to_zyx(R_particle_to_reference)
    matrix = R_zyx.T

    corners = np.array([
        [z, y, x]
        for z in (0.0, float(shape[0] - 1))
        for y in (0.0, float(shape[1] - 1))
        for x in (0.0, float(shape[2] - 1))
    ], dtype=np.float64)
    return np.asarray(center_zyx, dtype=np.float64)[None, :] + \
        (corners - out_center[None, :]) @ matrix.T


def particle_fits(volume_shape, center_zyx, crop_size, R_particle_to_reference):
    corners = transformed_output_corners(center_zyx, crop_size, R_particle_to_reference)
    shape = np.asarray(volume_shape, dtype=np.float64)
    return bool(np.all(corners >= 0.0) and np.all(corners <= (shape - 1.0)[None, :]))


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


TILT_AXIS_XYZ = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def tomogram_wedge_mask_centered(shape, min_tilt_angle, max_tilt_angle):
    """
    Binary Fourier-space sampling mask for a tomogram reconstructed from a
    continuous tilt range about the fixed +Y axis. The returned array is
    fftshifted, so the zero frequency is at the center and the mask can be
    rotated around the particle center before being shifted back for FFT use.
    """
    min_tilt = float(min_tilt_angle)
    max_tilt = float(max_tilt_angle)
    if not np.isfinite(min_tilt) or not np.isfinite(max_tilt):
        raise ValueError('min_tilt_angle and max_tilt_angle must be finite.')
    if min_tilt >= max_tilt:
        raise ValueError('min_tilt_angle must be smaller than max_tilt_angle.')
    if (max_tilt - min_tilt) >= 180.0:
        return np.ones(tuple(int(v) for v in shape), dtype=np.float32)

    D, H, W = [int(v) for v in shape]
    fz = np.fft.fftshift(np.fft.fftfreq(D)).astype(np.float32)
    fy = np.fft.fftshift(np.fft.fftfreq(H)).astype(np.float32)
    fx = np.fft.fftshift(np.fft.fftfreq(W)).astype(np.float32)
    zz, yy, xx = np.meshgrid(fz, fy, fx, indexing='ij')

    # For a tilt around +Y, the beam direction is
    # b(theta) = [sin(theta), 0, cos(theta)]. A Fourier vector q is observed
    # when q dot b(theta) = 0 for at least one theta in the acquired range.
    # The solution is periodic by 180 degrees because +b and -b define the
    # same Fourier central section.
    theta = np.degrees(np.arctan2(-zz, xx)).astype(np.float32)
    observed = ((theta >= min_tilt) & (theta <= max_tilt))
    observed |= ((theta + 180.0 >= min_tilt) & (theta + 180.0 <= max_tilt))
    observed |= ((theta - 180.0 >= min_tilt) & (theta - 180.0 <= max_tilt))

    # Frequencies along the tilt axis lie in every central section.
    axis_line = (np.abs(xx) < 1e-12) & (np.abs(zz) < 1e-12)
    observed |= axis_line
    return observed.astype(np.float32)


def rotate_centered_cube(volume, R_particle_to_reference, order=1):
    R_zyx = xyz_rotation_to_zyx(R_particle_to_reference)
    matrix = R_zyx.T
    center = output_center(volume.shape)
    offset = center - matrix @ center
    return affine_transform(
        volume,
        matrix=matrix,
        offset=offset,
        output_shape=volume.shape,
        order=int(order),
        mode='constant',
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)


def fast_input_cube_shape(crop_size):
    """Fixed source cube large enough for any rotation of the output crop."""
    half_extent = (np.asarray(crop_size, dtype=np.float64) - 1.0) / 2.0
    radius = float(np.linalg.norm(half_extent))
    side = int(2 * np.ceil(radius) + 3)
    if side % 2 == 0:
        side += 1
    return (side, side, side)


def extract_fixed_patch_into(dest, volume, center_zyx, fill_value=0.0):
    """Extract a fixed-size patch while preserving the fractional particle center."""
    shape = np.asarray(dest.shape, dtype=np.int64)
    center = np.asarray(center_zyx, dtype=np.float64)
    start = np.floor(center - (shape.astype(np.float64) - 1.0) / 2.0).astype(np.int64)
    stop = start + shape

    dest.fill(float(fill_value))
    vol_shape = np.asarray(volume.shape, dtype=np.int64)
    src_start = np.maximum(start, 0)
    src_stop = np.minimum(stop, vol_shape)
    if np.any(src_stop <= src_start):
        return center - start.astype(np.float64)

    dst_start = src_start - start
    dst_stop = dst_start + (src_stop - src_start)
    src_slc = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_stop))
    dst_slc = tuple(slice(int(a), int(b)) for a, b in zip(dst_start, dst_stop))
    dest[dst_slc] = np.asarray(volume[src_slc], dtype=np.float32)
    return center - start.astype(np.float64)


def torch_output_offsets_zyx(shape, device):
    D, H, W = [int(v) for v in shape]
    center = torch.tensor(
        [(D - 1.0) / 2.0, (H - 1.0) / 2.0, (W - 1.0) / 2.0],
        device=device, dtype=torch.float32)
    z = torch.arange(D, device=device, dtype=torch.float32) - center[0]
    y = torch.arange(H, device=device, dtype=torch.float32) - center[1]
    x = torch.arange(W, device=device, dtype=torch.float32) - center[2]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    return torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)


def torch_rotation_grid(rotations_xyz, source_centers_zyx, source_shape,
                        target_shape, offsets_zyx, device):
    rotations_xyz = np.asarray(rotations_xyz, dtype=np.float32)
    # P @ R_xyz @ P is equivalent to reversing both matrix axes.
    rotations_zyx = np.ascontiguousarray(rotations_xyz[:, ::-1, ::-1])
    R = torch.from_numpy(rotations_zyx).to(device=device, dtype=torch.float32)
    centers = torch.as_tensor(
        np.asarray(source_centers_zyx, dtype=np.float32),
        device=device, dtype=torch.float32)

    # scipy affine_transform used input = center + R_zyx.T @ output_offset.
    # With row vectors this is output_offset @ R_zyx.
    coords = torch.matmul(offsets_zyx.unsqueeze(0), R) + centers[:, None, :]

    D, H, W = [float(v) for v in source_shape]
    gz = 2.0 * (coords[..., 0] + 0.5) / D - 1.0
    gy = 2.0 * (coords[..., 1] + 0.5) / H - 1.0
    gx = 2.0 * (coords[..., 2] + 0.5) / W - 1.0
    grid = torch.stack((gx, gy, gz), dim=-1)
    return grid.reshape(len(rotations_xyz), *tuple(int(v) for v in target_shape), 3)


def torch_align_particle_batch(patches_np, source_centers_zyx, rotations_xyz,
                               crop_size, offsets_zyx, device):
    patches = torch.from_numpy(patches_np).to(
        device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
    grid = torch_rotation_grid(
        rotations_xyz, source_centers_zyx, patches_np.shape[-3:],
        crop_size, offsets_zyx, device)
    return F.grid_sample(
        patches, grid, mode='bilinear', padding_mode='zeros',
        align_corners=False)[:, 0]


def torch_rotate_wedge_batch(base_wedge_centered, rotations_xyz, crop_size,
                             offsets_zyx, device):
    B = len(rotations_xyz)
    source = base_wedge_centered.expand(B, -1, -1, -1, -1)
    center = output_center(crop_size).astype(np.float32)
    centers = np.repeat(center[None, :], B, axis=0)
    grid = torch_rotation_grid(
        rotations_xyz, centers, crop_size, crop_size, offsets_zyx, device)
    wedge = F.grid_sample(
        source, grid, mode='bilinear', padding_mode='zeros',
        align_corners=False)[:, 0]

    # scipy.ndimage mode='constant' returns cval as soon as the requested
    # coordinate lies outside [0, size-1]. grid_sample otherwise blends with
    # zero for the half-voxel border region, so explicitly mask that region to
    # preserve the previous wedge-rotation semantics.
    D, H, W = [int(v) for v in crop_size]
    valid = (
            (grid[..., 0].abs() <= (1.0 - 1.0 / float(W) + 1e-6)) &
            (grid[..., 1].abs() <= (1.0 - 1.0 / float(H) + 1e-6)) &
            (grid[..., 2].abs() <= (1.0 - 1.0 / float(D) + 1e-6))
    )
    wedge = wedge * valid.to(wedge.dtype)
    return wedge.clamp_(0.0, 1.0)


def torch_normalize_particles(particles, mask, normalize=True, apply_mask=True):
    if normalize:
        core = (mask > 0.5).to(particles.dtype)
        count = core.sum().clamp_min(1.0)
        mean = (particles * core).sum(dim=(-3, -2, -1), keepdim=True) / count
        centered = particles - mean
        var = (centered.square() * core).sum(
            dim=(-3, -2, -1), keepdim=True) / count
        particles = centered / torch.sqrt(var.clamp_min(1e-12))
    if apply_mask:
        particles = particles * mask
    return particles


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


def save_middle_slice_png(path, volume):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError('Saving PNG previews requires Pillow.') from exc

    image = np.asarray(volume[volume.shape[0] // 2], dtype=np.float32)
    finite = np.isfinite(image)
    if np.any(finite):
        lo, hi = np.percentile(image[finite], [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        norm = np.zeros_like(image, dtype=np.float32)
    else:
        norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.round(255.0 * norm).astype(np.uint8)
    Image.fromarray(rgb, mode='L').save(path)


def rotation_from_row(row, use_orientations=True):
    if use_orientations:
        return star_angles_to_particle_to_reference(row['rot'], row['tilt'], row['psi'])
    return np.eye(3, dtype=np.float64)


def _lambert_equal_area_xy(directions, hemisphere='north'):
    dirs = np.asarray(directions, dtype=np.float64)
    if dirs.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = np.clip(dirs[:, 2], -1.0, 1.0)
    if hemisphere == 'north':
        denom = np.clip(1.0 + z, 1e-8, None)
    else:
        denom = np.clip(1.0 - z, 1e-8, None)
    scale = np.sqrt(1.0 / denom)
    return x * scale, y * scale


def _point_density(x, y):
    """Local density per point, used only for color -- purely cosmetic."""
    if len(x) < 3:
        return np.ones(len(x), dtype=np.float64)
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y])
        return gaussian_kde(xy)(xy)
    except Exception:
        return np.ones(len(x), dtype=np.float64)


def save_angular_distribution_plot(rows, path, title='Angular distribution'):
    directions = []
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for row in rows:
        R = star_angles_to_particle_to_reference(row['rot'], row['tilt'], row['psi'])
        d = np.asarray(R @ axis, dtype=np.float64)
        n = np.linalg.norm(d)
        if np.isfinite(n) and n > 0:
            directions.append(d / n)
    directions = np.asarray(directions, dtype=np.float64)
    if len(directions) == 0:
        return None

    # Each viewing direction is a point on the unit sphere. A single 3D plot
    # hides whichever half of the sphere faces away from the camera, so
    # instead we split into front/back hemispheres and flatten each with a
    # Lambert azimuthal equal-area projection: this preserves the *density*
    # of points per unit area, so the resulting disc genuinely reflects how
    # densely a given viewing direction was sampled, and a well-covered
    # dataset fills the whole disc solidly, as in cryoSPARC/RELION-style
    # angular distribution plots.
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 12.5))
    boundary = np.linspace(0.0, 2.0 * np.pi, 200)

    panels = [
        ('north', 'Front hemisphere (z \u2265 0)'),
        ('south', 'Back hemisphere (z < 0)'),
    ]

    for ax, (hemisphere, subtitle) in zip(axes, panels):
        mask = directions[:, 2] >= 0.0 if hemisphere == 'north' else directions[:, 2] < 0.0
        pts = directions[mask]

        ax.plot(np.cos(boundary), np.sin(boundary), color='0.3', linewidth=1.2, zorder=1)

        if len(pts) > 0:
            x, y = _lambert_equal_area_xy(pts, hemisphere=hemisphere)
            density = _point_density(x, y)
            # Draw densest points last so they aren't buried under sparse ones.
            order = np.argsort(density)
            ax.scatter(
                x[order], y[order], c=density[order], cmap='plasma',
                s=5.0, alpha=0.55, linewidths=0.0, zorder=2,
            )

        ax.set_aspect('equal')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis('off')
        ax.set_title(subtitle, fontsize=11)

    fig.suptitle(f'{title} (n={len(directions)})', fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches='tight', transparent=False)
    plt.close(fig)
    return path


def load_params(config_file):
    cfg = read_yaml(config_file)
    if 'data_folder' not in cfg:
        raise KeyError('Config is missing top-level data_folder.')

    avg_cfg = cfg.get('subtomogram_average') or {}
    if not isinstance(avg_cfg, dict):
        raise ValueError('subtomogram_average must be a mapping.')

    gpu_value = (cfg.get('parameters') or {}).get('gpu_devices', 1)
    if torch.cuda.is_available():
        if isinstance(gpu_value, (list, tuple)):
            gpu_id = int(gpu_value[0]) if gpu_value else None
        else:
            gpu_id = 0 if int(gpu_value) > 0 else None
        if gpu_id is not None and (gpu_id < 0 or gpu_id >= torch.cuda.device_count()):
            raise ValueError(
                f'Invalid GPU device {gpu_id}; this process sees '
                f'{torch.cuda.device_count()} CUDA device(s).')
    else:
        gpu_id = None
    device = torch.device(f'cuda:{gpu_id}') if gpu_id is not None else torch.device('cpu')

    star_value = avg_cfg.get('star_file')
    if not star_value:
        raise ValueError('Set subtomogram_average.star_file in the config.')
    star_path = Path(star_value)
    if not star_path.is_file():
        raise FileNotFoundError(star_path)

    crop_value = avg_cfg.get(
        'crop_size',
        (cfg.get('initial_orientations') or {}).get('registration_crop_size', 64))
    if isinstance(crop_value, dict):
        crop_value = crop_value.get('default', 64)

    mode = str(avg_cfg.get('average_mode', 'real_space')).lower()
    if mode not in ('real_space', 'wedge_weighted'):
        raise ValueError("average_mode must be 'real_space' or 'wedge_weighted'.")

    out = Path(avg_cfg.get('output_folder') or star_path.parent / 'subtomogram_averages')

    params = {
        'star_file': star_path,
        'data_folder': Path(cfg['data_folder']),
        'file_extension': str(cfg.get('file_extension', '.mrc')),
        'output_folder': out,
        'crop_size': normalize_size(crop_value),
        'average_mode': mode,
        'class_name': avg_cfg.get('class_name'),
        'group_by_class': bool(avg_cfg.get('group_by_class', False)),
        'skip_border_particles': bool(avg_cfg.get('skip_border_particles', True)),
        'normalize_particles': bool(avg_cfg.get('normalize_particles', True)),
        'apply_soft_mask': bool(avg_cfg.get('apply_soft_mask', True)),
        'mask_radius_fraction': float(avg_cfg.get('mask_radius_fraction', 0.46)),
        'mask_edge_fraction': float(avg_cfg.get('mask_edge_fraction', 0.06)),
        'max_particles': avg_cfg.get('max_particles'),
        'min_tilt_angle': avg_cfg.get('min_tilt_angle'),
        'max_tilt_angle': avg_cfg.get('max_tilt_angle'),
        'wedge_weight_floor': float(avg_cfg.get('wedge_weight_floor', 1e-3)),
        'batch_size': int(avg_cfg.get('batch_size', 32)),
        'device': device,
        'save_no_rotation_average': bool(avg_cfg.get('save_no_rotation_average', True)),
        'save_oriented_average': bool(avg_cfg.get('save_oriented_average', True)),
        'save_angular_distribution': bool(avg_cfg.get('save_angular_distribution', True)),
    }

    if params['max_particles'] is not None:
        params['max_particles'] = int(params['max_particles'])
        if params['max_particles'] < 1:
            raise ValueError('max_particles must be >= 1 or omitted.')
    if params['batch_size'] < 1:
        raise ValueError('subtomogram_average.batch_size must be >= 1.')

    if params['mask_radius_fraction'] <= 0 or params['mask_radius_fraction'] > 1:
        raise ValueError('mask_radius_fraction must be in (0, 1].')
    if params['mask_edge_fraction'] < 0:
        raise ValueError('mask_edge_fraction must be >= 0.')
    if params['wedge_weight_floor'] <= 0:
        raise ValueError('wedge_weight_floor must be > 0.')

    if params['average_mode'] == 'wedge_weighted':
        if params['min_tilt_angle'] is None or params['max_tilt_angle'] is None:
            raise ValueError(
                'wedge_weighted mode requires subtomogram_average.min_tilt_angle '
                'and subtomogram_average.max_tilt_angle.')
        params['min_tilt_angle'] = float(params['min_tilt_angle'])
        params['max_tilt_angle'] = float(params['max_tilt_angle'])
        if not np.isfinite(params['min_tilt_angle']) or not np.isfinite(params['max_tilt_angle']):
            raise ValueError('min_tilt_angle and max_tilt_angle must be finite.')
        if params['min_tilt_angle'] >= params['max_tilt_angle']:
            raise ValueError('min_tilt_angle must be smaller than max_tilt_angle.')

    return params


def select_rows(rows, params):
    class_name = params['class_name']
    if class_name is not None:
        if not rows or 'class_name' not in rows[0]:
            raise ValueError('class_name filtering requires rlnClassLabel in the STAR file.')
        rows = [row for row in rows if str(row.get('class_name')) == str(class_name)]

    if params['max_particles'] is not None:
        rows = rows[:params['max_particles']]

    if not rows:
        raise RuntimeError('No particles remain after STAR filtering.')
    return rows


def group_rows(rows, group_by_class=False):
    if not group_by_class:
        return {'all': rows}

    if 'class_name' not in rows[0]:
        raise ValueError('group_by_class=True requires rlnClassLabel in the STAR file.')

    grouped = {}
    for row in rows:
        grouped.setdefault(str(row['class_name']), []).append(row)
    return grouped


def average_group_cpu(rows, params, output_prefix, use_orientations=True):
    crop_size = params['crop_size']
    mask = spherical_mask(
        crop_size,
        radius_fraction=params['mask_radius_fraction'],
        edge_fraction=params['mask_edge_fraction'])

    real_sum = np.zeros(crop_size, dtype=np.float64)
    real_count = 0
    fourier_sum = np.zeros(crop_size, dtype=np.complex128)
    fourier_weight = np.zeros(crop_size, dtype=np.float64)
    base_wedge_centered = None
    if params['average_mode'] == 'wedge_weighted':
        base_wedge_centered = tomogram_wedge_mask_centered(
            crop_size, params['min_tilt_angle'], params['max_tilt_angle'])

    skipped_border = 0
    skipped_missing = 0
    voxel_size = None

    by_tomo = {}
    for row in rows:
        by_tomo.setdefault(row['tomo'], []).append(row)

    for tomo_name, tomo_rows in by_tomo.items():
        path = tomo_path(params['data_folder'], tomo_name, params['file_extension'])
        print(f'  {tomo_name}: {len(tomo_rows)} particle(s)')

        with mrcfile.mmap(path, permissive=True, mode='r') as mrc:
            volume = mrc.data
            if voxel_size is None:
                try:
                    voxel_size = mrc.voxel_size.copy()
                except Exception:
                    voxel_size = None

            fill_value = (0.0 if params['skip_border_particles']
                          else float(np.median(volume)))
            for row in tomo_rows:
                center = np.array([row['z'], row['y'], row['x']], dtype=np.float64)
                R = rotation_from_row(row, use_orientations=use_orientations)

                if params['skip_border_particles'] and not particle_fits(
                        volume.shape, center, crop_size, R):
                    skipped_border += 1
                    continue

                try:
                    particle = aligned_subtomogram(
                        volume, center, crop_size, R,
                        fill_value=fill_value, order=1)
                except Exception:
                    skipped_missing += 1
                    continue

                if params['normalize_particles']:
                    particle = normalize_particle(particle, mask)
                if params['apply_soft_mask']:
                    particle = particle * mask

                if params['average_mode'] == 'real_space':
                    real_sum += particle.astype(np.float64, copy=False)
                    real_count += 1
                else:
                    aligned_wedge_centered = rotate_centered_cube(
                        base_wedge_centered, R, order=1)
                    aligned_wedge_centered = np.clip(
                        aligned_wedge_centered, 0.0, 1.0)
                    aligned_wedge = np.fft.ifftshift(aligned_wedge_centered)
                    particle_fft = np.fft.fftn(particle)
                    fourier_sum += particle_fft * aligned_wedge
                    fourier_weight += aligned_wedge
                    real_count += 1

    if real_count == 0:
        raise RuntimeError('No particles could be averaged.')

    if params['average_mode'] == 'real_space':
        average = (real_sum / float(real_count)).astype(np.float32)
    else:
        valid = fourier_weight >= float(params['wedge_weight_floor'])
        average_fft = np.zeros(crop_size, dtype=np.complex128)
        average_fft[valid] = fourier_sum[valid] / fourier_weight[valid]
        average = np.fft.ifftn(average_fft).real.astype(np.float32)

    mrc_path = params['output_folder'] / f'{output_prefix}.mrc'
    png_path = params['output_folder'] / f'{output_prefix}_mid_z.png'
    write_mrc(mrc_path, average, voxel_size=voxel_size)
    save_middle_slice_png(png_path, average)

    return {
        'average': average,
        'n_input': len(rows),
        'n_averaged': real_count,
        'n_border_skipped': skipped_border,
        'n_failed': skipped_missing,
        'mrc_path': mrc_path,
        'png_path': png_path,
    }


def average_group_gpu(rows, params, output_prefix, use_orientations=True):
    device = params['device']
    crop_size = params['crop_size']
    batch_size = int(params['batch_size'])
    input_shape = fast_input_cube_shape(crop_size)

    mask_np = spherical_mask(
        crop_size,
        radius_fraction=params['mask_radius_fraction'],
        edge_fraction=params['mask_edge_fraction'])
    mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
    offsets = torch_output_offsets_zyx(crop_size, device)

    real_sum = torch.zeros(crop_size, device=device, dtype=torch.float32)
    fourier_sum = torch.zeros(crop_size, device=device, dtype=torch.complex64)
    fourier_weight = torch.zeros(crop_size, device=device, dtype=torch.float32)

    base_wedge = None
    if params['average_mode'] == 'wedge_weighted':
        base_wedge_np = tomogram_wedge_mask_centered(
            crop_size, params['min_tilt_angle'], params['max_tilt_angle'])
        base_wedge = torch.from_numpy(base_wedge_np).to(
            device=device, dtype=torch.float32)[None, None]

    skipped_border = 0
    skipped_missing = 0
    n_averaged = 0
    voxel_size = None

    by_tomo = {}
    for row in rows:
        by_tomo.setdefault(row['tomo'], []).append(row)

    for tomo_name, tomo_rows in by_tomo.items():
        path = tomo_path(params['data_folder'], tomo_name, params['file_extension'])
        print(f'  {tomo_name}: {len(tomo_rows)} particle(s)')

        with mrcfile.mmap(path, permissive=True, mode='r') as mrc:
            volume = mrc.data
            if voxel_size is None:
                try:
                    voxel_size = mrc.voxel_size.copy()
                except Exception:
                    voxel_size = None

            fill_value = (0.0 if params['skip_border_particles']
                          else float(np.median(volume)))
            valid_items = []
            for row in tomo_rows:
                center = np.array([row['z'], row['y'], row['x']], dtype=np.float64)
                R = rotation_from_row(row, use_orientations=use_orientations)
                if params['skip_border_particles'] and not particle_fits(
                        volume.shape, center, crop_size, R):
                    skipped_border += 1
                    continue
                valid_items.append((center, R))

            for start in range(0, len(valid_items), batch_size):
                items = valid_items[start:start + batch_size]
                B = len(items)
                if B == 0:
                    continue

                patches = np.empty((B, *input_shape), dtype=np.float32)
                source_centers = np.empty((B, 3), dtype=np.float32)
                rotations = np.empty((B, 3, 3), dtype=np.float32)
                good = np.ones(B, dtype=bool)

                for i, (center, R) in enumerate(items):
                    try:
                        source_centers[i] = extract_fixed_patch_into(
                            patches[i], volume, center, fill_value=fill_value)
                        rotations[i] = np.asarray(R, dtype=np.float32)
                    except Exception:
                        good[i] = False
                        skipped_missing += 1

                if not np.all(good):
                    patches = patches[good]
                    source_centers = source_centers[good]
                    rotations = rotations[good]
                    B = len(rotations)
                    if B == 0:
                        continue

                particles = torch_align_particle_batch(
                    patches, source_centers, rotations,
                    crop_size, offsets, device)
                particles = torch_normalize_particles(
                    particles, mask,
                    normalize=params['normalize_particles'],
                    apply_mask=params['apply_soft_mask'])

                if params['average_mode'] == 'real_space':
                    real_sum.add_(particles.sum(dim=0))
                else:
                    wedge_centered = torch_rotate_wedge_batch(
                        base_wedge, rotations, crop_size, offsets, device)
                    wedge = torch.fft.ifftshift(
                        wedge_centered, dim=(-3, -2, -1))
                    particle_fft = torch.fft.fftn(
                        particles, dim=(-3, -2, -1))
                    fourier_sum.add_((particle_fft * wedge).sum(dim=0))
                    fourier_weight.add_(wedge.sum(dim=0))

                n_averaged += B

    if n_averaged == 0:
        raise RuntimeError('No particles could be averaged.')

    if params['average_mode'] == 'real_space':
        average = (real_sum / float(n_averaged)).detach().cpu().numpy().astype(np.float32)
    else:
        valid = fourier_weight >= float(params['wedge_weight_floor'])
        average_fft = torch.zeros_like(fourier_sum)
        average_fft[valid] = fourier_sum[valid] / fourier_weight[valid]
        average = torch.fft.ifftn(
            average_fft, dim=(-3, -2, -1)).real.detach().cpu().numpy().astype(np.float32)

    mrc_path = params['output_folder'] / f'{output_prefix}.mrc'
    png_path = params['output_folder'] / f'{output_prefix}_mid_z.png'
    write_mrc(mrc_path, average, voxel_size=voxel_size)
    save_middle_slice_png(png_path, average)

    return {
        'average': average,
        'n_input': len(rows),
        'n_averaged': n_averaged,
        'n_border_skipped': skipped_border,
        'n_failed': skipped_missing,
        'mrc_path': mrc_path,
        'png_path': png_path,
    }


def average_group(rows, params, output_prefix, use_orientations=True):
    if params['device'].type == 'cuda':
        return average_group_gpu(rows, params, output_prefix, use_orientations=use_orientations)
    return average_group_cpu(rows, params, output_prefix, use_orientations=use_orientations)


def main(config_file):
    params = load_params(config_file)
    params['output_folder'].mkdir(parents=True, exist_ok=True)

    rows = read_star(params['star_file'])
    rows = select_rows(rows, params)
    groups = group_rows(rows, group_by_class=params['group_by_class'])

    print('Aligned subtomogram averaging:')
    print(f"  STAR: {params['star_file']}")
    print(f"  data folder: {params['data_folder']}")
    print(f"  crop size: {params['crop_size']}")
    print(f"  mode: {params['average_mode']}")
    print(f"  device: {params['device']}")
    if params['device'].type == 'cuda':
        print(f"  GPU batch size: {params['batch_size']}")
    if params['average_mode'] == 'wedge_weighted':
        print(f"  tilt range: [{params['min_tilt_angle']:.1f}, "
              f"{params['max_tilt_angle']:.1f}] deg")
        print('  tilt axis XYZ: [0, 1, 0] (fixed)')
    print(f"  particles: {len(rows)}")
    print(f"  output: {params['output_folder']}")

    for group_name, group in groups.items():
        if group_name == 'all':
            prefix_root = f'average_{params["average_mode"]}'
        else:
            safe = group_name.replace('/', '_').replace(' ', '_')
            prefix_root = f'{safe}_average_{params["average_mode"]}'

        print(f'\nAverage group: {group_name} ({len(group)} particle(s))')

        if params['save_angular_distribution']:
            ang_path = params['output_folder'] / f'{prefix_root}_angular_distribution.png'
            out = save_angular_distribution_plot(group, ang_path, title=f'{group_name} angular distribution')
            if out is not None:
                print(f"  angular distribution: {out}")

        if params['save_no_rotation_average']:
            no_rot_prefix = f'{prefix_root}_no_rotation'
            result = average_group(group, params, no_rot_prefix, use_orientations=False)
            print('  [no rotation]')
            print(f"    averaged: {result['n_averaged']}/{result['n_input']}")
            print(f"    skipped at borders: {result['n_border_skipped']}")
            print(f"    failed: {result['n_failed']}")
            print(f"    MRC: {result['mrc_path']}")
            print(f"    PNG: {result['png_path']}")

        if params['save_oriented_average']:
            oriented_prefix = f'{prefix_root}_oriented'
            result = average_group(group, params, oriented_prefix, use_orientations=True)
            print('  [oriented]')
            print(f"    averaged: {result['n_averaged']}/{result['n_input']}")
            print(f"    skipped at borders: {result['n_border_skipped']}")
            print(f"    failed: {result['n_failed']}")
            print(f"    MRC: {result['mrc_path']}")
            print(f"    PNG: {result['png_path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create an aligned subtomogram average from RELION STAR orientations.')
    parser.add_argument('--config_file', required=True, help='CryoSiam YAML config file')
    args = parser.parse_args()
    main(args.config_file)