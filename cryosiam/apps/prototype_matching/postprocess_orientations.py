import os
import yaml
import h5py
import torch
import starfile
import mrcfile
import numpy as np
import pandas as pd
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from cryosiam.utils import parser_helper


def get_uniform_rotations(angular_sampling_deg: float) -> np.ndarray:
    """
    Approximately uniform SO3 sampling via ZYZ Euler angle grid.
    Tilt sampling is weighted by sin(tilt) for uniform sphere coverage.
    Does not require healpy.

    Args:
        angular_sampling_deg: angular step in degrees (e.g. 7.0)

    Returns:
        (N, 3, 3) rotation matrices, float32
    """
    step = angular_sampling_deg
    rotations = []

    for tilt in np.arange(0, 181, step):
        tilt_rad = np.radians(tilt)
        n_rot = (1 if tilt in (0., 180.)
                 else max(1, int(360 * np.sin(tilt_rad) / step)))
        rot_angles = np.linspace(0, 360, n_rot, endpoint=False)
        psi_angles = np.arange(0, 360, step)

        for rot in rot_angles:
            for psi in psi_angles:
                R = Rotation.from_euler('ZYZ', [rot, tilt, psi], degrees=True).as_matrix()
                rotations.append(R)

    rotations = np.array(rotations, dtype=np.float32)
    print(f'Rotation sampling: {angular_sampling_deg}° → '
          f'{len(rotations)} rotations')
    return rotations


def compute_fourier_masks(patch_size: int,
                          tilt_angle: float,
                          freq_low: float,
                          freq_high: float,
                          device: str) -> torch.Tensor:
    """
    Combined missing-wedge + bandpass mask in Fourier space.
    Returns (D, H, W//2+1) float tensor on device.
    Applied symmetrically to both reference and patch FFTs.
    """
    D = H = W = patch_size
    fz = torch.fft.fftfreq(D).to(device)
    fy = torch.fft.fftfreq(H).to(device)
    fx = torch.fft.rfftfreq(W).to(device)
    FZ, FY, FX = torch.meshgrid(fz, fy, fx, indexing='ij')

    f_xy = torch.sqrt(FX ** 2 + FY ** 2)
    wedge_limit = f_xy * np.tan(np.radians(tilt_angle))
    wedge_mask = (FZ.abs() <= wedge_limit).float()

    freq = torch.sqrt(FX ** 2 + FY ** 2 + FZ ** 2)
    bandpass = ((freq > freq_low) & (freq < freq_high)).float()

    return (wedge_mask * bandpass).to(device)


def load_reference(reference_path: str,
                   patch_size: int,
                   low_pass_sigma: float,
                   device: str) -> torch.Tensor:
    """
    Load, smooth, normalise and resize reference map.
    Returns (1, 1, D, H, W) float tensor on device.
    """
    with mrcfile.open(reference_path, permissive=True) as m:
        ref = m.data.copy().astype(np.float32)

    ref = gaussian_filter(ref, sigma=low_pass_sigma)
    ref = (ref - ref.mean()) / (ref.std() + 1e-8)

    ref_t = torch.from_numpy(ref).float().to(device)
    ref_t = F.interpolate(ref_t.unsqueeze(0).unsqueeze(0),
                          size=(patch_size,) * 3,
                          mode='trilinear', align_corners=False)
    return ref_t  # (1, 1, D, H, W)


def rotate_volume_batch(volume_t: torch.Tensor,
                        rotations_t: torch.Tensor) -> torch.Tensor:
    """
    Rotate (1,1,D,H,W) volume by (N,3,3) rotation matrices on GPU.
    Returns (N, D, H, W).
    """
    N = rotations_t.shape[0]
    D, H, W = volume_t.shape[2:]
    device = volume_t.device

    grid = F.affine_grid(
        torch.eye(3, 4, device=device).unsqueeze(0).expand(N, -1, -1),
        (N, 1, D, H, W), align_corners=True)
    coords_flat = grid.view(N, -1, 3)
    rot_coords = torch.bmm(coords_flat,
                           rotations_t.transpose(1, 2)).view(N, D, H, W, 3)

    return F.grid_sample(volume_t.expand(N, -1, -1, -1, -1),
                         rot_coords, mode='bilinear',
                         padding_mode='zeros',
                         align_corners=True).squeeze(1)  # (N, D, H, W)


def precompute_ref_batch(ref_t: torch.Tensor,
                         rotations_batch: np.ndarray,
                         combined_mask: torch.Tensor) -> tuple:
    """
    Precompute masked FFTs for a batch of rotations.
    Returns:
        ref_f_masked: (B, D, H, W//2+1) complex on GPU
        ref_power:    (B,) float on GPU
    """
    rot_t = torch.from_numpy(rotations_batch).to(ref_t.device)
    rotated = rotate_volume_batch(ref_t, rot_t)  # (B, D, H, W)
    ref_f = torch.fft.rfftn(rotated, dim=(1, 2, 3))
    ref_f_masked = ref_f * combined_mask.unsqueeze(0)
    ref_power = (ref_f_masked.abs() ** 2).sum(dim=(1, 2, 3))
    return ref_f_masked, ref_power


def fourier_cc_batch(patch_masked_t: torch.Tensor,
                     combined_mask: torch.Tensor,
                     ref_f_masked: torch.Tensor,
                     ref_power: torch.Tensor) -> torch.Tensor:
    """
    Normalised Fourier CC — symmetric missing wedge.

    patch_masked_t: (D, H, W) real-space patch, already spatially masked
    combined_mask:  (D, H, W//2+1) Fourier mask — applied to patch too
    ref_f_masked:   (B, D, H, W//2+1) complex — pre-masked reference FFTs
    ref_power:      (B,) float

    Returns: (B,) CC scores
    """
    patch_f = torch.fft.rfftn(patch_masked_t)
    patch_f_masked = patch_f * combined_mask  # symmetric mask on patch
    patch_power = (patch_f_masked.abs() ** 2).sum()

    cc = (torch.conj(patch_f_masked).unsqueeze(0) * ref_f_masked).real
    scores = cc.sum(dim=(1, 2, 3))
    return scores / (torch.sqrt(patch_power * ref_power) + 1e-8)


def estimate_orientations(binary_volume: np.ndarray,
                          peaks_df: pd.DataFrame,
                          ref_t: torch.Tensor,
                          rotations: np.ndarray,
                          combined_mask: torch.Tensor,
                          patch_size: int,
                          binary_threshold: float,
                          rotation_batch_size: int,
                          device: str) -> pd.DataFrame:
    """
    GPU Fourier template matching around each detected center.

    Processes rotations in batches to avoid GPU OOM.
    Binary mask suppresses background — only foreground contributes to CC.

    Args:
        binary_volume:       (D, H, W) float32 binary prediction [0,1]
        peaks_df:            DataFrame with z/y/x columns
        ref_t:               (1,1,D,H,W) preprocessed reference on GPU
        rotations:           (N, 3, 3) rotation matrices
        combined_mask:       (D, H, W//2+1) Fourier mask on GPU
        patch_size:          extraction patch size
        binary_threshold:    threshold for spatial mask
        rotation_batch_size: rotations processed per GPU batch
        device:              torch device string

    Returns:
        peaks_df with rot/tilt/psi/cc_score columns appended
    """
    half = patch_size // 2
    N_rot = len(rotations)
    N_peaks = len(peaks_df)

    # accumulators on GPU
    best_scores = torch.full((N_peaks,), -float('inf'), device=device)
    best_rot_idx = torch.zeros(N_peaks, dtype=torch.long, device=device)

    print(f'  {N_peaks} particles × {N_rot} rotations '
          f'(batch={rotation_batch_size})')

    with torch.no_grad():
        for rot_start in range(0, N_rot, rotation_batch_size):
            rot_end = min(rot_start + rotation_batch_size, N_rot)
            rot_batch = rotations[rot_start:rot_end]

            # precompute masked FFTs for this rotation batch — GPU
            ref_f_masked, ref_power = precompute_ref_batch(
                ref_t, rot_batch, combined_mask)

            # score each particle against this rotation batch
            for i, (_, row) in enumerate(peaks_df.iterrows()):
                z, y, x = int(row['z']), int(row['y']), int(row['x'])

                z0 = max(0, z - half)
                z1 = min(binary_volume.shape[0], z + half)
                y0 = max(0, y - half)
                y1 = min(binary_volume.shape[1], y + half)
                x0 = max(0, x - half)
                x1 = min(binary_volume.shape[2], x + half)

                patch = binary_volume[z0:z1, y0:y1, x0:x1]
                if patch.shape != (patch_size, patch_size, patch_size):
                    continue  # edge particle — skip

                patch_t = torch.from_numpy(patch).float().to(device)

                # spatial mask from binary — suppresses background
                spatial_mask = (patch_t > binary_threshold).float()
                patch_norm = ((patch_t - patch_t.mean()) /
                              (patch_t.std() + 1e-8))
                patch_masked = patch_norm * spatial_mask

                # Fourier CC — symmetric mask on both patch and ref
                scores = fourier_cc_batch(patch_masked, combined_mask,
                                          ref_f_masked, ref_power)

                # update best score for this particle
                batch_best_val, batch_best_loc = scores.max(dim=0)
                if batch_best_val > best_scores[i]:
                    best_scores[i] = batch_best_val
                    best_rot_idx[i] = rot_start + batch_best_loc

            print(f'  Rotation batch {rot_start}–{rot_end}/{N_rot} done', end='\r')

    print()

    # reconstruct Euler angles from best rotation indices
    best_scores_np = best_scores.cpu().numpy()
    best_rot_idx_np = best_rot_idx.cpu().numpy()

    results = []
    for i in range(N_peaks):
        idx = int(best_rot_idx_np[i])
        score = float(best_scores_np[i])
        R = rotations[idx]
        euler = Rotation.from_matrix(R).as_euler('ZYZ', degrees=True)
        results.append({'rot': float(euler[0]),
                        'tilt': float(euler[1]),
                        'psi': float(euler[2]),
                        'cc_score': score})

    return pd.concat([peaks_df.reset_index(drop=True),
                      pd.DataFrame(results).reset_index(drop=True)], axis=1)


def main(config_file_path, filename=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    # --- folders ---
    centers_folder = cfg['centers_folder']
    prediction_folder = cfg['prediction_folder']
    output_folder = cfg.get('output_folder', centers_folder)
    os.makedirs(output_folder, exist_ok=True)

    reference_path = cfg['reference_file']
    file_ext = cfg['file_extension']
    query_name = cfg.get('query_name',
                         os.path.basename(reference_path).split(file_ext)[0])

    # --- orientation settings ---
    patch_size = int(cfg.get('orientation_patch_size', 64))
    low_pass_sigma = float(cfg.get('orientation_low_pass_sigma', 2.0))
    tilt_angle = float(cfg.get('tilt_angle', 60.0))
    freq_low = float(cfg.get('freq_low', 0.05))
    freq_high = float(cfg.get('freq_high', 0.4))
    binary_threshold = float(cfg.get('binary_threshold', 0.5))
    angular_sampling = float(cfg.get('angular_sampling', 7.0))
    rotation_batch_size = int(cfg.get('rotation_batch_size', 200))
    use_binary_mask = cfg.get('use_binary_mask', True)

    # --- precompute once on GPU ---
    rotations = get_uniform_rotations(angular_sampling)
    combined_mask = compute_fourier_masks(patch_size, tilt_angle,
                                          freq_low, freq_high, device)
    ref_t = load_reference(reference_path, patch_size,
                           low_pass_sigma, device)
    print(f'Reference loaded and resized to {patch_size}³')

    # --- collect center files ---
    if filename:
        tomo_root = os.path.basename(filename).split(file_ext)[0]
        center_files = [f'{tomo_root}_{query_name}_centers.csv']
    else:
        center_files = sorted([
            f for f in os.listdir(centers_folder)
            if f.endswith(f'_{query_name}_centers.csv')])

    print(f'Found {len(center_files)} center files')

    all_peaks = []
    rename_map = {
        'z': 'rlnCoordinateZ',
        'y': 'rlnCoordinateY',
        'x': 'rlnCoordinateX',
        'center_intensity': 'rlnCenterScore',
        'binary_score': 'rlnBinaryScore',
        'tomo': 'rlnMicrographName',
        'query': 'rlnQueryName',
        'rot': 'rlnAngleRot',
        'tilt': 'rlnAngleTilt',
        'psi': 'rlnAnglePsi',
        'cc_score': 'rlnCCScore'}

    for center_file in center_files:
        tomo_name = center_file.replace(f'_{query_name}_centers.csv', '')
        print(f'\nProcessing: {tomo_name}')

        centers_path = os.path.join(centers_folder, center_file)
        if not os.path.exists(centers_path):
            print(f'  WARNING: centers file not found, skipping')
            continue

        peaks = pd.read_csv(centers_path)
        print(f'  Loaded {len(peaks)} centers')
        if len(peaks) == 0:
            continue

        # load binary volume from h5 — kept on CPU, patches transferred to GPU
        binary_volume = None
        if use_binary_mask:
            pred_path = os.path.join(prediction_folder,
                                     f'{tomo_name}_{query_name}.h5')
            if os.path.exists(pred_path):
                with h5py.File(pred_path, 'r') as hf:
                    binary_volume = hf['binary'][()]  # (D, H, W) float32
                print(f'  Loaded binary mask: {binary_volume.shape}')
            else:
                print(f'  WARNING: no H5 found at '
                      f'{os.path.basename(pred_path)} — no mask applied')

        if binary_volume is None:
            # fallback: ones — no suppression, pure CC
            max_z = int(peaks['z'].max()) + patch_size
            max_y = int(peaks['y'].max()) + patch_size
            max_x = int(peaks['x'].max()) + patch_size
            binary_volume = np.ones((max_z, max_y, max_x), dtype=np.float32)

        # --- estimate orientations — batched over rotations, all on GPU ---
        peaks = estimate_orientations(
            binary_volume=binary_volume,
            peaks_df=peaks,
            ref_t=ref_t,
            rotations=rotations,
            combined_mask=combined_mask,
            patch_size=patch_size,
            binary_threshold=binary_threshold,
            rotation_batch_size=rotation_batch_size,
            device=device)

        print(f'  CC score range: '
              f'[{peaks["cc_score"].min():.4f}, '
              f'{peaks["cc_score"].max():.4f}]')

        # save per-tomogram CSV
        csv_out = os.path.join(output_folder,
                               f'{tomo_name}_{query_name}_oriented.csv')
        peaks.to_csv(csv_out, index=False)

        # save RELION-compatible STAR
        peaks_star = peaks.rename(columns=rename_map, errors='ignore')
        star_out = os.path.join(output_folder,
                                f'{tomo_name}_{query_name}_oriented.star')
        starfile.write(peaks_star, star_out, overwrite=True)
        print(f'  Saved: {os.path.basename(star_out)}')

        all_peaks.append(peaks)

    # --- combined output ---
    if all_peaks:
        combined = pd.concat(all_peaks, ignore_index=True)
        combined_star = combined.rename(columns=rename_map, errors='ignore')

        combined.to_csv(
            os.path.join(output_folder, f'{query_name}_all_oriented.csv'),
            index=False)
        starfile.write(
            combined_star,
            os.path.join(output_folder, f'{query_name}_all_oriented.star'),
            overwrite=True)
        print(f'\nDone. {len(combined)} total particles '
              f'from {len(all_peaks)} tomograms')

    print('Done.')


if __name__ == '__main__':
    parser = parser_helper('Orientation estimation via Fourier template matching')
    args = parser.parse_args()
    main(args.config_file, getattr(args, 'filename', None))
