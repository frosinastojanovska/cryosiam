import os
import csv
import h5py
import starfile
import argparse
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops_table
from scipy.spatial.transform import Rotation as R

from cryosiam.data import MrcReader, MrcWriter


def find_optimal_int_dtype(data):
    """Determines the smallest NumPy integer dtype for the given data."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.size == 0:
        return np.dtype('int8')

    min_val = data.min()
    max_val = data.max()

    # Ordered list of dtypes from smallest to largest
    dtypes_to_check = [np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32'), np.dtype('uint64')]

    for dtype in dtypes_to_check:
        info = np.iinfo(dtype)
        if min_val >= info.min and max_val <= info.max:
            return dtype

    return None


def rotate_reference(ref, rot, tilt, psi):
    rotation_matrix = R.from_euler("ZYZ", [rot, tilt, psi], degrees=True).as_matrix()

    dimensions = ref.shape  # (Z, Y, X)

    # Create centered coordinate grid
    z, y, x = np.meshgrid(np.arange(dimensions[0]), np.arange(dimensions[1]), np.arange(dimensions[2]), indexing='ij')
    coords = np.vstack([x.ravel() - dimensions[2] / 2, y.ravel() - dimensions[1] / 2,
                        z.ravel() - dimensions[0] / 2, ])  # Shape: (3, N), in (X, Y, Z)

    # Rotate coordinates
    rotated_coords = rotation_matrix @ coords

    # Shift back to image coordinates
    x_new = rotated_coords[0, :] + dimensions[2] / 2
    y_new = rotated_coords[1, :] + dimensions[1] / 2
    z_new = rotated_coords[2, :] + dimensions[0] / 2

    # Reshape to 3D coordinate arrays (must match original shape for map_coordinates)
    coords_interp = [z_new.reshape(dimensions), y_new.reshape(dimensions), x_new.reshape(dimensions), ]

    # Interpolate rotated volume
    new_ref = map_coordinates(ref, coords_interp, order=1)

    return new_ref


def read_star_file(star_file):
    data = starfile.read(star_file)
    if type(data) == dict:
        data = data['particles']
    return data


def main(star_file, map_file, output_dir, map_threshold, example_tomogram, tomo_name):
    os.makedirs(output_dir, exist_ok=True)
    data = read_star_file(star_file)
    if 'rlnTomoName' in data.columns:
        header_name = 'rlnTomoName'
    else:
        header_name = 'rlnMicrographName'

    if tomo_name is not None:
        data = data[data[header_name] == tomo_name]

    reader = MrcReader(read_in_mem=True)
    tomogram = reader.read(example_tomogram)
    tomogram = tomogram.data
    tomogram.setflags(write=True)

    reader = MrcReader(read_in_mem=True)
    reference_map = reader.read(map_file)
    reference_map = reference_map.data
    reference_map.setflags(write=True)
    reference_map = (reference_map > map_threshold).astype(np.uint8)

    radius = reference_map.shape[0] // 2
    size = tomogram.shape
    z_dim, y_dim, x_dim = size

    for t_name in np.unique(data[header_name]):
        print(f'Processing tomo: {t_name}')
        current_data = data[data[header_name] == t_name]
        print(f'Placing {current_data.shape[0]} instances')
        output = np.zeros(size)
        for i, row in current_data.iterrows():
            x = int(row['rlnCoordinateX'])
            y = int(row['rlnCoordinateY'])
            z = int(row['rlnCoordinateZ'])
            rot = row['rlnAngleRot']
            tilt = row['rlnAngleTilt']
            psi = row['rlnAnglePsi']

            rotated_ref = rotate_reference(reference_map, rot, tilt, psi) * (i + 1)

            rotated_ref = rotated_ref[
                          max(radius - z, 0): (
                              radius + z_dim - z if z_dim - z <= radius else radius * 2
                          ),
                          max(radius - y, 0): (
                              radius + y_dim - y if y_dim - y <= radius else radius * 2
                          ),
                          max(radius - x, 0): (
                              radius + x_dim - x if x_dim - x <= radius else radius * 2
                          ), ]

            output[
            max(0, z - radius): min(z + radius, z_dim),
            max(0, y - radius): min(y + radius, y_dim),
            max(0, x - radius): min(x + radius, x_dim),
            ][rotated_ref > 0] = rotated_ref[rotated_ref > 0]

        optimal_dtype = find_optimal_int_dtype(output)
        if optimal_dtype:
            output = output.astype(optimal_dtype)

        with h5py.File(os.path.join(output_dir, f'{t_name}_instance_preds.h5'), 'w') as f:
            f.create_dataset('instances', data=output)

        regions = regionprops_table(output, properties=['label', 'area', 'bbox', 'centroid'])
        regions_file = os.path.join(output_dir, f'{t_name}_instance_regions.csv')
        # TODO: Remove the following two lines
        regions['second_label'] = current_data['rlnLabel'].values.tolist()
        regions['cc_score'] = current_data['rlnCC'].values.tolist()
        with open(regions_file, 'w') as f:
            w = csv.writer(f)
            w.writerow(list(regions.keys()))
            for i in range(regions['label'].shape[0]):
                w.writerow([regions[l][i] for l in regions.keys()])


def parser_helper(description=None):
    description = "Create instance map from given average map and orientations" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--star_file', type=str, required=True, help='path to star file with orientations after STA')
    parser.add_argument('--map_file', type=str, required=True,
                        help='path to the map file that will be placed in 3D tomogram (it is expected to be cubic subvolume)')
    parser.add_argument('--output_dir', type=str, required=True, help='path to folder to save the output tomogram/s')
    parser.add_argument("--map_threshold", type=float, required=True,
                        help="Threshold for the map to binarize it.")
    parser.add_argument('--example_tomogram', type=str, required=True,
                        help='path to one tomogram to determine the 3D size of the output')
    parser.add_argument('--tomo_name', type=str, required=False,
                        help='process only this tomogram, the name should match the rlnMicrographName')
    return parser


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.star_file, args.map_file, args.output_dir, args.map_threshold, args.example_tomogram, args.tomo_name)
