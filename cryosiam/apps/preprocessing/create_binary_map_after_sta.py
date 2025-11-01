import os
import mrcfile
import starfile
import argparse
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R

from cryosiam.data import MrcReader, MrcWriter


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
    voxel_size = tomogram.voxel_size
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

            rotated_ref = rotate_reference(reference_map, rot, tilt, psi)

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

        output = output.astype(np.uint8)

        writer = MrcWriter(output_dtype=np.uint8, overwrite=True)
        writer.set_metadata({'voxel_size': voxel_size})
        writer.set_data_array(output, channel_dim=None)
        writer.write(os.path.join(output_dir, f'{t_name}.mrc'))


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
