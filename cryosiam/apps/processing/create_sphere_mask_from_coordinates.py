import os
import starfile
import argparse
import numpy as np
import pandas as pd

from cryosiam.data import MrcReader, MrcWriter


def create_sphere(radius):
    N = radius * 4
    x0, y0, z0 = (N // 2, N // 2, N // 2)

    x, y, z = np.mgrid[0:N:1, 0:N:1, 0:N:1]
    r = - np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    r[r < -1 * radius] = np.min(r)
    r = r.astype(np.float32) - np.min(r)
    r = (r - r.max()) / (r.max() - r.min())
    return r


def read_csv_file(filename):
    data = pd.read_csv(filename)
    return data


def read_star_file(star_file):
    data = starfile.read(star_file)
    if type(data) == dict:
        data = data['particles']
    return data[['rlnCoordinateZ', 'rlnCoordinateY', 'rlnCoordinateX']]


def main(coordinates_file, sphere_radius, output_dir, tomogram_path, tomo_name):
    os.makedirs(output_dir, exist_ok=True)
    if coordinates_file.endswith('.mrc'):
        data = read_star_file(coordinates_file)
        file_type = 0
    else:
        data = read_csv_file(coordinates_file)
        file_type = 1

    if 'rlnTomoName' in data.columns:
        header_name = 'rlnTomoName'
    elif 'rlnMicrographName' in data.columns:
        header_name = 'rlnMicrographName'
    else:
        header_name = 'tomo'

    if tomo_name is not None:
        data = data[data[header_name] == tomo_name]

    reader = MrcReader(read_in_mem=True)
    if not os.path.isdir(tomogram_path):
        tomogram = reader.read(tomogram_path)
        voxel_size = tomogram.voxel_size
        tomogram = tomogram.data
        tomogram.setflags(write=True)

    sphere = (create_sphere(radius=sphere_radius) > -0.5).astype(np.uint8)

    radius = sphere.shape[0] // 2

    for t_name in np.unique(data[header_name]):
        print(f'Processing tomo: {t_name}')
        if os.path.isdir(tomogram_path):
            tomogram = reader.read(os.path.join(tomogram_path, t_name + '.mrc'))
            voxel_size = tomogram.voxel_size
            tomogram = tomogram.data
            tomogram.setflags(write=True)
        size = tomogram.shape
        z_dim, y_dim, x_dim = size
        current_data = data[data[header_name] == t_name]
        print(f'Placing {current_data.shape[0]} instances')
        output = np.zeros(size)
        for i, row in current_data.iterrows():
            if file_type == 0:
                x = int(row['rlnCoordinateX'])
                y = int(row['rlnCoordinateY'])
                z = int(row['rlnCoordinateZ'])
            else:
                x = int(row['centroid-2'])
                y = int(row['centroid-1'])
                z = int(row['centroid-0'])
            rotated_ref = sphere[
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
    description = "Create binary tomogram mask with sphere map from given center coordinates" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--coordinates_file', type=str, required=True,
                        help='path to star file or csv file with X,Y,Z coordinates. The star file needs the '
                             '[rlnCoordinateX, rlnCoordinateY, rlnCoordinateZ] header, while the csv file header '
                             'should be [centroid-0, centroid-1, centroid-2] for z, y, x order')
    parser.add_argument('--sphere_radius', type=int, required=True,
                        help='radius in number of pixels for the sphere')
    parser.add_argument('--output_dir', type=str, required=True, help='path to folder to save the output tomogram/s')
    parser.add_argument('--tomogram_path', type=str, required=True,
                        help='path to one tomogram to determine the 3D size of the output')
    parser.add_argument('--tomo_name', type=str, required=False,
                        help='process only this tomogram, the name should match the rlnMicrographName in '
                             'the starfile or the tomo in the csv file')
    return parser


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.coordinates_file, args.sphere_radius, args.output_dir, args.tomogram_path, args.tomo_name)
