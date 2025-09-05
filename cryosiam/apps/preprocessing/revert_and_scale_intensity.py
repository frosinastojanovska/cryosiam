import os
import mrcfile
import argparse
import numpy as np


def invert_tomogram(tomo):
    return tomo * -1


def scale_tomogram(tomo, percentile_lower=None, percentile_upper=None):
    if percentile_lower:
        min_val = np.percentile(tomo, percentile_lower)
    else:
        min_val = tomo.min()

    if percentile_upper:
        max_val = np.percentile(tomo, percentile_upper)
    else:
        max_val = tomo.max()

    tomo = (tomo - min_val) / (max_val - min_val)

    return np.clip(tomo, 0, 1)

def main(input_path, output_path, invert, lower_end_percentage, upper_end_percentage):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for tomo in os.listdir(input_path):
            if tomo.endswith(".mrc") or tomo.endswith(".rec"):
                with mrcfile.open(os.path.join(input_path, tomo), permissive=True) as m:
                    tomogram = m.data
                    voxel_size = m.voxel_size
                if invert:
                    tomogram = invert_tomogram(tomogram)

                tomogram = scale_tomogram(tomogram, lower_end_percentage, upper_end_percentage)

                with mrcfile.new(os.path.join(output_path, tomo), overwrite=True) as m:
                    m.set_data(tomogram)
                    m.voxel_size = voxel_size
    else:
        with mrcfile.open(input_path, permissive=True) as m:
            tomogram = m.data
            voxel_size = m.voxel_size

        if invert:
            tomogram = invert_tomogram(tomogram)

        tomogram = scale_tomogram(tomogram, lower_end_percentage, upper_end_percentage)

        with mrcfile.new(output_path, overwrite=True) as m:
            m.set_data(tomogram)
            m.voxel_size = voxel_size


def parser_helper(description=None):
    description = "Invert contrast of a tomogram and perform scaling of the values" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str, required=True, help='path to the input tomogram or '
                                                                      'path to the folder with input tomogram/s')
    parser.add_argument('--output_path', type=str, required=True, help='path to save the output tomogram or '
                                                                       'path to folder to save the output tomogram/s')
    parser.add_argument("--invert", action="store_true", default=False, help="Inverts contrast of images.")
    parser.add_argument("--lower_end_percentage", type=float, required=True, default=0.1,
                        help="Cut off values from the lower percentile end of the intensities.")
    parser.add_argument("--upper_end_percentage", type=float, required=True, default=99.9,
                        help="Cut off values from the upper percentile end of the intensities.")
    return parser


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.invert, args.lower_end_percentage, args.upper_end_percentage)
