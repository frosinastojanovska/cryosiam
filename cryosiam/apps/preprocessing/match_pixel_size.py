import os
import mrcfile
import argparse
import numpy as np

from cryosiam.utils import match_pixel_size

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


def parser_helper(description=None):
    description = "Match the pixel size of a tomogram with a desired pixel size" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str, required=True, help='path to the input tomogram or '
                                                                      'path to the folder with input tomogram/s')
    parser.add_argument('--output_path', type=str, required=True, help='path to save the output tomogram or '
                                                                       'path to folder to save the output tomogram/s')
    parser.add_argument("--pixel_size_in", type=float, required=True, help="Pixel size (angstroms) of the input.")
    parser.add_argument("--pixel_size_out", type=float, required=True, help="Pixel size (angstroms) of the output.")
    parser.add_argument("--disable_smooth", action="store_true", default=True, help="Disable smoothing of the output.")
    return parser


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        os.makedirs(args.output_path, exist_ok=True)
        for tomo in os.listdir(args.input_path):
            if tomo.endswith(".mrc") or tomo.endswith(".rec"):
                with mrcfile.open(os.path.join(args.input_path, tomo), permissive=True) as m:
                    tomogram = m.data
                    voxel_size = m.voxel_size
                if args.pixel_size_in != args.pixel_size_out:
                    tomogram = match_pixel_size(tomogram, args.pixel_size_in, args.pixel_size_out, args.disable_smooth)

                with mrcfile.new(os.path.join(args.output_path, tomo), overwrite=True) as m:
                    m.set_data(tomogram.astype(np.float32))
                    m.voxel_size = args.pixel_size_out
    else:
        with mrcfile.open(args.input_path, permissive=True) as m:
            tomogram = m.data
            voxel_size = m.voxel_size

        if args.pixel_size_in != args.pixel_size_out:
            tomogram = match_pixel_size(tomogram, args.pixel_size_in, args.pixel_size_out, args.disable_smooth)

        with mrcfile.new(args.output_path, overwrite=True) as m:
            m.set_data(tomogram.astype(np.float32))
            m.voxel_size = args.pixel_size_out
