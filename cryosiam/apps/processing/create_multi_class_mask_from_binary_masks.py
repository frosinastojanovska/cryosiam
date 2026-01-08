import os
import argparse
import numpy as np

from cryosiam.data import MrcReader, MrcWriter


def main(root_binary_masks_folder, output_dir, tomo_name):
    os.makedirs(output_dir, exist_ok=True)

    folders = [x for x in os.listdir(root_binary_masks_folder)
               if os.path.isdir(os.path.join(root_binary_masks_folder, x))]

    if len(folders) == 0:
        print('No folders found')
        return

    tomos = os.path.listdir(os.path.join(root_binary_masks_folder, folders[0]))

    if tomo_name is not None:
        tomos = [tomo_name]

    reader = MrcReader(read_in_mem=True)
    writer = MrcWriter(output_dtype=np.uint16, overwrite=True)

    for tomo in tomos:
        multi_class = None
        for i, folder in enumerate(folders):
            file = os.path.join(root_binary_masks_folder, folder, tomo)
            tomogram = reader.read(file)
            voxel_size = tomogram.voxel_size
            tomogram = tomogram.data
            tomogram.setflags(write=True)
            if multi_class is None:
                multi_class = np.zeros(tomo.shape, dtype=np.uint16)
            multi_class[tomogram > 0] = (tomogram[tomogram > 0] > 0) * (i + 1)

        multi_class = multi_class.astype(np.uint16)
        writer.set_metadata({'voxel_size': voxel_size})
        writer.set_data_array(multi_class, channel_dim=None)
        writer.write(os.path.join(output_dir, f'{tomo}.mrc'))


def parser_helper(description=None):
    description = "Create multi-class masks from folders of binary masks" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_binary_masks_folder', type=str, required=True,
                        help='path to a folder that contains subfolders for every binary label. The subfolders contain the binary masks for every separate label')
    parser.add_argument('--output_dir', type=str, required=True, help='path to folder to save the output tomogram/s')
    parser.add_argument('--tomo_name', type=str, required=False,
                        help='process only this tomogram (include the file extension in the name)')
    return parser


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.star_file, args.output_dir, args.tomo_name)
