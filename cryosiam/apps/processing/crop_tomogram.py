import os
import mrcfile
import argparse


def load_tomogram(file_path):
    with mrcfile.open(file_path, permissive=True) as m:
        return m.data.copy(), m.voxel_size


def save_tomogram(file_path, data, voxel_size=None):
    with mrcfile.new(file_path, overwrite=True) as m:
        m.set_data(data)
        if voxel_size is not None:
            m.voxel_size = voxel_size


def main(input_path, output_path, z1, z2, y1, y2, x1, x2):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for file_name in os.listdir(input_path):
            if file_name.endswith(('.mrc', '.rec')):
                in_path = os.path.join(input_path, file_name)
                out_name = f'{os.path.splitext(file_name)[0]}_crop.mrc'
                out_path = os.path.join(output_path, out_name)

                data, voxel_size = load_tomogram(in_path)
                crop = data[z1:z2, y1:y2, x1:x2]
                save_tomogram(out_path, crop, voxel_size)
                print(f'  {file_name}  {data.shape} → {crop.shape}  '
                      f'→ {out_name}')
    else:
        data, voxel_size = load_tomogram(input_path)
        crop = data[z1:z2, y1:y2, x1:x2]

        if os.path.isdir(output_path):
            base = os.path.splitext(os.path.basename(input_path))[0]
            out_path = os.path.join(output_path, f'{base}_crop.mrc')
        else:
            out_path = output_path

        save_tomogram(out_path, crop, voxel_size)
        print(f'{os.path.basename(input_path)}  {data.shape} → {crop.shape}'
              f'  → {os.path.basename(out_path)}')


def parser_helper():
    parser = argparse.ArgumentParser(
        'Crop a tomogram to defined z/y/x slices',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str, required=True,
                        help='MRC/REC file or folder of MRC/REC files')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output MRC file or output folder')
    parser.add_argument('--z1', type=int, default=0, help='Z start')
    parser.add_argument('--z2', type=int, default=None, help='Z end (None = full)')
    parser.add_argument('--y1', type=int, default=0, help='Y start')
    parser.add_argument('--y2', type=int, default=None, help='Y end (None = full)')
    parser.add_argument('--x1', type=int, default=0, help='X start')
    parser.add_argument('--x2', type=int, default=None, help='X end (None = full)')
    return parser


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.z1, args.z2, args.y1, args.y2, args.x1, args.x2)
