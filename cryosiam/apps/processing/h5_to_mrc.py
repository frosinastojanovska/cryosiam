import os
import h5py
import mrcfile
import argparse


def save_tomogram(file_path, data):
    """Save a numpy array as tomogram in MRC or REC file format.
    :param file_path: path to the file
    :type file_path: str
    :param data: the data to be stored as tomogram
    :type data: np.array
    :return: tomogram as numpy array as confirmation of the saving
    :rtype: np.array
    """
    with mrcfile.new(file_path, data=data) as m:
        return m.data


def load_numpy_from_h5_file(filename):
    """Load numpy matrix data from .h5 file
    :param filename: path of the .h5 file
    :type filename: str
    :param label: label of the data in the saved file
    :type label: str
    :return: the loaded numpy array
    :rtype: np.array
    """
    output = {}
    with h5py.File(filename, 'r') as hf:
        labels = [key for key in hf.keys()]
        for label in labels:
            output[label] = hf[label][()]
    return output


def main(input_path, output_path):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for file_name in os.listdir(input_path):
            if file_name.endswith(".h5"):
                data = load_numpy_from_h5_file(os.path.join(input_path, os.path.basename(file_name)))
                for key in data.keys():
                    save_tomogram(os.path.join(output_path,
                                               f'{os.path.basename(file_name).split(".h5")[0]}_{key}.mrc'), data[key])
    else:
        data = load_numpy_from_h5_file(input_path)
        for key in data.keys():
            save_tomogram(os.path.join(output_path,
                                       f'{os.path.basename(input_path).split(".h5")[0]}_{key}.mrc'), data[key])


def parser_helper(description=None):
    description = "Convert a h5 file into an mrc file" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str, required=True, help='path to the input h5 file or '
                                                                      'path to the folder with input h5 file/s')
    parser.add_argument('--output_path', type=str, required=True, help='path to save the output mrc file or '
                                                                       'path to folder to save the output mrc file/s')
    return parser


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.input_path, args.output_path)
