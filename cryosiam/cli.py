import argparse

from cryosiam.apps.dense_simsiam_regression.predict import main as denoise_predict_main
from cryosiam.apps.dense_simsiam_semantic.predict import main as semantic_predict_main
from cryosiam.apps.dense_simsiam_instance.predict import main as instance_predict_main
from cryosiam.apps.preprocessing.revert_and_scale_intensity import main as preprocessing
from cryosiam.apps.dense_simsiam_semantic.prediction_postprocessing import main as semantic_postprocessing

from cryosiam.apps.dense_simsiam_semantic.filter_ground_truth_labels import main as semantic_filter_ground_truth
from cryosiam.apps.dense_simsiam_semantic.preprocess_segmentation_maps import main as semantic_train_preprocess
from cryosiam.apps.dense_simsiam_semantic.create_patches import main as semantic_train_create_patches
from cryosiam.apps.dense_simsiam_semantic.train import main as semantic_train

__version__ = "1.0"


def main():
    parser = argparse.ArgumentParser(prog="cryosiam", description="CryoSiam Command Line Interface")
    parser.add_argument("--version", action="version", version=f"CryoSiam {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ############# Prediction commands #############

    # denoise_predict subcommand
    sp_denoise = subparsers.add_parser("denoise_predict", help="Run denoising prediction")
    sp_denoise.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_denoise.add_argument('--filename', type=str, required=False,
                            help='Process only this specific tomogram filename', default=None)
    sp_denoise.set_defaults(func=lambda args: denoise_predict_main(args.config_file, args.filename))

    # semantic_predict subcommand
    sp_semantic = subparsers.add_parser("semantic_predict", help="Run semantic segmentation prediction")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.add_argument('--filename', type=str, required=False,
                             help='Process only this specific tomogram filename', default=None)
    sp_semantic.set_defaults(func=lambda args: semantic_predict_main(args.config_file, args.filename))

    # instance_predict subcommand
    sp_instance = subparsers.add_parser("instance_predict", help="Run instance segmentation prediction")
    sp_instance.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_instance.add_argument('--filename', type=str, required=False,
                             help='Process only this specific tomogram filename', default=None)
    sp_instance.set_defaults(func=lambda args: instance_predict_main(args.config_file, args.filename))

    # semantic_postprocessing subcommand
    sp_semantic = subparsers.add_parser("semantic_postprocessing", help="Run semantic segmentation postprocessing")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.add_argument('--filename', type=str, required=False,
                             help='Process only this specific tomogram filename', default=None)
    sp_semantic.set_defaults(func=lambda args: semantic_postprocessing(args.config_file, args.filename))

    ############# Training commands #############

    # semantic segmentation training
    sp_semantic = subparsers.add_parser("semantic_filter_ground_truth",
                                        help="Run filtering of the ground truth labels data for semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_filter_ground_truth(args.config_file))

    sp_semantic = subparsers.add_parser("semantic_train_preprocess",
                                        help="Run preprocessing of the data for semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_train_preprocess(args.config_file))

    sp_semantic = subparsers.add_parser("semantic_train_create_patches",
                                        help="Run creating patches for semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_train_create_patches(args.config_file))

    sp_semantic = subparsers.add_parser("semantic_train", help="Run semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_train(args.config_file))

    args = parser.parse_args()
    # Run selected command
    args.func(args)
