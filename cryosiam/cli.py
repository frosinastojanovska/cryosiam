import argparse

from cryosiam.apps.dense_simsiam_regression.predict import main as denoise_predict_main
from cryosiam.apps.dense_simsiam_semantic.predict import main as semantic_predict_main
from cryosiam.apps.dense_simsiam_instance.predict import main as instance_predict_main
from cryosiam.apps.dense_simsiam_instance.filtering_instances import main as instance_filter_main
from cryosiam.apps.dense_simsiam_semantic.prediction_postprocessing import main as semantic_postprocessing
from cryosiam.apps.dense_simsiam_semantic.semantic_prediction_to_center_points import \
    main as semantic_prediction_to_center_points
from cryosiam.apps.simsiam_prediction.extract_patches_embeddings import main as extract_patches_embeddings_main
from cryosiam.apps.simsiam_prediction.extract_patches_embeddings_from_centers import \
    main as extract_patches_embeddings_from_centers_main
from cryosiam.apps.simsiam_prediction.embeddings_kmeans_clustering import main as embeddings_kmeans_clustering_main
from cryosiam.apps.simsiam_prediction.embeddings_spectral_clustering import main as embeddings_spectral_clustering_main
from cryosiam.apps.simsiam_prediction.visualize_embeddings import main as visualize_embeddings_main

from cryosiam.apps.dense_simsiam_semantic.filter_ground_truth_labels import main as semantic_filter_ground_truth
from cryosiam.apps.dense_simsiam_semantic.preprocess_segmentation_maps import main as semantic_train_preprocess
from cryosiam.apps.dense_simsiam_semantic.create_patches import main as semantic_train_create_patches
from cryosiam.apps.dense_simsiam_semantic.train import main as semantic_train

from cryosiam.apps.processing.invert_and_scale_intensity import main as invert_and_scale_intensity_main
from cryosiam.apps.processing.create_sphere_mask_from_coordinates import \
    main as create_sphere_mask_from_coordinates_main
from cryosiam.apps.processing.create_sphere_mask_from_coordinates_multiclass import \
    main as create_sphere_mask_from_coordinates_multiclass_main
from cryosiam.apps.processing.create_binary_map_after_sta import main as create_binary_map_after_sta_main
from cryosiam.apps.processing.create_multi_class_mask_from_binary_masks import \
    main as create_multi_class_mask_from_binary_masks_main

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

    # instance_filter subcommand
    sp_instance = subparsers.add_parser("instance_filter", help="Run instance segmentation prediction")
    sp_instance.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_instance.add_argument('--filename', type=str, required=False,
                             help='Process only this specific tomogram filename', default=None)
    sp_instance.set_defaults(func=lambda args: instance_filter_main(args.config_file, args.filename))

    # semantic_postprocessing subcommand
    sp_semantic = subparsers.add_parser("semantic_postprocessing", help="Run semantic segmentation postprocessing")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.add_argument('--filename', type=str, required=False,
                             help='Process only this specific tomogram filename', default=None)
    sp_semantic.set_defaults(func=lambda args: semantic_postprocessing(args.config_file, args.filename))

    # semantic_to_centers subcommand
    sp_semantic = subparsers.add_parser("semantic_to_centers", help="Run semantic segmentation postprocessing")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_prediction_to_center_points(args.config_file))

    # simsiam_embeddings_predict subcommand
    sp_simsiam = subparsers.add_parser("simsiam_embeddings_predict", help="Run SimSiam embeddings generation")
    sp_simsiam.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_simsiam.add_argument('--filename', type=str, required=False,
                            help='Process only this specific tomogram filename', default=None)
    sp_simsiam.set_defaults(func=lambda args: extract_patches_embeddings_main(args.config_file, args.filename))

    # simsiam_embeddings_from_centers_predict subcommand
    sp_simsiam = subparsers.add_parser("simsiam_embeddings_from_centers_predict",
                                       help="Run SimSiam embeddings generation from center points")
    sp_simsiam.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_simsiam.add_argument('--filename', type=str, required=False,
                            help='Process only this specific tomogram filename', default=None)
    sp_simsiam.set_defaults(func=lambda args: extract_patches_embeddings_from_centers_main(args.config_file,
                                                                                           args.filename))

    # simsiam_embeddings_kmeans_cluster subcommand
    sp_simsiam = subparsers.add_parser("simsiam_embeddings_kmeans_clustering",
                                       help="Run SimSiam embeddings KMeans clustering")
    sp_simsiam.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_simsiam.set_defaults(func=lambda args: embeddings_kmeans_clustering_main(args.config_file))

    # simsiam_embeddings_spectral_cluster subcommand
    sp_simsiam = subparsers.add_parser("simsiam_embeddings_spectral_clustering",
                                       help="Run SimSiam embeddings spectral clustering")
    sp_simsiam.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_simsiam.set_defaults(func=lambda args: embeddings_spectral_clustering_main(args.config_file))

    # simsiam_visualize_embeddings subcommand
    sp_simsiam = subparsers.add_parser("simsiam_visualize_embeddings",
                                       help="Run SimSiam visualize embeddings")
    sp_simsiam.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_simsiam.set_defaults(func=lambda args: visualize_embeddings_main(args.config_file))

    ############# Training commands #############

    # semantic segmentation training
    sp_semantic = subparsers.add_parser("semantic_filter_ground_truth",
                                        help="Run filtering of the ground truth labels data for semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_filter_ground_truth(args.config_file))

    sp_semantic = subparsers.add_parser("semantic_train_preprocess",
                                        help="Run processing of the data for semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_train_preprocess(args.config_file))

    sp_semantic = subparsers.add_parser("semantic_train_create_patches",
                                        help="Run creating patches for semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_train_create_patches(args.config_file))

    sp_semantic = subparsers.add_parser("semantic_train", help="Run semantic segmentation training")
    sp_semantic.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    sp_semantic.set_defaults(func=lambda args: semantic_train(args.config_file))

    ############# Processing commands #############
    # Invert and scale command
    sp_process = subparsers.add_parser("processing_invert_scale",
                                       help="Run invert and/or scale of the given tomogram/s intensities")
    sp_process.add_argument('--input_path', type=str, required=True, help='path to the input tomogram or '
                                                                          'path to the folder with input tomogram/s')
    sp_process.add_argument('--output_path', type=str, required=True, help='path to save the output tomogram or '
                                                                           'path to folder to save the output tomogram/s')
    sp_process.add_argument("--invert", action="store_true", default=False, help="Inverts contrast of images.")
    sp_process.add_argument("--lower_end_percentage", type=float, required=False,
                            help="Cut off values from the lower percentile end of the intensities.")
    sp_process.add_argument("--upper_end_percentage", type=float, required=False,
                            help="Cut off values from the upper percentile end of the intensities.")
    sp_process.set_defaults(
        func=lambda args: invert_and_scale_intensity_main(args.input_path, args.output_path, args.invert,
                                                          args.lower_end_percentage, args.upper_end_percentage))

    # Create binary sphere mask
    sp_process = subparsers.add_parser("processing_create_sphere_mask",
                                       help="Create binary tomogram mask with sphere map from given center coordinates")
    sp_process.add_argument('--coordinates_file', type=str, required=True,
                            help='path to star file or csv file with X,Y,Z coordinates. The star file needs the '
                                 '[rlnCoordinateX, rlnCoordinateY, rlnCoordinateZ] header, while the csv file header '
                                 'should be [centroid-0, centroid-1, centroid-2] for z, y, x order')
    sp_process.add_argument('--sphere_radius', type=int, required=True,
                            help='radius in number of pixels for the sphere')
    sp_process.add_argument('--output_dir', type=str, required=True,
                            help='path to folder to save the output tomogram/s')
    sp_process.add_argument('--tomogram_path', type=str, required=True,
                            help='path to one folder with tomograms to determine the 3D size of the output. '
                                 'If the path is path to one tomogram, it will use the size of that tomogram '
                                 'to all of the tomograms')
    sp_process.add_argument('--tomo_name', type=str, required=False,
                            help='process only this tomogram, the name should match the rlnMicrographName in '
                                 'the starfile or the tomo in the csv file')
    sp_process.set_defaults(
        func=lambda args: create_sphere_mask_from_coordinates_main(args.coordinates_file, args.sphere_radius,
                                                                   args.output_dir, args.tomogram_path,
                                                                   args.tomo_name))

    # Create multiclass sphere mask
    sp_process = subparsers.add_parser("processing_create_sphere_mask_multiclass",
                                       help="Create binary tomogram mask with sphere map from given center coordinates")
    sp_process.add_argument('--coordinates_file', type=str, required=True,
                            help='path to star file or csv file with X,Y,Z coordinates. The star file needs the '
                                 '[rlnCoordinateX, rlnCoordinateY, rlnCoordinateZ, rlnClassNumber] header, while the '
                                 'csv file header should be [centroid-0, centroid-1, centroid-2, semantic_class] '
                                 'for z, y, x order. The class numbers have to start from 1 and to have sequential order.')
    sp_process.add_argument('--sphere_radius', type=str, required=True,
                            help='radius in number of pixels for the sphere for different classes. The expected input is '
                                 'N integers separated by comma, where N is the number of different classes of particles')
    sp_process.add_argument('--output_dir', type=str, required=True,
                            help='path to folder to save the output tomogram/s')
    sp_process.add_argument('--tomogram_path', type=str, required=True,
                            help='path to one folder with tomograms to determine the 3D size of the output. '
                                 'If the path is path to one tomogram, it will use the size of that tomogram '
                                 'to all of the tomograms')
    sp_process.add_argument('--tomo_name', type=str, required=False,
                            help='process only this tomogram, the name should match the rlnMicrographName in '
                                 'the starfile or the tomo in the csv file')
    sp_process.set_defaults(
        func=lambda args: create_sphere_mask_from_coordinates_multiclass_main(args.coordinates_file, args.sphere_radius,
                                                                              args.output_dir, args.tomogram_path,
                                                                              args.tomo_name))

    # Create binary density map mask
    sp_process = subparsers.add_parser("processing_create_binary_map_after_sta",
                                       help="Create binary map from given sta average map, canter coordinates and orientations")
    sp_process.add_argument('--star_file', type=str, required=True,
                            help='path to star file with orientations after STA')
    sp_process.add_argument('--map_file', type=str, required=True,
                            help='path to the map file that will be placed in 3D tomogram (it is expected to be cubic subvolume)')
    sp_process.add_argument('--output_dir', type=str, required=True,
                            help='path to folder to save the output tomogram/s')
    sp_process.add_argument("--map_threshold", type=float, required=True,
                            help="Threshold for the map to binarize it.")
    sp_process.add_argument('--example_tomogram', type=str, required=True,
                            help='path to one tomogram to determine the 3D size of the output')
    sp_process.add_argument('--tomo_name', type=str, required=False,
                            help='process only this tomogram, the name should match the rlnMicrographName')
    sp_process.set_defaults(
        func=lambda args: create_binary_map_after_sta_main(args.star_file, args.map_file,
                                                           args.output_dir, args.map_threshold,
                                                           args.example_tomogram, args.tomo_name))

    # Create multi-class masks from binary masks
    sp_process = subparsers.add_parser("processing_create_multiclass_from_binary_masks",
                                       help="Create binary map from given sta average map, canter coordinates and orientations")
    sp_process.add_argument('--root_binary_masks_folder', type=str, required=True,
                            help='path to a folder that contains subfolders for every binary label. The subfolders contain the binary masks for every separate label')
    sp_process.add_argument('--output_dir', type=str, required=True,
                            help='path to folder to save the output tomogram/s')
    sp_process.add_argument('--tomo_name', type=str, required=False,
                            help='process only this tomogram (include the file extension in the name)')
    sp_process.set_defaults(
        func=lambda args: create_multi_class_mask_from_binary_masks_main(args.root_binary_masks_folder, args.output_dir,
                                                                         args.tomo_name))

    args = parser.parse_args()
    # Run selected command
    args.func(args)
