import os
import csv
import yaml
import umap
import h5py
import mrcfile
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from cryosiam.utils import parser_helper


def save_tomogram(file_path, data):
    """Save a numpy array as tomogram in MRC or REC file format.
    :param file_path: path to the file
    :type file_path: str
    :param data: the data to be stored as tomogram
    :type data: np.array
    :return: tomogram as numpy array as confirmation of the saving
    :rtype: np.array
    """
    with mrcfile.new(file_path, data=data, overwrite=True) as m:
        return m.data


def pca_reduce_dimensions(patches_embeddings, n=3):
    """Create PCA representation of the first n principal components of the embeddings
    :param patches_embeddings: the embeddings
    :type patches_embeddings: np.array
    :param n: number of principal components
    :type n: int
    :return: first n principal components
    :rtype: np.array
    """
    prediction_out_shape = patches_embeddings.shape
    print('PCA')
    if len(prediction_out_shape) > 2:
        if len(prediction_out_shape) == 3:
            patches_embeddings = patches_embeddings.reshape(prediction_out_shape[0],
                                                            prediction_out_shape[1] * prediction_out_shape[2]).T
        else:
            patches_embeddings = patches_embeddings.reshape(prediction_out_shape[0],
                                                            prediction_out_shape[1] * prediction_out_shape[2] *
                                                            prediction_out_shape[3]).T
    else:
        patches_embeddings = patches_embeddings.T
    pca = PCA(n_components=n, svd_solver='arpack')
    pca_result = pca.fit_transform(patches_embeddings)
    if len(prediction_out_shape) > 2:
        if len(prediction_out_shape) == 3:
            pca_result = pca_result.T.reshape(n, prediction_out_shape[1], prediction_out_shape[2])
        else:
            pca_result = pca_result.T.reshape(n, prediction_out_shape[1], prediction_out_shape[2],
                                              prediction_out_shape[3])
    else:
        pca_result = pca_result.T
    # pca_result = (pca_result - np.min(pca_result)) / (np.max(pca_result) - np.min(pca_result))
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    return pca_result


def visualize_features_space(image_features, filename, data, three_dimensions=False,
                             discrete_colors=False, distance='euclidean', n_neighbors=10, min_dist=0,
                             pca_components=None):
    """Create UMAP 2D/3D representation of the embeddings with labels from given segmentation
    :param image_features: the embeddings
    :type image_features: np.array
    :param filename: name of the file to save the visualization
    :type filename: str
    :param data: the metadata of every embedding point
    :type data: list(list)
    :param three_dimensions: flag if the UMAP should be 3D instead of 2D
    :type three_dimensions: bool
    :param discrete_colors: whether the colors of the scatter plot need to be discrete
    :type discrete_colors: bool
    :param distance: distance metric parameter for the UMAP
    :type distance: str
    :param n_neighbors: n_neighbors parameter for the UMAP
    :type n_neighbors: int
    :param min_dist: min_dist parameter for the UMAP
    :type min_dist: float
    :param pca_components: number of components for the PCA before the UMAP, leave None to not apply PCA
    :type pca_components: int or None
    :return: None
    :rtype: None
    """
    if pca_components is not None:
        image_features = pca_reduce_dimensions(image_features, pca_components)
    image_features_shape = image_features.shape
    if len(image_features_shape) > 2:
        if len(image_features_shape) == 3:
            image_features = image_features.reshape(image_features_shape[0],
                                                    image_features_shape[1] * image_features_shape[2]).T
        else:
            image_features = image_features.reshape(image_features_shape[0],
                                                    image_features_shape[1] * image_features_shape[2] *
                                                    image_features_shape[3]).T
    else:
        image_features = image_features.T

    # if 'semantic_class' in data.columns:
    #     data.loc[data['semantic_class'] == -1, 'semantic_class'] = 0
    #     data = data.sort_values(by='semantic_class')

    if discrete_colors and 'semantic_class' in data.columns:
        data['semantic_class'] = data['semantic_class'].apply(str)
        if 'semantic_class_2' in data.columns:
            data['semantic_class_2'] = data['semantic_class_2'].apply(str)

    if three_dimensions:
        if pca_components is None:
            u = umap.UMAP(n_components=3, metric=distance, n_neighbors=n_neighbors, min_dist=min_dist, random_state=10)
            projections = u.fit_transform(image_features)
            x, y, z = projections[:, 0], projections[:, 1], projections[:, 2]
        else:
            x, y, z = image_features[:, 0], image_features[:, 1], image_features[:, 2]

        data['x'] = list(x)
        data['y'] = list(y)
        data['z'] = list(z)
        fig = px.scatter_3d(data, x='x', y='y', z='z',
                            color='semantic_class_2' if 'semantic_class_2' in data.columns else 'semantic_class' if 'semantic_class' in data.columns else 'area',
                            hover_data=data.columns)
        fig.write_html(filename)
    else:
        if pca_components is None:
            u = umap.UMAP(n_components=2, metric=distance, n_neighbors=n_neighbors, min_dist=min_dist, random_state=10)
            projections = u.fit_transform(image_features)
            x, y = projections[:, 0], projections[:, 1]
        else:
            x, y = image_features[:, 0], image_features[:, 1]
        data['x'] = list(x)
        data['y'] = list(y)
        fig = px.scatter(data, x='x', y='y',
                         # color='distance' if 'distance' in data.columns else 'semantic_class_2' if 'semantic_class_2' in data.columns else 'semantic_class' if 'semantic_class' in data.columns else 'log_area',
                         color='semantic_class_2' if 'semantic_class_2' in data.columns else 'semantic_class' if 'semantic_class' in data.columns else 'log_area',
                         hover_data=data.columns, opacity=0.5,
                         color_discrete_sequence=px.colors.qualitative.Light24)
        # fig.write_image(filename.split('.html')[0] + '.svg')
        fig.write_html(filename)
        data.to_csv(filename.split('.html')[0] + '_data.csv')


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    prediction_folder = cfg['prediction_folder']
    prediction_visualization_folder = cfg['visualization']['prediction_folder']
    instances_folder = cfg['instances_mask_folder']
    os.makedirs(prediction_folder, exist_ok=True)
    os.makedirs(prediction_visualization_folder, exist_ok=True)
    files = cfg['visualization_files']
    if files is None:
        files = [x for x in os.listdir(prediction_folder) if x.endswith('embeds.h5')]
    else:
        files = [x.split(cfg['file_extension'])[0] + '_embeds.h5' for x in files]

    for file in files:
        if not os.path.exists(os.path.join(prediction_folder, file)):
            continue
        with h5py.File(os.path.join(prediction_folder, file), 'r') as f:
            embeddings = f['embeddings'][()]
        labels_file = f'{file.split("embeds.h5")[0]}instance_labels.h5'
        with h5py.File(os.path.join(prediction_folder, labels_file), 'r') as f:
            source_labels = f['labels'][()]

        file_name = f'{file.split("embeds.h5")[0]}{cfg["visualization"]["visualization_suffix"] if "visualization_suffix" in cfg["visualization"] else "instance_regions.csv"}'
        with open(os.path.join(prediction_folder, file_name)) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            metadata = {}
            header = None
            for row in reader:
                if header is None:
                    header = row
                    continue
                metadata[int(row[0])] = [int(float(x)) if header[i] not in ['distance', 'prob'] else float(x) for i, x
                                         in enumerate(row)]

        hover_data = pd.DataFrame([metadata[label] for label in source_labels], columns=header)
        hover_data['log_area'] = np.log2(hover_data['area'])

        if cfg['visualization']['visualize_umap']:
            file_name = os.path.join(prediction_visualization_folder,
                                     f'{file.split(".")[0]}_{"pca" if cfg["visualization"]["pca_components"] else "umap"}.html')
            visualize_features_space(embeddings, file_name, hover_data,
                                     distance=cfg['visualization']['distance'],
                                     pca_components=cfg['visualization']['pca_components'],
                                     three_dimensions=cfg['visualization']['3d_umap'],
                                     discrete_colors=True)


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
