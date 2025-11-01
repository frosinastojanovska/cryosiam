import os
import umap
import yaml
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cryosiam.utils import parser_helper


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
            pca_result = pca_result.reshape(prediction_out_shape[1], prediction_out_shape[2], n)
        else:
            pca_result = pca_result.reshape(prediction_out_shape[1], prediction_out_shape[2],
                                            prediction_out_shape[3], n)
    pca_result = (pca_result - np.min(pca_result)) / (np.max(pca_result) - np.min(pca_result))
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    return pca_result


def cluster_kmeans(embeddings, num_clusters=4, max_iter=1000, return_centers=True):
    """Perform clustering with given embeddings with kmeans
    :param embeddings: the embeddings
    :type embeddings: np.array
    :param num_clusters: number of clusters for the Kmeans
    :type num_clusters: int
    :param return_centers: if the center of clusters should be returned
    :type return_centers: bool
    :return: segmentation of the image (and optional center of clusters)
    :rtype: np.array
    """
    print('kMeans')
    clustering = KMeans(n_clusters=num_clusters, max_iter=max_iter, n_init='auto')
    labels = clustering.fit_predict(embeddings)
    if return_centers:
        return labels, clustering.cluster_centers_
    return labels


def visualize_features_space(image_features, filename, classes, labels, discrete_colors=True, distance='euclidean',
                             n_neighbors=10, min_dist=0, pca_components=None):
    """Create UMAP 2D representation of the embeddings with labels from given segmentation
    :param image_features: the embeddings
    :type image_features: np.array
    :param filename: name of the file to save the visualization
    :type filename: str
    :param classes: the class labels for every point
    :type classes: list(list)
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

    data = pd.DataFrame({'class': classes})
    if discrete_colors:
        data['class'] = data['class'].apply(str)

    if pca_components is None:
        u = umap.UMAP(n_components=2, metric=distance, n_neighbors=n_neighbors, min_dist=min_dist,
                      random_state=10, n_jobs=1)
        projections = u.fit_transform(image_features)
        x, y = projections[:, 0], projections[:, 1]
    else:
        x, y = image_features[:, 0], image_features[:, 1]
    data['x'] = list(x)
    data['y'] = list(y)
    data['labels'] = labels
    fig = px.scatter(data, x='x', y='y', color='class', hover_data=data.columns, opacity=0.5,
                     color_discrete_sequence=px.colors.qualitative.Light24)
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(showline=True, linecolor='black', linewidth=1)
    fig.update_yaxes(showline=True, linecolor='black', linewidth=1)
    fig.write_html(filename)
    data.to_csv(filename.split('.html')[0] + '_umap_data.csv', index=False)


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    prediction_folder = cfg['prediction_folder']
    files = cfg['clustering_files']
    if files is None:
        files = [x.split('_embeds.h5')[0] for x in os.listdir(prediction_folder) if
                 '_embeds.h5' in x and os.path.isfile(os.path.join(prediction_folder, x))]

    files = sorted(files)

    if not os.path.isfile(os.path.join(prediction_folder, 'embeddings.h5')):
        embeddings = []
        num_samples = []
        for ind, file in enumerate(files):
            file = file.split(cfg['file_extension'])[0]
            with h5py.File(os.path.join(prediction_folder, f'{file}_embeds.h5'), 'r') as f:
                patch_embeddings = f['embeddings'][()].T
            embeddings.append(patch_embeddings)
            num_samples.append(patch_embeddings.shape[0])
        num_samples = np.array(num_samples)
        embeddings = np.concatenate(embeddings, axis=0)
        with h5py.File(os.path.join(prediction_folder, 'embeddings.h5'), 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            f.create_dataset('number_of_samples', data=num_samples)
    else:
        with h5py.File(os.path.join(prediction_folder, 'embeddings.h5'), 'r') as f:
            embeddings = f['embeddings'][()]
            num_samples = f['number_of_samples'][()]

    clusters, centers = cluster_kmeans(embeddings,
                                       cfg['clustering_kmeans']['num_clusters'],
                                       return_centers=True)
    clusters = clusters.flatten()
    print(f'Number or K-Means clusters: {np.unique(clusters).shape[0]}')
    with h5py.File(os.path.join(prediction_folder, 'kmeans_clusters.h5'), 'w') as f:
        f.create_dataset('clusters', data=clusters)
        f.create_dataset('centers', data=centers)

    i = 0
    labels = []
    for ind, file in enumerate(files):
        filename = file.split(cfg['file_extension'])[0]
        n = num_samples[ind]
        predicted_labels = []
        for _ in range(n):
            predicted_labels.append(clusters[i])
            i += 1

        with h5py.File(os.path.join(prediction_folder, f'{filename}_clusters_kmeans.h5'), 'w') as f:
            f.create_dataset('predictions', data=np.array(predicted_labels))

        df = pd.read_csv(os.path.join(prediction_folder, f'{filename}_instance_regions.csv'))
        df['semantic_class'] = predicted_labels
        df.to_csv(os.path.join(prediction_folder, f'{filename}_instance_regions_kmeans_clustered.csv'), index=False)
        labels += [f'{filename}{cfg["file_extension"]}_{x}' for x in df['label']]

    if cfg['clustering_kmeans']['visualization']:
        file = os.path.join(prediction_folder, f'kmeans_clusters.html')
        visualize_features_space(embeddings[:20000].T, file, clusters[:20000], labels[:20000])


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
