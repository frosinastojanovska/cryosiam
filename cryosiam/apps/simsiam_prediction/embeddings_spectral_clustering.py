import os
import umap
import yaml
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering
from sklearn.manifold._spectral_embedding import _set_diag
from scipy.sparse.csgraph import laplacian as graph_laplacian

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


def predict_k(affinity_matrix, max_k=100):
    """
    Predict number of clusters based on the eigengap.

    Parameters
    ----------
    affinity_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix.
        Each element of this matrix contains a measure of similarity between two of the data points.

    Returns
    ----------
    k : integer
        estimated number of cluster.

    Note
    ---------
    If graph is not fully connected, zero component as single cluster.

    References
    ----------
    A Tutorial on Spectral Clustering, 2007
        Luxburg, Ulrike
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf

    """

    """
    If normed=True, L = D^(-1/2) * (D - A) * D^(-1/2) else L = D - A.
    normed=True is recommended.
    """
    normed_laplacian, dd = graph_laplacian(affinity_matrix, normed=True, return_diag=True)
    laplacian = _set_diag(normed_laplacian, 1, True)

    """
    n_components size is N - 1.
    Setting N - 1 may lead to slow execution time...
    """
    n_components = max_k

    """
    shift-invert mode
    The shift-invert mode provides more than just a fast way to obtain a few small eigenvalues.
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html

    The normalized Laplacian has eigenvalues between 0 and 2.
    I - L has eigenvalues between -1 and 1.
    """
    eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues = -eigenvalues[::-1]  # Reverse and sign inversion.

    max_gap = 0
    gap_pre_index = 0
    for i in range(1, eigenvalues.size):
        gap = eigenvalues[i] - eigenvalues[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_pre_index = i - 1

    k = gap_pre_index + 1

    return k


def spectral_clustering(embeddings, num_clusters=10, estimate_num_clusters=False,
                        affinity='nearest_neighbors', n_neighbours=10):
    if estimate_num_clusters:
        clustering = SpectralClustering(n_clusters=2, affinity=affinity, n_neighbors=n_neighbours)
        clustering.fit(embeddings)
        num_clusters = predict_k(clustering.affinity_matrix_, max_k=num_clusters)
    clustering = SpectralClustering(n_clusters=num_clusters, affinity=affinity, n_neighbors=n_neighbours)
    labels = clustering.fit_predict(embeddings)
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

    spectral_clusters = spectral_clustering(embeddings, num_clusters=cfg['clustering_spectral']['num_clusters'],
                                            estimate_num_clusters=cfg['clustering_spectral'][
                                                'estimate_num_clusters'])
    spectral_clusters += 1

    clusters = spectral_clusters.flatten()
    print(f'Number or spectral clusters: {np.unique(spectral_clusters).shape[0]}')
    with h5py.File(os.path.join(prediction_folder, 'spectral_clusters.h5'), 'w') as f:
        f.create_dataset('clusters', data=spectral_clusters)

    i = 0
    labels = []
    for ind, file in enumerate(files):
        filename = file.split(cfg['file_extension'])[0]
        n = num_samples[ind]
        predicted_labels = []
        for _ in range(n):
            predicted_labels.append(clusters[i])
            i += 1

        with h5py.File(os.path.join(prediction_folder, f'{filename}_clusters_spectral.h5'), 'w') as f:
            f.create_dataset('predictions', data=np.array(predicted_labels))

        df = pd.read_csv(os.path.join(prediction_folder, f'{filename}_instance_regions.csv'))
        df['semantic_class'] = predicted_labels
        df.to_csv(os.path.join(prediction_folder, f'{filename}_instance_regions_spectral_clustered.csv'), index=False)
        labels += [f'{filename}{cfg["file_extension"]}_{x}' for x in df['label']]

    if cfg['clustering_spectral']['visualization']:
        file = os.path.join(prediction_folder, f'spectral_clusters.html')
        visualize_features_space(embeddings[:20000].T, file, clusters[:20000], labels[:20000])


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
