#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import starfile
import umap
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CLUSTER_COLORS = np.asarray([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 215, 180],
    [0, 0, 128],
    [128, 128, 128],
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cluster saved prediction embeddings, write labelled CSV/STAR files, and create a UMAP HTML.'
    )
    parser.add_argument('--config_file', required=True)
    return parser.parse_args()


def get_analysis_config(config):
    analysis = dict(config.get('embedding_analysis', {}))
    legacy = config.get('clustering_kmeans', {}) or {}
    analysis.setdefault('n_clusters', legacy.get('num_clusters', 4))
    analysis.setdefault('visualization', legacy.get('visualization', True))
    return analysis


def normalize_stem(value, file_extension):
    name = Path(str(value)).name
    if name.endswith('_embeds.h5'):
        name = name[:-len('_embeds.h5')]
    if file_extension and name.endswith(file_extension):
        name = name[:-len(file_extension)]
    return name


def discover_stems(prediction_folder, config, file_extension):
    requested = config.get('clustering_files')
    if requested:
        stems = [normalize_stem(item, file_extension) for item in requested]
    else:
        stems = []
        for path in sorted(prediction_folder.glob('*_embeds.h5')):
            stems.append(normalize_stem(path.name, file_extension))
    if not stems:
        raise FileNotFoundError(
            f'No *_embeds.h5 files found in {prediction_folder}. '
            'The prediction step must save embeddings before this script can run.'
        )
    return list(dict.fromkeys(stems))


def load_embeddings(path, n_rows, dataset_key='embeddings'):
    with h5py.File(path, 'r') as handle:
        if dataset_key not in handle:
            raise KeyError(f'{path} does not contain the HDF5 dataset {dataset_key!r}')
        values = np.asarray(handle[dataset_key][()])

    if values.ndim != 2:
        raise ValueError(f'Expected a 2D embedding array in {path}, got shape {values.shape}')
    if values.shape[0] == n_rows:
        values = values
    elif values.shape[1] == n_rows:
        values = values.T
    else:
        raise ValueError(
            f'Embedding/CSV row mismatch for {path}: embeddings have shape {values.shape}, '
            f'but the corresponding CSV has {n_rows} rows.'
        )

    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite values found in {path}')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError(f'Zero-length embedding found in {path}')
    return values / norms


def get_cluster_names(analysis, n_clusters):
    names = analysis.get('cluster_names')
    if isinstance(names, dict):
        result = []
        for cluster_id in range(n_clusters):
            result.append(str(names.get(cluster_id, names.get(str(cluster_id), f'cluster_{cluster_id + 1}'))))
        return result
    if isinstance(names, (list, tuple)):
        if len(names) != n_clusters:
            raise ValueError('embedding_analysis.cluster_names must have n_clusters entries')
        return [str(name) for name in names]
    return [f'cluster_{cluster_id + 1}' for cluster_id in range(n_clusters)]


def get_cluster_colors(analysis, n_clusters):
    configured = analysis.get('cluster_colors')
    if configured is not None:
        colors = np.asarray(configured, dtype=np.uint8)
        if colors.shape != (n_clusters, 3):
            raise ValueError('embedding_analysis.cluster_colors must have shape [n_clusters, 3]')
        return np.vstack([np.zeros((1, 3), dtype=np.uint8), colors])
    repeats = int(np.ceil(n_clusters / len(CLUSTER_COLORS)))
    colors = np.tile(CLUSTER_COLORS, (repeats, 1))[:n_clusters]
    return np.vstack([np.zeros((1, 3), dtype=np.uint8), colors])


def to_star(data):
    required = ['centroid-0', 'centroid-1', 'centroid-2']
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f'Cannot create STAR file; missing columns: {missing}')

    result = pd.DataFrame(index=data.index)
    tomo = data['tomo'].astype(str) if 'tomo' in data.columns else data['_source_stem'].astype(str)
    result['rlnTomoName'] = tomo
    result['rlnMicrographName'] = tomo
    result['rlnCoordinateX'] = data['centroid-2'].astype(float)
    result['rlnCoordinateY'] = data['centroid-1'].astype(float)
    result['rlnCoordinateZ'] = data['centroid-0'].astype(float)

    optional_columns = {
        'label': 'rlnLabel',
        'area': 'rlnArea',
        'bbox-0': 'rlnBbox-0',
        'bbox-1': 'rlnBbox-1',
        'bbox-2': 'rlnBbox-2',
        'bbox-3': 'rlnBbox-3',
        'bbox-4': 'rlnBbox-4',
        'bbox-5': 'rlnBbox-5',
        'embedding_cluster': 'rlnEmbeddingCluster',
        'semantic_class': 'rlnClass',
        'class_name': 'rlnClassLabel',
    }
    for source, target in optional_columns.items():
        if source in data.columns:
            result[target] = data[source].to_numpy()
    return result.reset_index(drop=True)


def write_cluster_volume(prediction_path, group, cluster_names, cluster_colors, analysis):
    if not bool(analysis.get('write_cluster_mask', True)):
        return
    if not prediction_path.is_file():
        print(f'No prediction H5 for {group.iloc[0]["_source_stem"]}: {prediction_path}')
        return
    if 'label' not in group.columns:
        print(f'Cannot create cluster mask for {prediction_path.name}: CSV has no label column')
        return

    with h5py.File(prediction_path, 'a') as handle:
        if 'instance_mask' not in handle:
            print(f'No instance_mask in {prediction_path}; skipping cluster mask')
            return
        instance_mask = np.asarray(handle['instance_mask'][()])
        if instance_mask.ndim != 3:
            raise ValueError(f'Expected a 3D instance_mask in {prediction_path}, got {instance_mask.shape}')

        instance_labels = group['label'].to_numpy()
        if not np.all(np.isfinite(instance_labels)):
            raise ValueError(f'Non-finite instance labels in {prediction_path}')
        instance_labels = instance_labels.astype(np.int64)
        semantic_classes = group['semantic_class'].to_numpy(np.int32)
        present_labels = np.unique(instance_mask)
        missing_labels = sorted(set(instance_labels.tolist()) - set(present_labels.tolist()))
        if missing_labels:
            print(
                f'  WARNING: {len(missing_labels)} CSV instance labels are absent from '
                f'{prediction_path.name}; those instances will not appear in the cluster mask.'
            )

        cluster_mask = np.zeros(instance_mask.shape, dtype=np.uint16)
        max_label = int(instance_mask.max()) if instance_mask.size else 0
        if max_label >= 0 and max_label <= 10_000_000:
            lookup = np.zeros(max_label + 1, dtype=np.uint16)
            valid = (instance_labels >= 0) & (instance_labels <= max_label)
            lookup[instance_labels[valid]] = semantic_classes[valid].astype(np.uint16)
            cluster_mask = lookup[instance_mask.astype(np.int64)]
        else:
            for label, semantic_class in zip(instance_labels, semantic_classes):
                cluster_mask[instance_mask == label] = semantic_class

        for key in (
        'embedding_cluster_mask', 'embedding_cluster_rgb', 'embedding_cluster_colors', 'embedding_cluster_names'):
            if key in handle:
                del handle[key]
        handle.create_dataset(
            'embedding_cluster_mask',
            data=cluster_mask,
            compression='gzip',
            shuffle=True,
        )
        handle.create_dataset('embedding_cluster_colors', data=cluster_colors)
        string_type = h5py.string_dtype(encoding='utf-8')
        handle.create_dataset(
            'embedding_cluster_names',
            data=np.asarray(['background'] + list(cluster_names), dtype=object),
            dtype=string_type,
        )
        handle['embedding_cluster_mask'].attrs['values_are_one_based_semantic_classes'] = True

        if bool(analysis.get('write_cluster_rgb', False)):
            rgb = cluster_colors[cluster_mask]
            handle.create_dataset(
                'embedding_cluster_rgb',
                data=rgb,
                compression='gzip',
                shuffle=True,
            )
            print(f'  Saved embedding_cluster_rgb in {prediction_path.name}')


def write_per_tomogram_outputs(all_data, output_folder, prediction_folder, cluster_names, cluster_colors, analysis):
    for stem, group in all_data.groupby('_source_stem', sort=False):
        group = group.sort_values('_row_index').copy()
        public = group.drop(columns=['_source_stem', '_row_index'])
        public.to_csv(output_folder / f'{stem}_instance_regions_clustered.csv', index=False)
        starfile.write(
            to_star(group),
            output_folder / f'{stem}_instance_regions_clustered.star',
            overwrite=True,
        )
        with h5py.File(output_folder / f'{stem}_embedding_clusters.h5', 'w') as handle:
            handle.create_dataset('embedding_cluster', data=group['embedding_cluster'].to_numpy(np.int32))
            handle.create_dataset('semantic_class', data=group['semantic_class'].to_numpy(np.int32))
            handle.create_dataset('cluster_colors', data=cluster_colors)
            string_type = h5py.string_dtype(encoding='utf-8')
            handle.create_dataset(
                'cluster_names',
                data=np.asarray(['background'] + list(cluster_names), dtype=object),
                dtype=string_type,
            )
        write_cluster_volume(
            prediction_folder / f'{stem}_preds.h5',
            group,
            cluster_names,
            cluster_colors,
            analysis,
        )


def write_umap(all_data, embeddings, analysis, output_folder, seed):
    if not bool(analysis.get('visualization', True)):
        return
    if len(embeddings) < 3:
        print('Skipping UMAP: fewer than three instances.')
        return

    max_points = int(analysis.get('max_umap_points', 20000))
    rng = np.random.default_rng(seed)
    if max_points > 0 and len(embeddings) > max_points:
        selected = np.sort(rng.choice(len(embeddings), size=max_points, replace=False))
    else:
        selected = np.arange(len(embeddings))

    features = embeddings[selected]
    pca_components = analysis.get('pca_components', 50)
    if pca_components is not None:
        pca_components = min(int(pca_components), features.shape[0] - 1, features.shape[1])
        if pca_components >= 2:
            features = PCA(n_components=pca_components, random_state=seed).fit_transform(features)

    n_neighbors = min(int(analysis.get('n_neighbors', 15)), len(features) - 1)
    reducer = umap.UMAP(
        n_components=2,
        metric=analysis.get('metric', 'cosine'),
        n_neighbors=max(2, n_neighbors),
        min_dist=float(analysis.get('min_dist', 0.1)),
        random_state=seed,
        n_jobs=1,
    )
    projection = reducer.fit_transform(features)
    plot_data = all_data.iloc[selected].copy()
    plot_data['umap_1'] = projection[:, 0]
    plot_data['umap_2'] = projection[:, 1]
    plot_data['tomogram'] = plot_data.get('tomo', plot_data['_source_stem'])
    hover_columns = [
        column for column in [
            'tomogram', 'label', 'centroid-0', 'centroid-1', 'centroid-2',
            'area', 'embedding_cluster', 'semantic_class', 'class_name',
        ] if column in plot_data.columns
    ]
    fig = px.scatter(
        plot_data,
        x='umap_1',
        y='umap_2',
        color='class_name',
        hover_data=hover_columns,
        opacity=0.7,
        title='Prediction-instance embeddings',
        color_discrete_sequence=px.colors.qualitative.Light24,
    )
    fig.update_traces(marker={'size': 6})
    fig.update_layout(plot_bgcolor='white')
    fig.write_html(output_folder / 'instance_embeddings_umap.html', include_plotlyjs=True)
    plot_data.drop(columns=['_source_stem', '_row_index'], errors='ignore').to_csv(
        output_folder / 'instance_embeddings_umap_data.csv', index=False
    )


def main(config_file):
    with open(config_file, 'r') as handle:
        config = yaml.safe_load(handle)

    prediction_folder = Path(config['prediction_folder'])
    analysis = get_analysis_config(config)
    output_folder = Path(analysis.get('output_folder', prediction_folder))
    output_folder.mkdir(parents=True, exist_ok=True)
    file_extension = config.get('file_extension', '.mrc')
    embedding_key = analysis.get('embedding_key', 'embeddings')
    seed = int(analysis.get('seed', 10))
    stems = discover_stems(prediction_folder, config, file_extension)

    frames = []
    embedding_parts = []
    for stem in stems:
        csv_path = prediction_folder / f'{stem}_instance_regions.csv'
        embedding_path = prediction_folder / f'{stem}_embeds.h5'
        if not csv_path.is_file():
            raise FileNotFoundError(f'Missing instance CSV: {csv_path}')
        if not embedding_path.is_file():
            raise FileNotFoundError(f'Missing embedding file: {embedding_path}')
        frame = pd.read_csv(csv_path).reset_index(drop=True)
        if len(frame) == 0:
            print(f'Skipping empty tomogram: {stem}')
            continue
        frame['_source_stem'] = stem
        frame['_row_index'] = np.arange(len(frame), dtype=np.int64)
        embedding_parts.append(load_embeddings(embedding_path, len(frame), embedding_key))
        frames.append(frame)

    if not frames:
        raise RuntimeError('No non-empty instance CSV/embedding pairs were found.')

    all_data = pd.concat(frames, ignore_index=True)
    embeddings = np.concatenate(embedding_parts, axis=0)
    n_clusters = int(analysis.get('n_clusters', 4))
    if n_clusters < 2 or n_clusters > len(embeddings):
        raise ValueError(f'n_clusters must be between 2 and {len(embeddings)}')

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=int(analysis.get('n_init', 10)),
        max_iter=int(analysis.get('max_iter', 300)),
        random_state=seed,
    )
    cluster_ids = kmeans.fit_predict(embeddings).astype(np.int32)
    cluster_names = get_cluster_names(analysis, n_clusters)
    cluster_colors = get_cluster_colors(analysis, n_clusters)
    all_data['embedding_cluster'] = cluster_ids
    all_data['semantic_class'] = cluster_ids + 1
    all_data['class_name'] = [cluster_names[index] for index in cluster_ids]

    all_public = all_data.drop(columns=['_source_stem', '_row_index'])
    all_public.to_csv(output_folder / 'all_instance_regions_clustered.csv', index=False)
    starfile.write(to_star(all_data), output_folder / 'all_instance_regions_clustered.star', overwrite=True)
    with h5py.File(output_folder / 'embedding_clusters.h5', 'w') as handle:
        handle.create_dataset('embedding_cluster', data=cluster_ids)
        handle.create_dataset('semantic_class', data=cluster_ids + 1)
        handle.create_dataset('centers', data=kmeans.cluster_centers_)
        handle.create_dataset('cluster_colors', data=cluster_colors)
        string_type = h5py.string_dtype(encoding='utf-8')
        handle.create_dataset(
            'cluster_names',
            data=np.asarray(['background'] + list(cluster_names), dtype=object),
            dtype=string_type,
        )

    write_per_tomogram_outputs(
        all_data,
        output_folder,
        prediction_folder,
        cluster_names,
        cluster_colors,
        analysis,
    )
    write_umap(all_data, embeddings, analysis, output_folder, seed)

    print(f'Processed {len(all_data)} instances from {len(frames)} tomograms.')
    print(f'Cluster counts: {np.bincount(cluster_ids, minlength=n_clusters).tolist()}')
    print(f'Outputs: {output_folder}')


if __name__ == '__main__':
    args = parse_args()
    main(args.config_file)
