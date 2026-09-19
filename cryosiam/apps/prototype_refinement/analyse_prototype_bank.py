import os
import umap
import yaml
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import torch.nn.functional as F

from cryosiam.utils import parser_helper
from cryosiam.apps.prototype_refinement.utils import load_class_names


def print_matrix(matrix, names, title):
    matrix = matrix.cpu().numpy()
    print(f'\n{title}')
    print(''.ljust(22), end='')
    for name in names:
        print(f'{name[:12]:>13}', end='')
    print()
    for i, name in enumerate(names):
        print(f'{name[:20]:<22}', end='')
        for j in range(len(names)):
            print(f'{matrix[i, j]:13.3f}', end='')
        print()


def save_matrix(matrix, names, filename):
    pd.DataFrame(matrix.cpu().numpy(), index=names, columns=names).to_csv(filename)


def compute_class_mean_similarity(prototypes):
    class_prototypes = F.normalize(prototypes.mean(dim=1), dim=-1)
    return class_prototypes @ class_prototypes.T


def compute_max_prototype_similarity(prototypes):
    n_classes = prototypes.shape[0]
    similarity = torch.zeros(n_classes, n_classes, device=prototypes.device)
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise = prototypes[i] @ prototypes[j].T
            similarity[i, j] = pairwise.max()
    return similarity


def compute_mean_nearest_similarity(prototypes):
    n_classes = prototypes.shape[0]
    similarity = torch.zeros(n_classes, n_classes, device=prototypes.device)
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise = prototypes[i] @ prototypes[j].T
            i_to_j = pairwise.max(dim=1).values.mean()
            j_to_i = pairwise.max(dim=0).values.mean()
            similarity[i, j] = 0.5 * (i_to_j + j_to_i)
    return similarity


def analyze_within_class_similarity(bank, class_names, output_folder):
    rows = []

    print('\nWITHIN-CLASS SUB-PROTOTYPE SIMILARITY')
    print('(off-diagonal cosine similarities; 1.0 means identical prototypes)')

    for cls, name in enumerate(class_names):
        protos = bank[cls]
        pairwise = protos @ protos.T

        k = pairwise.shape[0]
        off_diag_mask = ~torch.eye(k, dtype=torch.bool, device=pairwise.device)
        off_diag = pairwise[off_diag_mask]

        pairwise_no_diag = pairwise.clone()
        pairwise_no_diag.fill_diagonal_(-float('inf'))
        nearest_other = pairwise_no_diag.max(dim=1).values

        row = {
            'class': name,
            'min_pairwise': off_diag.min().item(),
            'mean_pairwise': off_diag.mean().item(),
            'max_pairwise': off_diag.max().item(),
            'mean_nearest_other': nearest_other.mean().item(),
            'min_nearest_other': nearest_other.min().item(),
            'max_nearest_other': nearest_other.max().item()
        }
        rows.append(row)

        mean_cosine_distance = (1.0 - off_diag).mean().item()

        print(
            f'  {name}: '
            f'pairwise [{row["min_pairwise"]:.8f}, {row["mean_pairwise"]:.8f}, {row["max_pairwise"]:.8f}]  '
            f'nearest-other mean={row["mean_nearest_other"]:.8f}'
            f'  mean cosine distance={mean_cosine_distance:.8e}'
        )

        safe_name = name.replace('/', '_').replace(' ', '_')
        pd.DataFrame(pairwise.cpu().numpy()).to_csv(
            os.path.join(output_folder, f'prototype_within_{safe_name}.csv'),
            index=False)

    pd.DataFrame(rows).to_csv(
        os.path.join(output_folder, 'prototype_within_class_summary.csv'),
        index=False)


def save_umap(prototypes, names, filename):
    n_classes, k_per_class, feat_dim = prototypes.shape
    features = prototypes.reshape(-1, feat_dim).cpu().numpy()
    classes = np.repeat(names, k_per_class)
    prototype_id = np.tile(np.arange(k_per_class), n_classes)

    n_neighbors = min(10, features.shape[0] - 1)
    reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=n_neighbors,
                        min_dist=0.0, random_state=10, n_jobs=1)
    projection = reducer.fit_transform(features)

    data = pd.DataFrame({'x': projection[:, 0],
                         'y': projection[:, 1],
                         'class': classes,
                         'prototype': prototype_id})

    fig = px.scatter(data, x='x', y='y', color='class', hover_data=data.columns,
                     opacity=0.8, color_discrete_sequence=px.colors.qualitative.Light24)
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(showline=True, linecolor='black', linewidth=1)
    fig.update_yaxes(showline=True, linecolor='black', linewidth=1)
    fig.write_html(filename)
    data.to_csv(filename.split('.html')[0] + '_data.csv', index=False)


def main(config_file_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg['trained_model']
    class_names = cfg.get('class_names') or load_class_names(checkpoint_path)
    output_folder = cfg.get('output_folder', cfg.get('prediction_folder', os.path.dirname(checkpoint_path)))
    os.makedirs(output_folder, exist_ok=True)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)['state_dict']
    if 'proto_bank.prototypes' not in state:
        raise RuntimeError('No prototype bank in the checkpoint.')

    bank = F.normalize(state['proto_bank.prototypes'].float(), dim=-1)

    expected_classes = len(class_names)
    if bank.shape[0] == expected_classes + 1:
        print('Detected legacy prototype bank with background row; ignoring background for analysis.')
        bank = bank[1:]
    elif bank.shape[0] != expected_classes:
        raise ValueError(f'Expected {expected_classes} foreground prototype classes '
                         f'or {expected_classes + 1} classes including legacy background, '
                         f'got {bank.shape[0]}')

    names = list(class_names)

    analyze_within_class_similarity(bank, class_names, output_folder)

    print(f'Prototype bank shape: {tuple(bank.shape)}')
    print(f'Analyzing: {names}')

    mean_similarity = compute_class_mean_similarity(bank)
    max_similarity = compute_max_prototype_similarity(bank)
    nearest_similarity = compute_mean_nearest_similarity(bank)

    print_matrix(mean_similarity, names, 'COSINE SIMILARITY BETWEEN CLASS-MEAN PROTOTYPES')
    print_matrix(max_similarity, names, 'MAXIMUM COSINE SIMILARITY BETWEEN ANY TWO PROTOTYPES')
    print_matrix(nearest_similarity, names, 'MEAN NEAREST-PROTOTYPE SIMILARITY')

    save_matrix(mean_similarity, names, os.path.join(output_folder, 'prototype_class_mean_similarity.csv'))
    save_matrix(max_similarity, names, os.path.join(output_folder, 'prototype_max_similarity.csv'))
    save_matrix(nearest_similarity, names, os.path.join(output_folder, 'prototype_mean_nearest_similarity.csv'))

    rows = []
    if len(names) > 1:
        nearest_no_diag = nearest_similarity.clone()
        nearest_no_diag.fill_diagonal_(-float('inf'))
        print('\nClosest competing class for each class:')
        for i, name in enumerate(names):
            value, j = nearest_no_diag[i].max(dim=0)
            competitor = names[j.item()]
            print(f'  {name}: {competitor} ({value.item():.3f})')
            rows.append({'class': name, 'closest_class': competitor, 'similarity': value.item()})
    pd.DataFrame(rows, columns=['class', 'closest_class', 'similarity']).to_csv(
        os.path.join(output_folder, 'prototype_closest_classes.csv'), index=False)

    umap_file = os.path.join(output_folder, 'prototype_umap.html')
    save_umap(bank, names, umap_file)
    print(f'\nSaved UMAP: {umap_file}')
    print(f'Saved similarity matrices to: {output_folder}')


if __name__ == '__main__':
    parser = parser_helper('Prototype bank separation analysis')
    args = parser.parse_args()
    main(args.config_file)
