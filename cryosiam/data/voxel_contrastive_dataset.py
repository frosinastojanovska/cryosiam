import os
import json
import random
import collections
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


class VoxelContrastiveDataset(Dataset):
    """
    Episodic dataset: each item is one class + a support patch + a
    different query patch, BOTH drawn from the same "contains this
    class" pool (self.class_to_patches[cls]) -- so support and query
    are guaranteed, by construction, to both contain the target class
    per the manifest.

    Train: episodes are re-sampled (weighted by class_weights) every
    epoch via rebuild_index(), called from on_train_epoch_start.

    Val: episodes (class, support_idx, query_idx) are fixed ONCE at
    construction time and never re-randomized -- keeps val_loss/val_dice
    stable and comparable epoch-to-epoch. Capped to `samples_per_class`
    per class rather than being fully exhaustive.

    Classes with only 1 patch reuse that patch for both support and
    query (two independent stochastic transform() calls) rather than
    being dropped -- keeps every class present every epoch.
    """

    def __init__(self,
                 search_files,
                 search_transform=None,
                 samples_per_class=None,
                 is_val=False,
                 min_foreground=10,
                 exclude_labels=None,
                 use_distance_loss=False,
                 distance_clip=8):

        self.search_files = search_files
        self.search_transform = search_transform
        self.samples_per_class = samples_per_class
        self.is_val = is_val
        self.min_foreground = min_foreground
        self.exclude_labels = set(exclude_labels or [])
        self.use_distance_loss = bool(use_distance_loss)
        self.distance_clip = float(distance_clip)

        if self.use_distance_loss and self.distance_clip <= 0:
            raise ValueError('distance_clip must be greater than 0.')

        self.class_to_patches = self._build_class_to_patches()
        self.unique_labels = sorted(self.class_to_patches.keys())
        self.class_weights = {l: 1.0 for l in self.unique_labels}

        self.episodic_labels = list(self.unique_labels)
        self._single_patch_labels = sorted(
            l for l, p in self.class_to_patches.items() if len(p) == 1)

        self._index = []
        self._val_episodes = []

        if self.is_val:
            self._build_val_episodes()
        else:
            self.rebuild_index()

        self._print_summary()

    def _build_class_to_patches(self) -> dict:
        mask_dir = os.path.dirname(self.search_files[0]['mask'])
        patches_root = os.path.dirname(mask_dir)
        manifest_path = os.path.join(patches_root, 'class_manifest.json')

        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f'class_manifest.json not found at {manifest_path}. '
                f'Re-run the patching script to generate it.')

        with open(manifest_path) as f:
            manifest = json.load(f)

        def _base(p):
            return os.path.basename(p).split('.')[0]

        name_to_idx = {_base(sf['mask']): i
                       for i, sf in enumerate(self.search_files)}
        class_to_patches = collections.defaultdict(list)
        unmatched = 0

        for base_name, classes in manifest.items():
            idx = name_to_idx.get(base_name)
            if idx is None:
                unmatched += 1
                continue
            for label in classes:
                if label in self.exclude_labels:
                    continue
                class_to_patches[label].append(idx)

        if unmatched:
            print(f'  Note: {unmatched} manifest entries had no matching '
                  f'search_file (normal for train/val subsets).')
        print(f'  Loaded class→patch index '
              f'({sum(len(v) for v in class_to_patches.values())} entries, '
              f'{len(class_to_patches)} classes)')
        return dict(class_to_patches)

    def rebuild_index(self):
        if self.is_val:
            return

        covered = self.episodic_labels
        quota = self.samples_per_class or 50
        total_w = sum(self.class_weights.get(l, 1.0) for l in covered)
        mean_w = total_w / max(len(covered), 1)

        index = []
        for label in covered:
            weight = self.class_weights.get(label, 1.0) / mean_w
            n = max(10, int(quota * weight))
            index.extend([label] * n)

        random.shuffle(index)
        self._index = index

    def _build_val_episodes(self):
        quota = self.samples_per_class or 20
        rng = random.Random(42)

        episodes = []
        for label in self.episodic_labels:
            patches = self.class_to_patches[label]
            n = min(quota, len(patches))
            chosen_queries = rng.sample(patches, n)
            for q in chosen_queries:
                others = [p for p in patches if p != q]
                support = rng.choice(others) if others else q
                episodes.append((label, support, q))

        self._val_episodes = episodes

    def _sample_support_query(self, cls):
        patches = self.class_to_patches[cls]
        if len(patches) == 1:
            return patches[0], patches[0]
        return random.sample(patches, 2)

    def update_class_weights(self, class_metric: dict):
        for label in self.unique_labels:
            m = class_metric.get(label, None)
            self.class_weights[label] = (1.0 if m is None else max(0.1, 1.0 - m))

    def __len__(self):
        return len(self._val_episodes) if self.is_val else len(self._index)

    def _load(self, file_idx):
        sample = self.search_transform(dict(self.search_files[file_idx]))
        mask = sample['mask']
        for excl in self.exclude_labels:
            mask = mask.masked_fill(mask == excl, 0)
        return sample['image'], mask

    def _signed_distance(self, mask, class_id):
        spatial_mask = mask[0] if mask.ndim == 4 and mask.shape[0] == 1 else mask
        foreground = (spatial_mask == class_id).detach().cpu().numpy().astype(bool)

        if not foreground.any():
            return torch.full(tuple(spatial_mask.shape), -1.0, dtype=torch.float32)

        inside = distance_transform_edt(foreground)
        outside = distance_transform_edt(~foreground)
        distance = inside - outside
        distance = np.clip(distance / self.distance_clip, -1, 1)

        return torch.from_numpy(distance.astype(np.float32))

    def __getitem__(self, idx):
        if self.is_val:
            cls, support_idx, query_idx = self._val_episodes[idx]
        else:
            cls = self._index[idx]
            support_idx, query_idx = self._sample_support_query(cls)

        support_image, support_mask = self._load(support_idx)
        query_image, query_mask = self._load(query_idx)

        result = {
            'support_image': support_image, 'support_mask': support_mask,
            'query_image': query_image, 'query_mask': query_mask,
            'class_id': cls,
        }

        if self.use_distance_loss:
            result['query_distance'] = self._signed_distance(query_mask, cls)

        return result

    def _print_summary(self):
        mode = 'val (fixed episodes)' if self.is_val else 'train'
        print(f'VoxelContrastiveDataset [{mode}]:')
        print(f'  {len(self.unique_labels)} classes, all episodic')
        if self._single_patch_labels:
            print(f'  {len(self._single_patch_labels)} classes have exactly 1 patch '
                  f'(support/query reuse it, weaker signal): {self._single_patch_labels}')
        print(f'  {len(self)} episodes/epoch '
              f'(samples_per_class={self.samples_per_class})')

        counts = [len(self.class_to_patches.get(l, []))
                  for l in self.unique_labels
                  if self.class_to_patches.get(l)]
        if counts:
            print(f'  Patches per class: min={min(counts)}, '
                  f'median={int(np.median(counts))}, max={max(counts)}')


def episodic_collate(batch):
    result = {
        'support_image': torch.stack([b['support_image'] for b in batch]),
        'support_mask': torch.stack([b['support_mask'] for b in batch]),
        'query_image': torch.stack([b['query_image'] for b in batch]),
        'query_mask': torch.stack([b['query_mask'] for b in batch]),
        'class_id': torch.tensor([b['class_id'] for b in batch], dtype=torch.long),
    }

    if 'query_distance' in batch[0]:
        result['query_distance'] = torch.stack(
            [b['query_distance'] for b in batch])

    return result
