import os
import math
import random



def patch_train_val_split(images_folder, masks_folder, file_ext='.mrc',
                          ratio=0.1, centers_folder=None):

    all_patches = sorted(
        f for f in os.listdir(images_folder)
        if f.endswith(file_ext)
    )

    if not all_patches:
        raise ValueError(f'No {file_ext} files found in {images_folder}')

    if not 0 <= ratio < 1:
        raise ValueError('validation ratio must be in [0, 1)')

    def tomo_name(filename):
        return filename[:-len(file_ext)].rsplit('_z', 1)[0]

    # Collect valid patches first.
    tomo_groups = {}
    missing = 0

    for patch in all_patches:
        entry = {'image': os.path.join(images_folder, patch)}

        if masks_folder is not None:
            mask_path = os.path.join(masks_folder, patch)
            if not os.path.exists(mask_path):
                missing += 1
                continue
            entry['mask'] = mask_path

        if centers_folder is not None:
            center_patch = os.path.splitext(patch)[0] + '.npz'
            center_path = os.path.join(centers_folder, center_patch)
            if not os.path.exists(center_path):
                missing += 1
                continue
            entry['center'] = center_path

        tomo_groups.setdefault(tomo_name(patch), []).append(entry)

    if missing:
        print(f'  WARNING: {missing} patches skipped '
              '(missing masks or centers)')

    if not tomo_groups:
        raise ValueError('No valid patches found for training.')

    tomo_names = sorted(tomo_groups.keys())
    random.shuffle(tomo_names)

    # ---------------------------------------------------------
    # Single tomogram: split by patches.
    # ---------------------------------------------------------
    if len(tomo_names) == 1:

        patches = tomo_groups[tomo_names[0]].copy()
        random.shuffle(patches)

        if ratio == 0:
            train_files = patches
            val_files = []
        else:
            if len(patches) < 2:
                raise ValueError(
                    'At least 2 valid patches are required for '
                    'single-tomogram train/validation splitting.'
                )

            n_val = max(1, math.ceil(len(patches) * ratio))
            n_val = min(n_val, len(patches) - 1)

            train_files = patches[:-n_val]
            val_files = patches[-n_val:]

        print('Single tomogram detected — splitting by patches.')

    # ---------------------------------------------------------
    # Multiple tomograms: preserve tomogram-level splitting.
    # ---------------------------------------------------------
    else:

        if ratio == 0:
            train_tomos, val_tomos = tomo_names, []
        else:
            n_val = max(1, math.ceil(len(tomo_names) * ratio))
            n_val = min(n_val, len(tomo_names) - 1)

            train_tomos = tomo_names[:-n_val]
            val_tomos = tomo_names[-n_val:]

        train_files = [
            entry
            for tomo in train_tomos
            for entry in tomo_groups[tomo]
        ]

        val_files = [
            entry
            for tomo in val_tomos
            for entry in tomo_groups[tomo]
        ]

        print(f'Tomograms — train: {len(train_tomos)}, '
              f'val: {len(val_tomos)}')

    print(f'Patches   — train: {len(train_files)}, '
          f'val: {len(val_files)}')

    return train_files, val_files


def supervised_train_val_split(data_root, seg_root, files=None, ratio=0.1, val_files=None, patches_folder=None,
                               input_key_name='image', output_key_name='labels', file_ext='.mrc'):
    train_data = []
    val_data = []
    data_files = os.listdir(data_root) if files is None else files
    if val_files is None:
        random.shuffle(data_files)
        ratio_ind = math.ceil(len(data_files) * ratio)
        train_files = data_files[:-ratio_ind]
        val_files = data_files[-ratio_ind:]
    else:
        train_files = data_files
        val_files = val_files
    if patches_folder is not None:
        patches_files = [os.path.basename(x) for x in os.listdir(os.path.join(patches_folder, input_key_name))]

    for idx, file in enumerate(train_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + '_')]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                train_data.append({input_key_name: os.path.join(patches_folder, input_key_name, patch_file),
                                   output_key_name: os.path.join(patches_folder, output_key_name, patch_file)})
        else:
            train_data.append({input_key_name: os.path.join(data_root, file),
                               output_key_name: os.path.join(seg_root, file)})

    for idx, file in enumerate(val_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + '_')]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                val_data.append({input_key_name: os.path.join(patches_folder, input_key_name, patch_file),
                                 output_key_name: os.path.join(patches_folder, output_key_name, patch_file)})
        else:
            val_data.append({input_key_name: os.path.join(data_root, file),
                             output_key_name: os.path.join(seg_root, file)})

    return train_data, val_data


def supervised_instance_train_val_split(data_root, out_root, noisy_data_root=None, files=None, ratio=0.1,
                                        val_files=None, patches_folder=None, file_ext='.mrc'):
    train_data = []
    val_data = []
    data_files = os.listdir(data_root) if files is None else files
    if val_files is None:
        random.shuffle(data_files)
        ratio_ind = math.ceil(len(data_files) * ratio)
        train_files = data_files[:-ratio_ind]
        val_files = data_files[-ratio_ind:]
    else:
        train_files = data_files
        val_files = val_files
    if patches_folder is not None:
        patches_files = [os.path.basename(x) for x in os.listdir(os.path.join(patches_folder, 'image'))]

    for idx, file in enumerate(train_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + '_')]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    train_data.append({'image': os.path.join(patches_folder, 'image', patch_file),
                                       'noisy_image': os.path.join(patches_folder, 'noisy_image', patch_file),
                                       'foreground': os.path.join(patches_folder, 'foreground', patch_file),
                                       'distances': os.path.join(patches_folder, 'distances', patch_file),
                                       'boundaries': os.path.join(patches_folder, 'boundaries', patch_file)})
                else:
                    train_data.append({'image': os.path.join(patches_folder, 'image', patch_file),
                                       'foreground': os.path.join(patches_folder, 'foreground', patch_file),
                                       'distances': os.path.join(patches_folder, 'distances', patch_file),
                                       'boundaries': os.path.join(patches_folder, 'boundaries', patch_file)})
        else:
            if noisy_data_root:
                train_data.append({'image': os.path.join(data_root, file),
                                   'noisy_image': os.path.join(noisy_data_root, file),
                                   'foreground': os.path.join(out_root, 'foreground', file),
                                   'distances': os.path.join(out_root, 'distances', file),
                                   'boundaries': os.path.join(out_root, 'boundaries', file)})
            else:
                train_data.append({'image': os.path.join(data_root, file),
                                   'foreground': os.path.join(out_root, 'foreground', file),
                                   'distances': os.path.join(out_root, 'distances', file),
                                   'boundaries': os.path.join(out_root, 'boundaries', file)})

    for idx, file in enumerate(val_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + '_')]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    val_data.append({'image': os.path.join(patches_folder, 'image', patch_file),
                                     'noisy_image': os.path.join(patches_folder, 'noisy_image', patch_file),
                                     'foreground': os.path.join(patches_folder, 'foreground', patch_file),
                                     'distances': os.path.join(patches_folder, 'distances', patch_file),
                                     'boundaries': os.path.join(patches_folder, 'boundaries', patch_file)})
                else:
                    val_data.append({'image': os.path.join(patches_folder, 'image', patch_file),
                                     'foreground': os.path.join(patches_folder, 'foreground', patch_file),
                                     'distances': os.path.join(patches_folder, 'distances', patch_file),
                                     'boundaries': os.path.join(patches_folder, 'boundaries', patch_file)})
        else:
            if noisy_data_root:
                val_data.append({'image': os.path.join(data_root, file),
                                 'noisy_image': os.path.join(noisy_data_root, file),
                                 'foreground': os.path.join(out_root, 'foreground', file),
                                 'distances': os.path.join(out_root, 'distances', file),
                                 'boundaries': os.path.join(out_root, 'boundaries', file)})
            else:
                val_data.append({'image': os.path.join(data_root, file),
                                 'foreground': os.path.join(out_root, 'foreground', file),
                                 'distances': os.path.join(out_root, 'distances', file),
                                 'boundaries': os.path.join(out_root, 'boundaries', file)})

    return train_data, val_data


def supervised_semantic_train_val_split(data_root, seg_root, out_root, noisy_data_root=None, files=None, ratio=0.1,
                                        val_files=None, patches_folder=None, file_ext='.mrc',
                                        use_distances=True, use_skeletons=False):
    train_data = []
    val_data = []
    data_files = os.listdir(data_root) if files is None else files
    if val_files is None:
        random.shuffle(data_files)
        ratio_ind = math.ceil(len(data_files) * ratio)
        train_files = data_files[:-ratio_ind]
        val_files = data_files[-ratio_ind:]
    else:
        train_files = data_files
        val_files = val_files
    if patches_folder is not None:
        patches_files = [os.path.basename(x) for x in os.listdir(os.path.join(patches_folder, 'image'))]

    def build_entry(file, image_root, labels_root, npz_root, noisy_root=None):
        entry = {'image': os.path.join(image_root, file),
                 'labels': os.path.join(labels_root, file)}
        if noisy_root:
            entry['noisy_image'] = os.path.join(noisy_root, file)
        npz_name = f'{file.split(file_ext)[0]}.npz'
        if use_distances:
            entry['distances'] = os.path.join(npz_root, 'distances', npz_name)
        if use_skeletons:
            entry['skeletons'] = os.path.join(npz_root, 'skeletons', npz_name)
        return entry

    for split_files, split_data in ((train_files, train_data), (val_files, val_data)):
        for file in split_files:
            if patches_folder is not None:
                file_prefix = file.split(file_ext)[0]
                patch_files = [x for x in patches_files if x.startswith(file_prefix + '_')]
                for patch_file in patch_files:
                    patch_file = os.path.basename(patch_file)
                    split_data.append(build_entry(
                        patch_file,
                        os.path.join(patches_folder, 'image'),
                        os.path.join(patches_folder, 'labels'),
                        patches_folder,
                        os.path.join(patches_folder, 'noisy_image') if noisy_data_root else None))
            else:
                split_data.append(build_entry(file, data_root, seg_root, out_root, noisy_data_root))

    return train_data, val_data
