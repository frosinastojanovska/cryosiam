# CryoSiam Semantic Prediction

This guide explains how to run **semantic segmentation prediction** using CryoSiam. You'll use a YAML config file to specify your model, data, and output options.

---

## How to Run

Use the CryoSiam CLI with the `semantic_predict` subcommand:

```bash
cryosiam semantic_predict --config_file=configs/semantic_predict.yaml
```

This command loads the configuration, prepares the model and data, and saves the predicted segmentation.

---

## Example Configuration File

Below is a full example of a configuration file in YAML format. You can create this file at any location and pass the path to it with the `--config_file` argument.

<details>
<summary><strong>Click to expand the example YAML</strong></summary>

```yaml
data_folder: '/scratch/cryosiam/tomograms/'
prediction_folder: '/scratch/cryosiam/experiments/dense_simsiam_semantic/predictions'
trained_model: '/scratch/cryosiam/experiments/dense_simsiam_semantic/model/model-best.ckpt'
file_extension: '.mrc'

test_files: [ 'TS_56_6.80Apx.mrc', 'TS_61_6.80Apx.mrc' ]

save_internal_files: False
save_original_file_extension: False

parameters:
  gpu_devices: 1
  data:
    patch_size: [ 128, 128, 128 ]
    min: 0
    max: 1
    mean: 0
    std: 1
  network:
    in_channels: 1
    spatial_dims: 3
    out_channels: 1
    postprocessing: True
    postprocessing_sizes: [ -1, 10000, -1, -1, -1 ]

hyper_parameters:
  batch_size: 5
```

</details>

This configuration tells CryoSiam:
- Where the model is located
- Which data to use
- Where the prediction should be saved

---

## Configuration Fields Explained

| Field                          | Description                                                                                               |
|--------------------------------|-----------------------------------------------------------------------------------------------------------|
| `data_folder`                  | Path to the folder containing the denoised/noisy tomograms                                                |
| `prediction_folder`            | Path to the folder where the predictions will be saved                                                    |
| `trained_model`                | Path to the trained model checkpoint `.ckpt`                                                              |
| `file_extension`               | File extension of the tomograms, can be `.mrc` or `.rec`                                                  |
| `test_files`                   | `null` if all of the tomograms in `data_folder` will be predicted, else a list of selected tomogram names |
| `save_internal_files`          | If the probability and distance map will be saved in .h5 file                                             |
| `save_original_file_extension` | If `False` the output is saved in `.h5` format, else it will be saved in the `file_extension` format      |

---

### `parameters`

| Field        | Description                                          |
|--------------|------------------------------------------------------|
| `input_dir`  | Folder containing `.mrc` or `.tiff` tomograms        |
| `file_type`  | Format of input tomograms: `mrc` or `tiff`           |
| `voxel_size` | Size of one voxel in Ångströms (used for scaling)    |
| `normalize`  | Normalize intensity before prediction (`true/false`) |

---

### `hyper_parameters`

| Field                   | Description                                     |
|-------------------------|-------------------------------------------------|
| `batch_size`            | Batch size for prediction                       |

---

## Model Selection

CryoSiam provides pretrained models for semantic segmentation. You can download and use them by specifying the correct model file in your configuration.

### Available Pretrained Models

#### 1. **General Semantic Segmentation Model**

- **Purpose:** Identifies broad structural classes in cryo-ET tomograms.
- **Classes:** 
  - Membrane
  - Particles
  - Microtubules
  - Actin
  - Filaments (DNA/RNA)
- **Use Case:** Suitable for exploratory annotation and segmentation of standard biological samples.

#### 2. **Target Complex Segmentation Model**

- **Purpose:** Focused segmentation model for identifying specific protein complexes as described in the CryoSiam publication.
- **Use Case:** Use this model when your goal is to identify specific known structures, such as ribosomes or specific viral components.

---

## Troubleshooting

| Problem                      | Suggested Fix |
|------------------------------|---------------|
| `FileNotFoundError`          | Check paths to model or input directory |
| `CUDA out of memory`         | Lower `batch_size` or use `device: cpu` |

---

## Related Commands

- [CryoSiam Denoising](../denoising/prediction.md)
- [CryoSiam Instance Prediction](../instance/prediction.md)
- [Installation Guide](../installation.md)

---

## Need Help?

- Check the [FAQ](../faq.md)  
- Or [open an issue on GitHub](https://github.com/frosinastojanovska/cryosiam/issues)
