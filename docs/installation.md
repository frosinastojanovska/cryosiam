# Installation Guide

This guide provides instructions for installing CryoSiam using either pip or conda, with full support for GPU-enabled PyTorch.

---

## Requirements

- Python >= 3.8
- Conda (optional, for environment-based setup)
- (Optional) NVIDIA GPU with CUDA 11.8+ drivers for GPU acceleration

---

## Option 1: Install via Conda Environment

You can also use Conda to create an isolated environment with all dependencies.

### Step 1: Create the Environment

```bash
conda env create -f environment.yml
conda activate cryosiam
```

### Step 2: Install `cryosiam`

After activating the environment:

```bash
git clone https://github.com/frosinastojanovska/cryosiam.git
cd cryosiam
pip install --no-deps .
```

---

## Option 2: Install via pip

### Step 1: Install GPU-enabled PyTorch

Before installing this package, install the correct PyTorch build with GPU support:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

If you don’t need GPU support, you can install the CPU-only version instead:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install the Package and Dependencies

Clone the repository (if applicable) and install:

```bash
git clone https://github.com/frosinastojanovska/cryosiam.git
cd cryosiam

# Recommended: use a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -r requirements.txt
```

---


## Verify Installation

To verify the CryoSiam installation, run:

```bash
cryosiam --version
```

To check that PyTorch is installed correctly and can detect your GPU:

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

---

## Troubleshooting

- **CUDA errors**: Make sure your system has the correct NVIDIA drivers and CUDA version.
- **Missing packages**: Double-check that you followed the correct pip or conda install steps.
- **Conflicts**: It's recommended to use a clean virtual environment or conda environment.

---

## Need Help?

If you run into issues, feel free to open an issue on the [GitHub repository](https://github.com/frosinastojanovska/cryosiam/issues).

---


##  Don't Have Conda Installed?

If you don't have `conda` installed yet, we recommend using **Miniforge** (lightweight) or **Anaconda** (full-featured).

### Option 1: Install Miniforge (recommended)

Miniforge is a minimal installer for Conda that supports `conda-forge` by default.

**Linux / macOS:**

```bash
# Download and install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

**Windows:**

1. Download the latest installer from:  
   https://github.com/conda-forge/miniforge/releases

2. Run the `.exe` installer and follow the setup.

### Option 2: Install Anaconda (Full Distribution)

Anaconda includes Conda, Python, and hundreds of data science packages.

Download from: https://www.anaconda.com/products/distribution

---

After installing Miniforge or Anaconda, return to the main Conda installation steps above.
