from setuptools import setup, find_packages

setup(
    name="cryosiam",
    version="1.0",
    author="Frosina Stojanovska",
    author_email="stojanovska.frose@gmail.com",
    description="CryoSiam: Deep Learning-based Cryo-ET Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/frosinastojanovska/cryosiam",
    license="GPL-3.0-only",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "cryosiam = cryosiam.cli:main"
        ]
    },
    install_requires=[
        "edt",
        "h5py",
        "itk",
        "lightning",
        "monai>=1.3.1",
        "mrcfile",
        "numpy",
        "pandas",
        "scipy",
        "plotly",
        "PyYAML",
        "scikit-learn",
        "scikit-image",
        "torch>=2.1.0",
        "torchvision",
        "starfile",
        "umap-learn"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
