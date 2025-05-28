from setuptools import setup, find_packages

setup(
    name="cryosiam",
    version="0.1.0",
    author="Frosina Stojanovska",
    author_email="stojanovska.frose@gmail.com",
    description="CryoSiam: Deep Learning-based Cryo-ET Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/frosinastojanovska/cryosiam",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "cryosiam = cryosiam.cli:main"
        ]
    },
    install_requires=[
        "edt==2.4.1",
        "elf==0.5.2",
        "h5py==3.11.0",
        "itk==5.3.0",
        "lightning==2.1.3",
        "monai==1.3.2",
        "mrcfile==1.5.3",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scipy==1.9.1",
        "plotly==5.24.1",
        "PyYAML==6.0.2",
        "scikit_learn==1.3.2",
        "scipy==1.9.1",
        "scikit-image==0.20.0",
        "torch==2.1.2",
        "torchvision==0.16.2",
        "umap-learn==0.5.7"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
