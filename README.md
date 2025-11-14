# CryoSiam 


<img src="imgs/CryoSiam_logo.png" width=200 height=200>

Self-supervised deep learning framework that works on dense and subtomogram levels. The dense part of the framework is additionally trained on subtasks for tomogram denoising, tomogram semantic and instance segmentation. 

For installation and usage instructions, please visit the [documentation page](https://frosinastojanovska.github.io/cryosiam-docs/).

## Citation

[Preprint](https://www.biorxiv.org/content/10.1101/2025.11.11.687379v1)

```text
@article {Stojanovska2025.11.11.687379,
	author = {Stojanovska, Frosina and Sanchez, Ricardo M. and Jensen, Rasmus K. and Mahamid, Julia and Kreshuk, Anna and Zaugg, Judith B},
	title = {CryoSiam: self-supervised representation learning for automated analysis of cryo-electron tomograms},
	elocation-id = {2025.11.11.687379},
	year = {2025},
	doi = {10.1101/2025.11.11.687379},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Cryo-electron tomography (cryo-ET) enables visualization of macromolecular complexes in their native cellular context, but interpretation remains challenging due to high noise levels, missing information, and lack of ground-truth data. Here, we present CryoSiam (CRYO-electron tomography SIAMese networks), an open-source framework for self-supervised representation learning in cryo-ET. CryoSiam learns hierarchical representations of tomographic data spanning both voxel-level and subtomogram-level information. To train CryoSiam, we generated CryoETSim (CRYO-Electron Tomography SIMulated), a synthetic dataset that systematically models defocus variation, sample thickness, and molecular crowding. CryoSiam trained models transfer directly to experimental data without fine-tuning and support key aspects of cryo-ET data analysis, including tomogram denoising, segmentation of subcellular structures, and macromolecular detection and identification across both prokaryotic and eukaryotic systems. Publicly available pretrained models and the CryoETSim dataset provide a foundation for scalable and automated cryo-ET analysis.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2025/11/12/2025.11.11.687379},
	eprint = {https://www.biorxiv.org/content/early/2025/11/12/2025.11.11.687379.full.pdf},
	journal = {bioRxiv}
}
```