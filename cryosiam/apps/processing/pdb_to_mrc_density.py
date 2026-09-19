import argparse
import os.path
from pathlib import Path

import mrcfile
import numpy as np

ATOM_DICT = {
    "H": 0.0,
    "C": 6.0 + 1.3,
    "N": 7.0 + 1.1,
    "O": 8.0 + 0.2,
    "P": 15.0,
    "S": 16.0 + 0.6,
    "MG": 12.0,
    "ZN": 30.0,
    "MN": 25.0,
    "F": 9.0,
    "CL": 17.0,
    "CA": 20.0,
}

KNOWN_ELEMENTS = np.array(list(ATOM_DICT.keys()))
ELEMENT_INTENSITIES = np.array(list(ATOM_DICT.values()), dtype=np.float32)
ELEMENT_TO_IDX = {e: i for i, e in enumerate(KNOWN_ELEMENTS)}


def _infer_element_from_pdb_line(line: str) -> str:
    """Infer element symbol from a PDB ATOM/HETATM line."""
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element

    atom_name = line[12:16].strip().upper()
    atom_name = "".join(c for c in atom_name if c.isalpha())

    if not atom_name:
        return ""

    # Handle common PDB naming:
    # first two letters if valid element, otherwise first letter.
    if len(atom_name) >= 2 and atom_name[:2] in ELEMENT_TO_IDX:
        return atom_name[:2]
    return atom_name[:1]


def parse_pdb(pdb_path: str) -> dict:
    """Parse a PDB file and return {'atoms': array[str], 'coords': array[3, N]}."""
    atoms = []
    coords = []

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            element = _infer_element_from_pdb_line(line)
            atoms.append(element)
            coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No valid ATOM/HETATM coordinates found in {pdb_path}")

    return {
        "atoms": np.asarray(atoms),
        "coords": np.asarray(coords, dtype=np.float32).T,  # (3, N)
    }


def parse_cif(cif_path: str) -> dict:
    """Parse a CIF/mmCIF file and return {'atoms': array[str], 'coords': array[3, N]}."""
    with open(cif_path, "r") as f:
        lines = f.readlines()

    head_start = [i for i, l in enumerate(lines) if l.startswith("_atom_site.group_PDB")]
    head_end = [i for i, l in enumerate(lines) if l.startswith("_atom_site.pdbx_PDB_model_num")]
    loop_ends = [i for i, l in enumerate(lines) if l.startswith("#")]

    atoms_all = []
    coords_all = []

    for hs, he in zip(head_start, head_end):
        header = [l.strip().replace("_atom_site.", "") for l in lines[hs:he + 1]]
        loop_end = next((le for le in loop_ends if le > he), len(lines))

        atom_lines = " ".join(l.strip() for l in lines[he + 1:loop_end])
        tokens = atom_lines.split()
        n_cols = len(header)

        if n_cols == 0 or len(tokens) % n_cols != 0:
            continue

        rows = np.array(tokens, dtype=object).reshape(-1, n_cols)
        col = {h: i for i, h in enumerate(header)}

        required = {"type_symbol", "Cartn_x", "Cartn_y", "Cartn_z"}
        if not required.issubset(col):
            continue

        atoms = []
        coords = []

        for r in rows:
            try:
                atoms.append(str(r[col["type_symbol"]]).upper())
                coords.append(
                    [
                        float(r[col["Cartn_x"]]),
                        float(r[col["Cartn_y"]]),
                        float(r[col["Cartn_z"]]),
                    ]
                )
            except ValueError:
                continue

        atoms_all.extend(atoms)
        coords_all.extend(coords)

    if not coords_all:
        raise ValueError(f"No valid atom coordinates found in {cif_path}")

    return {
        "atoms": np.asarray(atoms_all),
        "coords": np.asarray(coords_all, dtype=np.float32).T,  # (3, N)
    }


def parse_structure(structure_path: str) -> dict:
    """Parse PDB/CIF/mmCIF into one atom/coordinate dictionary."""
    path = Path(structure_path)
    suffix = path.suffix.lower()

    if suffix in {".cif", ".mmcif"}:
        return parse_cif(structure_path)
    return parse_pdb(structure_path)


def trim_volume(vol: np.ndarray) -> np.ndarray:
    """Trim empty border planes from a volume."""
    nonzero = np.nonzero(vol)
    if len(nonzero[0]) == 0:
        return vol

    slices = tuple(slice(axis.min(), axis.max() + 1) for axis in nonzero)
    return vol[slices]


def build_volume(data: dict, pixel_size: float) -> np.ndarray:
    """
    Build one EM density volume from parsed atomic data and trim empty borders.

    Args:
        data: {'atoms': array[str], 'coords': array[3, N]}
        pixel_size: voxel size in angstroms

    Returns:
        3D numpy array
    """
    atoms = data["atoms"]
    coords = data["coords"]  # (3, N)

    if coords.shape[0] != 3:
        raise ValueError(f"Expected coords shape (3, N), got {coords.shape}")

    # Keep only supported elements
    elements = np.array([str(a).upper() for a in atoms])
    valid_mask = np.isin(elements, KNOWN_ELEMENTS)

    if not valid_mask.any():
        raise ValueError("No supported atom types found in structure.")

    valid_elements = elements[valid_mask]
    valid_coords = coords[:, valid_mask]

    # Map elements to intensities
    indices = np.array([ELEMENT_TO_IDX[e] for e in valid_elements], dtype=np.int64)
    atom_intensities = ELEMENT_INTENSITIES[indices]

    # Build a tight odd-sized box around the bounding-box center
    coord_min = valid_coords.min(axis=1)
    coord_max = valid_coords.max(axis=1)
    origin = (coord_min + coord_max) / 2.0

    span = np.maximum(origin - coord_min, coord_max - origin)
    span_pix = np.ceil(span / pixel_size).astype(np.int64)

    margin_voxels = 2
    span_pix = span_pix + margin_voxels

    center_idx = span_pix
    vol_shape = span_pix * 2 + 1

    # Convert to voxel coordinates with correct 0-based indexing
    vox_coords = np.round((valid_coords - origin[:, None]) / pixel_size).astype(np.int64)
    vox_coords = vox_coords + center_idx[:, None]

    in_bounds = (
            (vox_coords[0] >= 0) & (vox_coords[0] < vol_shape[0]) &
            (vox_coords[1] >= 0) & (vox_coords[1] < vol_shape[1]) &
            (vox_coords[2] >= 0) & (vox_coords[2] < vol_shape[2])
    )

    vox_coords = vox_coords[:, in_bounds]
    atom_intensities = atom_intensities[in_bounds]

    vol = np.zeros(vol_shape, dtype=np.float32)
    np.add.at(vol, (vox_coords[0], vox_coords[1], vox_coords[2]), atom_intensities)

    # Keep only old trim=1 behavior
    vol = trim_volume(vol)
    return vol


def structure_to_volume(structure_path: str, pixel_size: float) -> np.ndarray:
    """Parse structure and generate one EM density volume."""
    data = parse_structure(structure_path)
    return build_volume(data, pixel_size)


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] safely."""
    vmin = float(vol.min())
    vmax = float(vol.max())

    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)

    vol = (vol - vmin) / (vmax - vmin)
    return vol.astype(np.float32)


def main(structure_path: str, pixel_size: float, out_path: str = None):
    vol = structure_to_volume(structure_path, pixel_size)
    vol = normalize_volume(vol)

    output_path = Path(out_path) if out_path is not None else Path(structure_path).with_suffix(".mrc")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    print(f"Saving volume to {output_path} ...")
    with mrcfile.new(str(output_path), overwrite=True) as mrc:
        mrc.set_data(vol)
        mrc.voxel_size = pixel_size

    print("Done.")


def parser_helper():
    parser = argparse.ArgumentParser(description="Convert PDB/CIF/mmCIF to EM density map")
    parser.add_argument("--structure_path", type=str, help="Path to PDB/CIF/mmCIF file")
    parser.add_argument("--pixel_size", type=float, help="Pixel size in angstroms")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MRC path (default: input filename with .mrc extension)",
    )
    return parser


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.structure_path, args.pixel_size, args.output)
