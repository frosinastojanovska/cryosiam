import argparse
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


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def trim_volume(vol: np.ndarray) -> np.ndarray:
    nonzero = np.nonzero(vol)
    if len(nonzero[0]) == 0:
        return vol
    slices = tuple(slice(axis.min(), axis.max() + 1) for axis in nonzero)
    return vol[slices]


def infer_element_from_pdb_line(line: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element

    atom_name = line[12:16].strip().upper()
    atom_name = "".join(c for c in atom_name if c.isalpha())
    if not atom_name:
        return ""

    if len(atom_name) >= 2 and atom_name[:2] in ELEMENT_TO_IDX:
        return atom_name[:2]
    return atom_name[:1]


def parse_pdb_protein(path: str) -> dict:
    atoms = []
    coords = []

    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            parts = line.split()
            if "DUM" in parts:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atoms.append(infer_element_from_pdb_line(line))
            coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No valid protein atoms found in {path}")

    return {
        "atoms": np.asarray(atoms),
        "coords": np.asarray(coords, dtype=np.float32).T,  # (3, N)
    }


def parse_cif_protein(path: str) -> dict:
    with open(path, "r") as f:
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

        for r in rows:
            try:
                atoms_all.append(str(r[col["type_symbol"]]).upper())
                coords_all.append([
                    float(r[col["Cartn_x"]]),
                    float(r[col["Cartn_y"]]),
                    float(r[col["Cartn_z"]]),
                ])
            except ValueError:
                continue

    if not coords_all:
        raise ValueError(f"No valid atoms found in {path}")

    return {
        "atoms": np.asarray(atoms_all),
        "coords": np.asarray(coords_all, dtype=np.float32).T,  # (3, N)
    }


def parse_structure_protein(path: str) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return parse_cif_protein(path)
    return parse_pdb_protein(path)


def fit_plane(points: np.ndarray):
    center = points.mean(axis=0)
    X = points - center
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    return center.astype(np.float32), normal.astype(np.float32)


def parse_opm_dummy_planes(pdb_path: str):
    """
    Parse OPM DUM atoms and detect whether they form:
      - a full bilayer (two leaflet planes)
      - a single leaflet plane

    Returns:
        None
        or {
            "mode": "single_plane" | "bilayer",
            "normal": np.ndarray shape (3,),
            "plane_point": np.ndarray shape (3,),              # for single_plane
            "lower_plane_point": np.ndarray shape (3,),        # for bilayer
            "upper_plane_point": np.ndarray shape (3,),        # for bilayer
        }
    """
    pts = []
    atom_types = []

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue

            parts = line.split()
            if len(parts) < 8 or "DUM" not in parts:
                continue

            try:
                atom_name = parts[2].upper()  # e.g. O / N
                x = float(parts[-3])
                y = float(parts[-2])
                z = float(parts[-1])
            except ValueError:
                continue

            atom_types.append(atom_name)
            pts.append([x, y, z])

    if len(pts) < 3:
        return None

    pts = np.asarray(pts, dtype=np.float32)
    atom_types = np.asarray(atom_types)

    # --- Robustly detect membrane normal by finding the axis with 2 leaflet levels ---
    # For each axis, cluster coordinates by rounding to 0.1 Å and counting dominant levels.
    best_axis = None
    best_levels = None
    best_score = -1

    for axis in range(3):
        vals = pts[:, axis]
        rounded = np.round(vals, 1)
        uniq, counts = np.unique(rounded, return_counts=True)

        # Keep only levels with enough support
        strong = counts >= max(3, int(0.01 * len(vals)))
        uniq = uniq[strong]
        counts = counts[strong]

        if len(uniq) < 1:
            continue

        # score prefers exactly two strong levels
        if len(uniq) >= 2:
            top2 = np.argsort(counts)[-2:]
            levels = np.sort(uniq[top2])
            sep = abs(levels[1] - levels[0])
            score = counts[top2].sum() + sep
            if score > best_score:
                best_score = score
                best_axis = axis
                best_levels = levels
        else:
            # candidate for single plane
            score = counts[0]
            if score > best_score:
                best_score = score
                best_axis = axis
                best_levels = np.array([uniq[0]], dtype=np.float32)

    if best_axis is None:
        return None

    normal = np.zeros(3, dtype=np.float32)
    normal[best_axis] = 1.0

    # --- Bilayer case: two leaflet levels found ---
    if len(best_levels) >= 2:
        lower_val, upper_val = float(best_levels[0]), float(best_levels[1])

        tol = 1.0
        lower_mask = np.abs(pts[:, best_axis] - lower_val) <= tol
        upper_mask = np.abs(pts[:, best_axis] - upper_val) <= tol

        if lower_mask.sum() < 3 or upper_mask.sum() < 3:
            return None

        lower_plane_point = pts[lower_mask].mean(axis=0)
        upper_plane_point = pts[upper_mask].mean(axis=0)

        # orient normal from lower -> upper
        signed_sep = np.dot(upper_plane_point - lower_plane_point, normal)
        if signed_sep < 0:
            lower_plane_point, upper_plane_point = upper_plane_point, lower_plane_point
            normal = -normal

        return {
            "mode": "bilayer",
            "lower_plane_point": lower_plane_point.astype(np.float32),
            "upper_plane_point": upper_plane_point.astype(np.float32),
            "normal": normal.astype(np.float32),
        }

    # --- Single-plane case ---
    plane_val = float(best_levels[0])
    tol = 1.0
    plane_mask = np.abs(pts[:, best_axis] - plane_val) <= tol
    if plane_mask.sum() < 3:
        return None

    plane_point = pts[plane_mask].mean(axis=0)

    return {
        "mode": "single_plane",
        "plane_point": plane_point.astype(np.float32),
        "normal": normal.astype(np.float32),
    }


def build_membrane_info_from_opm(pdb_path: str, protein_coords: np.ndarray, bilayer_thickness: float):
    dum_info = parse_opm_dummy_planes(pdb_path)
    if dum_info is None:
        return None

    protein_center = protein_coords.mean(axis=1).astype(np.float32)

    if dum_info["mode"] == "bilayer":
        lower_plane_point = dum_info["lower_plane_point"]
        upper_plane_point = dum_info["upper_plane_point"]
        normal = dum_info["normal"]
        center_point = 0.5 * (lower_plane_point + upper_plane_point)
        half_thickness = 0.5 * float(np.dot(upper_plane_point - lower_plane_point, normal))
        return {
            "normal": normal.astype(np.float32),
            "center_point": center_point.astype(np.float32),
            "half_thickness": float(abs(half_thickness)),
            "source": "opm_bilayer",
        }

    plane_point = dum_info["plane_point"]
    normal = dum_info["normal"]

    signed_dist = float(np.dot(protein_center - plane_point, normal))
    if signed_dist < 0:
        normal = -normal

    center_point = plane_point - 0.5 * bilayer_thickness * normal

    return {
        "normal": normal.astype(np.float32),
        "center_point": center_point.astype(np.float32),
        "half_thickness": float(bilayer_thickness / 2.0),
        "source": "opm_single_plane_plus_thickness",
    }


def build_protein_volume(data: dict, pixel_size: float, margin_voxels: int = 4):
    atoms = data["atoms"]
    coords = data["coords"]

    elements = np.array([str(a).upper() for a in atoms])
    valid_mask = np.isin(elements, KNOWN_ELEMENTS)
    if not valid_mask.any():
        raise ValueError("No supported atom types found.")

    valid_elements = elements[valid_mask]
    valid_coords = coords[:, valid_mask]

    indices = np.array([ELEMENT_TO_IDX[e] for e in valid_elements], dtype=np.int64)
    atom_intensities = ELEMENT_INTENSITIES[indices]

    coord_min = valid_coords.min(axis=1)
    coord_max = valid_coords.max(axis=1)
    origin = (coord_min + coord_max) / 2.0

    span = np.maximum(origin - coord_min, coord_max - origin)
    span_pix = np.ceil(span / pixel_size).astype(np.int64) + margin_voxels

    center_idx = span_pix
    vol_shape = span_pix * 2 + 1

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

    return vol, origin.astype(np.float32), vol_shape.astype(np.int64), valid_coords


def gaussian_kernel_1d(sigma_vox: float):
    if sigma_vox <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(np.ceil(3 * sigma_vox))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma_vox ** 2))
    k /= k.sum()
    return k.astype(np.float32)


def convolve_along_axis(vol: np.ndarray, kernel: np.ndarray, axis: int):
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * vol.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(vol, pad_width, mode="edge")

    out = np.empty_like(vol, dtype=np.float32)
    indexer = [slice(None)] * vol.ndim
    for i in range(vol.shape[axis]):
        src = indexer.copy()
        src[axis] = slice(i, i + len(kernel))
        window = padded[tuple(src)]

        dst = indexer.copy()
        dst[axis] = i
        out[tuple(dst)] = np.tensordot(kernel, window, axes=(0, axis))
    return out


def smooth_3d(vol: np.ndarray, sigma_vox: float):
    if sigma_vox <= 0:
        return vol.astype(np.float32)
    k = gaussian_kernel_1d(sigma_vox)
    out = convolve_along_axis(vol, k, axis=0)
    out = convolve_along_axis(out, k, axis=1)
    out = convolve_along_axis(out, k, axis=2)
    return out.astype(np.float32)


def build_vesicle_membrane_volume(
        vol_shape,
        origin,
        pixel_size: float,
        membrane_info: dict,
        vesicle_radius: float = 200.0,
        vesicle_side: str = "outer",
):
    """
    Build a spherical-shell membrane patch tangent to the OPM membrane at the protein site.

    IMPORTANT:
    Volume axes follow the same convention as the protein volume:
      axis 0 -> x
      axis 1 -> y
      axis 2 -> z
    """
    Dx, Dy, Dz = map(int, vol_shape)

    normal = membrane_info["normal"].astype(np.float32)
    normal = normal / np.linalg.norm(normal)
    center_point = membrane_info["center_point"].astype(np.float32)
    half_thickness = float(membrane_info["half_thickness"])

    if vesicle_side == "outer":
        vesicle_center = center_point - vesicle_radius * normal
    elif vesicle_side == "inner":
        vesicle_center = center_point + vesicle_radius * normal
    else:
        raise ValueError("vesicle_side must be 'outer' or 'inner'")

    # axis 0 = x, axis 1 = y, axis 2 = z
    x = (np.arange(Dx, dtype=np.float32) - (Dx // 2)) * pixel_size + origin[0]
    y = (np.arange(Dy, dtype=np.float32) - (Dy // 2)) * pixel_size + origin[1]
    z = (np.arange(Dz, dtype=np.float32) - (Dz // 2)) * pixel_size + origin[2]

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    rho = np.sqrt(
        (xx - vesicle_center[0]) ** 2 +
        (yy - vesicle_center[1]) ** 2 +
        (zz - vesicle_center[2]) ** 2
    )

    dr = np.abs(rho - vesicle_radius)

    membrane = np.zeros((Dx, Dy, Dz), dtype=np.float32)

    core_mask = dr < max(0.0, half_thickness - 4.0)
    head_mask = (dr >= max(0.0, half_thickness - 4.0)) & (dr <= half_thickness)

    membrane[core_mask] = 0.5
    membrane[head_mask] = 1.0

    membrane = smooth_3d(membrane, 1.0)
    return membrane.astype(np.float32)


def main(
        structure_path: str,
        pixel_size: float,
        output: str = None,
        bilayer_thickness: float = 30.0,
        membrane_weight: float = 2.0,
        vesicle_radius: float = 200.0,
        vesicle_side: str = "outer",
        save_components: bool = False,
):
    structure_path = str(structure_path)
    suffix = Path(structure_path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        raise ValueError("Membrane extraction is implemented only for OPM-style PDB input.")

    protein_data = parse_structure_protein(structure_path)

    protein_vol, origin, vol_shape, protein_coords = build_protein_volume(
        protein_data,
        pixel_size=pixel_size,
        margin_voxels=4,
    )

    membrane_info = build_membrane_info_from_opm(
        structure_path,
        protein_coords=protein_coords,
        bilayer_thickness=bilayer_thickness,
    )
    if membrane_info is None:
        raise ValueError(
            "Could not find OPM membrane information in the PDB. "
            "Expected one or two DUM membrane planes."
        )

    print(f"Membrane source: {membrane_info['source']}")
    print(f"Half thickness:  {membrane_info['half_thickness']:.2f} Å")
    print(f"Vesicle radius:  {vesicle_radius:.2f} Å")
    print(f"Vesicle side:    {vesicle_side}")

    membrane_vol = build_vesicle_membrane_volume(
        vol_shape=vol_shape,
        origin=origin,
        pixel_size=pixel_size,
        membrane_info=membrane_info,
        vesicle_radius=vesicle_radius,
        vesicle_side=vesicle_side,
    )

    total_vol = protein_vol + membrane_vol * membrane_weight
    total_vol = trim_volume(total_vol)
    total_vol = normalize_volume(total_vol)

    out_path = Path(output) if output is not None else Path(structure_path).with_suffix(".mrc")

    with mrcfile.new(str(out_path), overwrite=True) as mrc:
        mrc.set_data(total_vol.astype(np.float32))
        mrc.voxel_size = pixel_size

    if save_components:
        protein_path = out_path.with_name(out_path.stem + "_protein.mrc")
        membrane_path = out_path.with_name(out_path.stem + "_membrane.mrc")

        with mrcfile.new(str(protein_path), overwrite=True) as mrc:
            mrc.set_data(normalize_volume(protein_vol).astype(np.float32))
            mrc.voxel_size = pixel_size

        with mrcfile.new(str(membrane_path), overwrite=True) as mrc:
            mrc.set_data(normalize_volume(membrane_vol).astype(np.float32))
            mrc.voxel_size = pixel_size

        print(f"Saved protein-only map to  {protein_path}")
        print(f"Saved membrane-only map to {membrane_path}")

    print(f"Saved combined map to {out_path}")


def parser_helper():
    parser = argparse.ArgumentParser(
        description="Convert OPM PDB to protein + vesicle-like membrane MRC density"
    )
    parser.add_argument("structure_path", type=str, help="Path to OPM PDB")
    parser.add_argument("pixel_size", type=float, help="Voxel size in angstroms")
    parser.add_argument("--output", type=str, default=None, help="Output MRC path")
    parser.add_argument("--bilayer_thickness", type=float, default=10.0,
                        help="Used only when only one DUM plane is present")
    parser.add_argument("--membrane_weight", type=float, default=1.0,
                        help="Relative strength of membrane after normalization")
    parser.add_argument("--vesicle_radius", type=float, default=300.0, help="Radius of the vesicle in angstroms")
    parser.add_argument("--vesicle_side", type=str, choices=["outer", "inner"], default="inner",
                        help="Whether the protein sits on the outer or inner side of the vesicle")
    parser.add_argument("--save_components", action="store_true")
    return parser


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()

    main(
        structure_path=args.structure_path,
        pixel_size=args.pixel_size,
        output=args.output,
        bilayer_thickness=args.bilayer_thickness,
        membrane_weight=args.membrane_weight,
        vesicle_radius=args.vesicle_radius,
        vesicle_side=args.vesicle_side,
        save_components=args.save_components,
    )
