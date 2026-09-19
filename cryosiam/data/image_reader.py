import mrcfile
import numpy as np
from monai.data import ImageReader
from monai.utils import ensure_tuple
from monai.config import DtypeLike, PathLike
from monai.data.utils import is_supported_format
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class MrcReader(ImageReader):
    """
    Load cryoET tomograms based on MRCFILE library.
    The loaded data array will be in C order, for example, a 3D image NumPy
    array index order will be `CZYX`.
    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `-1`.
        dtype: dtype of the output data array when loading with mrcfile library.
        kwargs: additional args for `mrcfile.open` API.
        """

    def __init__(
            self,
            channel_dim: Optional[int] = None,
            dtype: DtypeLike = np.float32,
            read_in_mem=False,
            writable=True,
            **kwargs,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.dtype = dtype
        self.read_in_mem = read_in_mem
        self.writable = writable
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by MRC reader.
        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["mrc", "mrc.gz", "mrc.bz2", "rec", "rec.gz", "rec.bz2"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs) -> Union[Sequence[Any], Any]:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.
        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.
        """
        img_: List[np.array] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            if self.read_in_mem:
                mrc = mrcfile.open(name, mode='r', permissive=True)
            else:
                mrc = mrcfile.mmap(name, mode='r', permissive=True)
            img_.append(mrc)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of metadata.
        Args:
            img: an image object loaded from an image file or a list of image objects.
        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        important_info = ['nx', 'ny', 'nz', 'mode', 'nxstart', 'nystart', 'nzstart', 'mx', 'my', 'mz']

        for i in ensure_tuple(img):
            header = i.header
            header = {name: header[name] for name in header.dtype.names if name in important_info}
            if img.header.mx == 0 or img.header.my == 0 or img.header.mz == 0:
                header['voxel_size'] = np.asarray((1.0, 1.0, 1.0))
            else:
                header['voxel_size'] = np.asarray((img.voxel_size['x'],
                                                   img.voxel_size['y'],
                                                   img.voxel_size['z']))
            data = i.data[:]
            if self.writable:
                data.setflags(write=True)
            img_array.append(data)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header["original_channel_dim"] = "no_channel"
            else:
                header["original_channel_dim"] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta


# Following two functions are direct copy from monai.data.image_reader.py

def _copy_compatible_dict(from_dict: Dict, to_dict: Dict):
    if not isinstance(to_dict, dict):
        raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
    if not to_dict:
        for key in from_dict:
            datum = from_dict[key]
            if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
                continue
            to_dict[key] = datum
    else:
        affine_key, shape_key = "affine", "spatial_shape"
        if affine_key in from_dict and not np.allclose(from_dict[affine_key], to_dict[affine_key]):
            raise RuntimeError(
                "affine matrix of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
            )
        if shape_key in from_dict and not np.allclose(from_dict[shape_key], to_dict[shape_key]):
            raise RuntimeError(
                "spatial_shape of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )


def _stack_images(image_list: List, meta_dict: Dict):
    if len(image_list) <= 1:
        return image_list[0]
    if meta_dict.get("original_channel_dim", None) not in ("no_channel", None):
        channel_dim = int(meta_dict["original_channel_dim"])
        return np.concatenate(image_list, axis=channel_dim)
    # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
    meta_dict["original_channel_dim"] = 0
    return np.stack(image_list, axis=0)
