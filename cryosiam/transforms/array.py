import torch
import numpy as np
from scipy import ndimage
from typing import Any, Optional, Tuple
from monai.data.meta_obj import get_track_meta
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform, RandomizableTransform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor


class ClipIntensity(Transform):
    """
    Clip the intensity for the entire image with given minimum and maximum intensity values.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, a_min=None, a_max=None) -> None:
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        out = torch.clip(img, min=self.a_min, max=self.a_max)
        out, *_ = convert_data_type(data=out, dtype=img.dtype)
        return out


class ScaleIntensity(Transform):
    """
        Scale intensity for the entire image by removing lower and upper percentage
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, lower_percentage=0.1, upper_percentage=99.9) -> None:
        self.lower_percentage = lower_percentage
        self.upper_percentage = upper_percentage

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """

        img = convert_to_tensor(img, track_meta=get_track_meta())

        img_t, *_ = convert_data_type(img, np.ndarray, dtype=np.float32)
        min_val = np.percentile(img_t, self.lower_percentage)
        max_val = np.percentile(img_t, self.upper_percentage)
        img_t = (img_t - min_val) / (max_val - min_val)
        out = np.clip(img_t, 0, 1)

        out, *_ = convert_data_type(data=out, dtype=img.dtype)
        return out


class InvertIntensity(Transform):
    """
    Invert intensity for the entire image by multiplying with -1.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """

        img = convert_to_tensor(img, track_meta=get_track_meta())
        out = img * -1
        out, *_ = convert_data_type(data=out, dtype=img.dtype)

        return out


class RandomSharpen(RandomizableTransform):
    """
    Implement high pass filtering on an image.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, sigma: Tuple[float, float], prob: float = 0.1) -> None:
        """
        Args:
            sigma: range of sigma value for bluring Gaussian filter.
            prob: probability to apply the sharpening.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        if len(self.sigma_range) != 2:
            raise AssertionError(f"Sigma range should be a sequence of two elements (start, end), "
                                 f"but got values {self.sigma_range}.")
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(f"First element of sigma range should be smaller than the second element of the "
                                 f"range, but got values {self.sigma_range}.")
        self.prob = prob
        self.sigma1 = self.sigma_range[0]

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma1 = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        low_pass = torch.from_numpy(ndimage.gaussian_filter(img_t.numpy(), self.sigma1)).float()
        sharpen = img_t + (img_t - low_pass)
        out_t = sharpen.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out


class RandomLowPassBlur(RandomizableTransform):
    """
    Implement low pass filtering on an image.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, sigma: Tuple[float, float], ignore_zeros: bool = False, prob: float = 0.1) -> None:
        """
        Args:
            sigma: range of sigma value for Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the blur.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        if len(self.sigma_range) != 2:
            raise AssertionError(f"Sigma range should be a sequence of two elements (start, end), "
                                 f"but got values {self.sigma_range}.")
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(f"First element of sigma range should be smaller than the second element of the "
                                 f"range, but got values {self.sigma_range}.")
        self.sigma = self.sigma_range[0]
        self.ignore_zeros = ignore_zeros

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        mask = img_t == 0
        low_pass = torch.from_numpy(ndimage.gaussian_filter(img_t.numpy(), self.sigma)).float()
        if self.ignore_zeros:
            low_pass[mask] = 0
        out_t = low_pass.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out


class RandomGaussianNoise(RandomizableTransform):
    """
    Implement random gaussian noise on an image.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, sigma: Tuple[float, float], ignore_zeros: bool = False, prob: float = 0.1) -> None:
        """
        Args:
            sigma: std of the added Gaussian noise.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the noise.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        if len(self.sigma_range) != 2:
            raise AssertionError(f"Sigma range should be a sequence of two elements (start, end), "
                                 f"but got values {self.sigma_range}.")
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(f"First element of sigma range should be smaller than the second element of the "
                                 f"range, but got values {self.sigma_range}.")
        self.sigma = self.sigma_range[0]
        self.ignore_zeros = ignore_zeros

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        mask = img_t == 0
        noise = np.random.normal(0, self.sigma, size=img_t.shape).astype(np.float32)
        noised = img_t + torch.from_numpy(noise)
        min_val, max_val = torch.min(noised), torch.max(noised)
        noised = (noised - min_val) / (max_val - min_val)
        if self.ignore_zeros:
            noised[mask] = 0
        out_t = noised.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out


class RandomHighPassSharpen(RandomizableTransform):
    """
    Implement high pass filtering on an image.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, sigma: Tuple[float, float], sigma2: Tuple[float, float], ignore_zeros: bool = False,
                 prob: float = 0.1) -> None:
        """
        Args:
            sigma: range of sigma value for first Gaussian filter.
            sigma2: range of sigma value for second Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the sharpening.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        self.sigma2_range = sigma2
        if len(self.sigma_range) != 2:
            raise AssertionError(f"Sigma range should be a sequence of two elements (start, end), "
                                 f"but got values {self.sigma_range}.")
        if len(self.sigma2_range) != 2:
            raise AssertionError(f"Sigma2 range should be a sequence of two elements (start, end), "
                                 f"but got values {self.sigma2_range}.")
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(f"First element of sigma range should be smaller than the second element of the "
                                 f"range, but got values {self.sigma_range}.")
        if self.sigma2_range[0] > self.sigma2_range[1]:
            raise AssertionError(f"First element of sigma2 range should be smaller than the second element of the "
                                 f"range, but got values {self.sigma2_range}.")
        self.prob = prob
        self.sigma1 = self.sigma_range[0]
        self.sigma2 = self.sigma2_range[0]
        self.ignore_zeros = ignore_zeros

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma1 = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])
        self.sigma2 = self.R.uniform(low=self.sigma2_range[0], high=self.sigma2_range[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        mask = img_t == 0
        low_pass = torch.from_numpy(ndimage.gaussian_filter(img_t.numpy(), self.sigma1)).float()
        low_pass2 = torch.from_numpy(ndimage.gaussian_filter(img_t.numpy(), self.sigma2)).float()
        high_pass = low_pass - low_pass2
        if self.ignore_zeros:
            high_pass[mask] = 0
        out_t = high_pass.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out
