import torch
import numpy as np
from monai.config import KeysCollection
from monai.utils import convert_to_tensor
from typing import Optional, Dict, Hashable, Tuple, Sequence
from monai.data.meta_obj import get_track_meta
from monai.transforms import GaussianSmooth
import torch.nn.functional as F
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.compose import MapTransform, RandomizableTransform

from .array import (
    NumpyToTensor,
    RandomSharpen,
    ClipIntensity,
    ScaleIntensity,
    InvertIntensity,
    RandomLowPassBlur,
    RandomGaussianNoise,
    RandomHighPassSharpen,
    RandomAmplitudeSpectrum
)


class ClipIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.ClipIntensity`.
    """

    backend = ClipIntensity.backend

    def __init__(self, keys: KeysCollection, a_min=None, a_max=None, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.clip = ClipIntensity(a_min=a_min, a_max=a_max)

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.clip(d[key])
        return d


class NumpyToTensord(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.ClipIntensity`.
    """

    backend = NumpyToTensor.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.convert = NumpyToTensor()

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.convert(d[key])
        return d


class ScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.ScaleIntensity`.
    """

    backend = ClipIntensity.backend

    def __init__(self, keys: KeysCollection, lower_percentage=0.1,
                 upper_percentage=99.9, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.scale = ScaleIntensity(lower_percentage=lower_percentage,
                                    upper_percentage=upper_percentage)

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scale(d[key])
        return d


class InvertIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.InvertIntensity`.
    """

    backend = InvertIntensity.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.inverter = InvertIntensity()

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.inverter(d[key])
        return d


class RandMaskFocusBlurd(RandomizableTransform, MapTransform):
    """
    Uses ground truth particle mask to blur or suppress everything
    outside the center particle in a crowded subtomogram.

    Requires both image and mask keys in the data dictionary.
    The mask is the binary ground truth of the center particle from CTS.

    Two modes:
        suppress: periphery → local mean value  (no zero artifacts)
        blur:     periphery → Gaussian blurred  (realistic background)
        mixed:    randomly choose per call

    The mask is dilated and smoothed at boundaries to avoid
    hard seam artifacts at the molecule edge.
    """
    backend = [TransformBackends.TORCH]

    def __init__(
            self,
            keys: KeysCollection,
            image_key: str = 'image',
            mask_key: str = 'mask',
            prob: float = 0.5,
            mode: str = 'mixed',
            blur_sigma_range: Sequence[float] = (2.0, 5.0),
            dilation_radius: int = 2,
            smooth_sigma: float = 1.5,
            allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys:              keys to process — should include image_key and mask_key
            image_key:         key for the crowded subtomogram
            mask_key:          key for the binary center particle mask
            prob:              probability of applying the transform
            mode:              'suppress', 'blur', or 'mixed'
            blur_sigma_range:  (min, max) Gaussian blur sigma for periphery
            dilation_radius:   voxels to expand mask — avoids hard boundary
            smooth_sigma:      Gaussian smoothing sigma for mask edges
            allow_missing_keys: don't raise exception if key is missing
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.image_key = image_key
        self.mask_key = mask_key
        self.mode = mode
        self.blur_sigma_range = blur_sigma_range
        self.dilation_radius = dilation_radius
        self.smooth_sigma = smooth_sigma

        # randomized per call
        self._do_blur = False
        self._blur_sigma = blur_sigma_range[0]

    def randomize(self, data: Optional[any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return

        # randomly choose mode if mixed
        if self.mode == 'mixed':
            self._do_blur = self.R.random() > 0.5
        else:
            self._do_blur = self.mode == 'blur'

        # random blur sigma
        self._blur_sigma = float(self.R.uniform(
            self.blur_sigma_range[0],
            self.blur_sigma_range[1]
        ))

    def _dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Expand mask using 3D max pooling.
        Ensures molecule boundary voxels are not suppressed.

        mask: (D, H, W) binary
        Returns: (D, H, W) dilated binary
        """
        r = self.dilation_radius
        kernel = 2 * r + 1
        mask_5d = mask.unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
        dilated = F.max_pool3d(
            mask_5d,
            kernel_size=kernel,
            stride=1,
            padding=r
        )
        return dilated.squeeze(0).squeeze(0)  # (D, H, W)

    def _smooth_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Smooth mask edges with Gaussian — creates soft transition
        between particle and background.

        mask: (D, H, W) float [0, 1]
        Returns: (D, H, W) smoothed float [0, 1]
        """
        smoother = GaussianSmooth(sigma=self.smooth_sigma)
        mask_4d = mask.unsqueeze(0)  # (1, D, H, W)
        smoothed = smoother(mask_4d).squeeze(0)  # (D, H, W)
        return smoothed.clamp(0, 1)

    def _apply_suppress(self, img, mask):
        """
        Suppress periphery to local mean value.

        img:  (C, D, H, W)
        mask: (D, H, W) float [0, 1] — 1=particle, 0=background
        Returns: (C, D, H, W)
        """
        # mean_val = img.mean()
        mask_4d = mask.unsqueeze(0)  # (1, D, H, W)
        # return mask_4d * img + (1 - mask_4d) * mean_val
        return mask_4d * img + (1 - mask_4d) * 0

    def _apply_blur(
            self,
            img: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Blur periphery with Gaussian, keep center sharp.

        img:  (C, D, H, W)
        mask: (D, H, W) float [0, 1]
        Returns: (C, D, H, W)
        """
        blur = GaussianSmooth(sigma=self._blur_sigma)
        blurred = blur(img)  # (C, D, H, W)
        mask_4d = mask.unsqueeze(0)  # (1, D, H, W)
        return mask_4d * img + (1 - mask_4d) * blurred

    def __call__(self, data, randomize: bool = True):
        d = dict(data)

        if randomize:
            self.randomize()

        if not self._do_transform:
            return d

        if self.image_key not in d:
            raise KeyError(f'Image key {self.image_key} not found in data. '
                           f'Available keys: {list(d.keys())}')
        if self.mask_key not in d:
            raise KeyError(f'Mask key {self.mask_key} not found in data. '
                           f'Available keys: {list(d.keys())}')

        img = d[self.image_key]  # (C, D, H, W)
        mask = d[self.mask_key]  # (D, H, W) or (1, D, H, W)

        # ensure mask is (D, H, W)
        if mask.dim() == 4:
            mask = mask.squeeze(0)

        mask = (mask > 0).float()

        # validate shapes match
        if tuple(img.shape[1:]) != tuple(mask.shape):
            raise ValueError(
                f'Image spatial shape {img.shape[1:]} does not match '
                f'mask shape {mask.shape}'
            )

        # dilate → smooth → soft mask
        mask = self._dilate_mask(mask)
        mask = self._smooth_mask(mask)

        # apply selected mode
        if self._do_blur:
            d[self.image_key] = self._apply_blur(img, mask)
        else:
            d[self.image_key] = self._apply_suppress(img, mask)

        return d


class AdaptiveFrequencyAugmentd(RandomizableTransform, MapTransform):
    """
    Computes particle size from the map density and derives
    blur/sharpen sigma proportional to it.

    sigma_blur    = random.uniform(radius * 0.10, radius * 0.30)
    sigma_sharpen derived from radius * sharpen_fraction with DoG

    This ensures small particles get small sigmas and large particles
    get large sigmas — both get meaningful augmentation without
    destroying shape information.

    Args:
        keys:              image keys to transform
        prob:              probability of applying any augmentation
        blur_prob:         probability of applying blur (default 1.0)
        sharpen_prob:      probability of applying DoG sharpening (default 0.5)
        density_threshold: voxels above this count as foreground (default 0.1)
        allow_missing_keys: don't raise exception if key is missing
    """
    backend = [TransformBackends.TORCH]

    def __init__(self,
                 keys,
                 prob: float = 1.0,
                 blur_prob: float = 1.0,
                 sharpen_prob: float = 0.5,
                 density_threshold: float = 0.1,
                 blur_fraction: float = None,
                 allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.blur_prob = blur_prob
        self.sharpen_prob = sharpen_prob
        self.density_threshold = density_threshold

        # randomized per call in randomize()
        self._apply_blur = False
        self.blur_fraction_fixed = blur_fraction
        self._blur_fraction = blur_fraction if blur_fraction is not None else 0.2
        self._apply_sharpen = False
        self._sigma2_fraction = 0.25
        self._ratio = 12.0
        self._alpha = 1.0

    def randomize(self, data=None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return

        # blur — fraction of radius sampled between 10% and 30%
        self._apply_blur = self.R.random() < self.blur_prob
        if self.blur_fraction_fixed is None:
            self._blur_fraction = self.R.uniform(0.10, 0.30)

        # DoG sharpening
        self._apply_sharpen = self.R.random() < self.sharpen_prob
        self._sigma2_fraction = self.R.uniform(0.15, 0.35)
        self._ratio = self.R.uniform(8.0, 20.0)
        self._alpha = self.R.uniform(0.5, 2.0)

    def _estimate_radius(self, image: torch.Tensor) -> float:
        """
        Estimate particle radius in voxels from foreground voxel count.
        Assumes roughly spherical particle:
            V = (4/3) * pi * r^3  →  r = (3V / 4pi)^(1/3)
        """
        n_foreground = (image > self.density_threshold).sum().item()
        if n_foreground < 1:
            return 5.0
        radius = (3 * n_foreground / (4 * torch.pi)) ** (1 / 3)
        return float(radius)

    def _gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel_size = int(6 * sigma + 1) | 1
        t = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        k1d = torch.exp(-0.5 * (t / sigma) ** 2)
        k1d = k1d / k1d.sum()

        out = x
        for view_shape, padding in [
            ((1, 1, -1, 1, 1), (kernel_size // 2, 0, 0)),
            ((1, 1, 1, -1, 1), (0, kernel_size // 2, 0)),
            ((1, 1, 1, 1, -1), (0, 0, kernel_size // 2)),
        ]:
            k = k1d.view(*view_shape)
            out = torch.nn.functional.conv3d(out, k, padding=padding)
        return out

    def __call__(self, data, randomize: bool = True):
        d = dict(data)

        if randomize:
            self.randomize()

        if not self._do_transform:
            return d

        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} not found. '
                               f'Available: {list(d.keys())}')

            image = d[key]  # (C, D, H, W)
            radius = self._estimate_radius(image)

            # --- blur ---
            if self._apply_blur:
                sigma_blur_min = max(0.3, radius * 0.10)
                sigma_blur_max = max(0.5, radius * 0.30)
                # map pre-sampled fraction into [min, max]
                sigma_blur = sigma_blur_min + self._blur_fraction * (
                        sigma_blur_max - sigma_blur_min)
                image = self._gaussian_blur(image.unsqueeze(0),
                                            sigma_blur).squeeze(0)

            # --- DoG sharpening ---
            if self._apply_sharpen:
                sigma2 = max(0.5, radius * self._sigma2_fraction)
                sigma1 = max(0.1, sigma2 / self._ratio)

                low_pass = self._gaussian_blur(
                    image.unsqueeze(0), sigma1).squeeze(0)
                low_pass2 = self._gaussian_blur(
                    image.unsqueeze(0), sigma2).squeeze(0)
                dog = low_pass - low_pass2
                image = image + self._alpha * dog

            d[key] = image

        return d


class RandomAmplitudeSpectrumd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomAmplitudeSpectrum`.
    """

    backend = RandomAmplitudeSpectrum.backend

    def __init__(self, keys: KeysCollection, n_bands: int, sigma: float = 0.2, prob: float = 0.1,
                 allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            n_bands: number of radial frequency bands (see RandomAmplitudeSpectrum).
            sigma: standard deviation of the per-step random-walk noise.
            prob: probability to apply the augmentation.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.amplitude_spectrum = RandomAmplitudeSpectrum(n_bands=n_bands, sigma=sigma, prob=prob)

    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomAmplitudeSpectrumd":
        super().set_random_state(seed, state)
        self.amplitude_spectrum.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random walk sequence
        self.amplitude_spectrum.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.amplitude_spectrum(d[key], randomize=False)
        return d


class RandomSharpend(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomSharpen`.
    """

    backend = RandomSharpen.backend

    def __init__(self, keys: KeysCollection, sigma: Tuple[float, float], prob: float = 0.1,
                 allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: range of sigma value for bluring Gaussian filter.
            prob: probability to apply the shapening.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.high_pass = RandomSharpen(sigma=sigma, prob=prob)

    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomSharpend":
        super().set_random_state(seed, state)
        self.high_pass.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma1, sigma2, etc.
        self.high_pass.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.high_pass(d[key], randomize=False)
        return d


class RandomLowPassBlurd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomLowPassBlur`.
    """

    backend = RandomLowPassBlur.backend

    def __init__(self, keys: KeysCollection, sigma: Tuple[float, float], ignore_zeros: bool = False, prob: float = 0.1,
                 allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: range of sigma value for Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the blur.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.low_pass = RandomLowPassBlur(sigma=sigma, ignore_zeros=ignore_zeros, prob=prob)

    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomLowPassBlurd":
        super().set_random_state(seed, state)
        self.low_pass.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma, etc.
        self.low_pass.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.low_pass(d[key], randomize=False)
        return d


class RandomGaussianNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomGaussianNoise`.
    """

    backend = RandomGaussianNoise.backend

    def __init__(self, keys: KeysCollection, sigma: Tuple[float, float], ignore_zeros: bool = False, prob: float = 0.1,
                 allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: std of the added Gaussian noise.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the noise.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.noised = RandomGaussianNoise(sigma=sigma, ignore_zeros=ignore_zeros, prob=prob)

    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomGaussianNoise":
        super().set_random_state(seed, state)
        self.noised.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma, etc.
        self.noised.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.noised(d[key], randomize=False)
        return d


class RandomHighPassSharpend(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomHighPassSharpen`.
    """

    backend = RandomHighPassSharpen.backend

    def __init__(self, keys: KeysCollection, sigma: Tuple[float, float], sigma2: Tuple[float, float],
                 ignore_zeros: bool = False, prob: float = 0.1, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: range of sigma value for first Gaussian filter.
            sigma2: range of sigma value for second Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the shapening.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.high_pass = RandomHighPassSharpen(sigma=sigma, sigma2=sigma2, ignore_zeros=ignore_zeros, prob=prob)

    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomHighPassSharpend":
        super().set_random_state(seed, state)
        self.high_pass.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma1, sigma2, etc.
        self.high_pass.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.high_pass(d[key], randomize=False)
        return d
