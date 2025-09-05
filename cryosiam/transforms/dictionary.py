import numpy as np
from monai.config import KeysCollection
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta
from typing import Optional, Dict, Hashable, Tuple
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.compose import MapTransform, RandomizableTransform

from .array import (
    RandomSharpen,
    ClipIntensity,
    ScaleIntensity,
    InvertIntensity,
    RandomLowPassBlur,
    RandomGaussianNoise,
    RandomHighPassSharpen
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
