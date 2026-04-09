from typing import Callable

from jaxtyping import Array, Float
from ripplegw import (
    IMRPhenomD,
    IMRPhenomPv2,
    TaylorF2,
    IMRPhenomD_NRTidalv2,
    IMRPhenomXAS,
    IMRPhenomXAS_NRTidalv3,
    IMRPhenomXPHM,
    SineGaussian,
)

#: Type alias for any initialized, callable waveform model.
#: Matches the ``__call__`` signature of :class:`ripplegw.interfaces.Waveform`:
#: ``f(freq, params) -> {"p": plus_strain, "c": cross_strain}``.
WaveformCallable = Callable[
    [Float[Array, " n_freq"], dict[str, Float]],
    dict[str, Float[Array, " n_freq"]],
]

# aliases
RippleIMRPhenomD = IMRPhenomD
RippleIMRPhenomPv2 = IMRPhenomPv2
RippleTaylorF2 = TaylorF2
RippleIMRPhenomD_NRTidalv2 = IMRPhenomD_NRTidalv2
RippleIMRPhenomXAS = IMRPhenomXAS
RippleIMRPhenomXAS_NRTidalv3 = IMRPhenomXAS_NRTidalv3
RippleIMRPhenomXPHM = IMRPhenomXPHM
RippleSineGaussian = SineGaussian
