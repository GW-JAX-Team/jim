import jax
import jax.numpy as jnp
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_detector_preset
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform

###########################################
########## First we grab data #############
###########################################

# Load cached GW150914 data
gps = 1126259462.4
fmin = 20.0
fmax = 1024.0

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

detector_preset = get_detector_preset()
ifos = [detector_preset["H1"], detector_preset["L1"]]

for ifo in ifos:
    data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
    ifo.set_data(data)
    psd = PowerSpectrum.from_file(str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz"))
    ifo.set_psd(psd)

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
M_c_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])
s1z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s1_z"])
s2z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s2_z"])
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
iota_prior = SinePrior(parameter_names=["iota"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        M_c_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        iota_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]
)

likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    prior=prior,
    waveform=RippleIMRPhenomD(),
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps,
    likelihood_transforms=likelihood_transforms,
    n_steps=5,
    popsize=10,
)

jim = Jim(
    likelihood,
    prior,
    likelihood_transforms=likelihood_transforms,
    n_training_loops=1,
    n_production_loops=1,
    n_local_steps=2,
    n_global_steps=2,
    n_chains=2,
    n_epochs=1,
    learning_rate=1e-4,
    n_max_examples=30,
    batch_size=100,
    mala_step_size=3e-3,
    global_thinning=1,
)

jim.sample()
jim.get_samples()
