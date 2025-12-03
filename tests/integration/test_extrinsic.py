import jax
import jax.numpy as jnp

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    SimpleConstrainedPrior,
)
from jimgw.core.single_event.detector import get_detector_preset
from jimgw.core.single_event.likelihood import ZeroLikelihood
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4

detector_preset = get_detector_preset()
ifos = [detector_preset["H1"], detector_preset["L1"], detector_preset["V1"]]

M_c_prior = UniformPrior(10.0, 80.0, parameter_names=["M_c"])
dL_prior = SimpleConstrainedPrior([PowerLawPrior(10.0, 2000.0, 2.0, parameter_names=["d_L"])])
t_c_prior = SimpleConstrainedPrior([UniformPrior(-0.05, 0.05, parameter_names=["t_c"])])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
iota_prior = SinePrior(parameter_names=["iota"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        M_c_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        iota_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]
)


sample_transforms = [
    # all the user reparametrization transform
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    # all the bound to unbound transform
    BoundToUnbound(
        name_mapping=[["M_c"], ["M_c_unbounded"]],
        original_lower_bound=10.0,
        original_upper_bound=80.0,
    ),
    BoundToUnbound(
        name_mapping=[["iota"], ["iota_unbounded"]],
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=[["psi"], ["psi_unbounded"]],
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=[["zenith"], ["zenith_unbounded"]],
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=[["azimuth"], ["azimuth_unbounded"]],
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=[["phase_det"], ["phase_det_unbounded"]],
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
]

likelihood_transforms = []

likelihood = ZeroLikelihood()

mass_matrix = jnp.eye(len(prior.base_prior))
local_sampler_arg = {"step_size": mass_matrix * 3e-3}


n_epochs = 2
n_loop_training = 1
learning_rate = 1e-4


jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_training_loops=n_loop_training,
    n_production_loops=1,
    n_local_steps=2,
    n_global_steps=2,
    n_chains=10,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30,
    batch_size=100,
    mala_step_size=3e-3,
)

print("Start sampling")
key = jax.random.PRNGKey(42)
jim.sample()
samples = jim.get_samples()
