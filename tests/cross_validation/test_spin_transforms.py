"""Cross-validation tests for spin transforms against bilby.

Requires bilby to be installed.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from tests.utils import check_bilby_available, common_keys_allclose

# Check if bilby is available before running tests
try:
    check_bilby_available()
    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BILBY_AVAILABLE,
    reason="bilby required for cross-validation tests",
)

jax.config.update("jax_enable_x64", True)


class TestSpinAnglesToCartesianSpinTransformBilby:
    """Cross-validation tests comparing Jim spin transforms to bilby."""

    def test_forward_spin_transform(self):
        """Test transformation from spin angles to cartesian spins against bilby.

        This test generates random spin angle parameters, transforms them using
        both Jim and bilby, and compares the results.
        """
        from bilby.gw.conversion import bilby_to_lalsimulation_spins
        from jimgw.core.single_event.transforms import (
            SpinAnglesToCartesianSpinTransform,
        )
        from jimgw.core.single_event.utils import Mc_q_to_m1_m2

        n_samples = 50
        key = jax.random.PRNGKey(42)
        subkeys = jax.random.split(key, 11)

        # Generate random spin angle parameters
        theta_jn = jax.random.uniform(subkeys[0], (n_samples,), minval=0, maxval=jnp.pi)
        phi_jl = jax.random.uniform(
            subkeys[1], (n_samples,), minval=0, maxval=2 * jnp.pi
        )
        tilt_1 = jax.random.uniform(subkeys[2], (n_samples,), minval=0, maxval=jnp.pi)
        tilt_2 = jax.random.uniform(subkeys[3], (n_samples,), minval=0, maxval=jnp.pi)
        phi_12 = jax.random.uniform(
            subkeys[4], (n_samples,), minval=0, maxval=2 * jnp.pi
        )
        a_1 = jax.random.uniform(subkeys[5], (n_samples,), minval=0.01, maxval=0.99)
        a_2 = jax.random.uniform(subkeys[6], (n_samples,), minval=0.01, maxval=0.99)
        M_c = jax.random.uniform(subkeys[7], (n_samples,), minval=5, maxval=50)
        q = jax.random.uniform(subkeys[8], (n_samples,), minval=0.125, maxval=1)
        phase_c = jax.random.uniform(
            subkeys[9], (n_samples,), minval=0, maxval=2 * jnp.pi
        )

        # Test with different reference frequencies
        for f_ref in [10.0, 20.0, 50.0]:
            # Get bilby results
            m1, m2 = Mc_q_to_m1_m2(M_c, q)
            bilby_results = []
            for i in range(n_samples):
                iota_b, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
                    theta_jn=float(theta_jn[i]),
                    phi_jl=float(phi_jl[i]),
                    tilt_1=float(tilt_1[i]),
                    tilt_2=float(tilt_2[i]),
                    phi_12=float(phi_12[i]),
                    a_1=float(a_1[i]),
                    a_2=float(a_2[i]),
                    mass_1=float(m1[i]),
                    mass_2=float(m2[i]),
                    reference_frequency=f_ref,
                    phase=float(phase_c[i]),
                )
                bilby_results.append(
                    {
                        "iota": iota_b,
                        "s1_x": s1x,
                        "s1_y": s1y,
                        "s1_z": s1z,
                        "s2_x": s2x,
                        "s2_y": s2y,
                        "s2_z": s2z,
                    }
                )

            bilby_spins = {
                key: jnp.array([r[key] for r in bilby_results])
                for key in bilby_results[0].keys()
            }

            # Get Jim results
            input_dict = {
                "theta_jn": theta_jn,
                "phi_jl": phi_jl,
                "tilt_1": tilt_1,
                "tilt_2": tilt_2,
                "phi_12": phi_12,
                "a_1": a_1,
                "a_2": a_2,
                "M_c": M_c,
                "q": q,
                "phase_c": phase_c,
            }
            transform = SpinAnglesToCartesianSpinTransform(freq_ref=f_ref)
            jimgw_spins, jacobian = jax.vmap(transform.transform)(input_dict)

            # Compare results
            assert common_keys_allclose(jimgw_spins, bilby_spins), (
                f"Jim and bilby spin transforms disagree at f_ref={f_ref}"
            )
            assert not jnp.isnan(jacobian).any(), "Jacobian contains NaN values"

    def test_backward_spin_transform(self):
        """Test transformation from cartesian spins to spin angles against bilby.

        This test generates random Cartesian spin parameters, transforms them using
        both Jim and bilby, and compares the results.
        """
        from bilby.gw.conversion import lalsimulation_spins_to_bilby
        from jimgw.core.single_event.transforms import (
            SpinAnglesToCartesianSpinTransform,
        )
        from jimgw.core.single_event.utils import Mc_q_to_m1_m2

        n_samples = 50
        key = jax.random.PRNGKey(123)
        subkeys = jax.random.split(key, 8)

        # Generate random cartesian spin parameters
        S1 = jax.random.uniform(subkeys[0], (3, n_samples), minval=-1, maxval=1)
        S2 = jax.random.uniform(subkeys[1], (3, n_samples), minval=-1, maxval=1)
        a1 = jax.random.uniform(subkeys[2], (n_samples,), minval=0.01, maxval=0.99)
        a2 = jax.random.uniform(subkeys[3], (n_samples,), minval=0.01, maxval=0.99)
        S1 = S1 * a1 / jnp.linalg.norm(S1, axis=0)
        S2 = S2 * a2 / jnp.linalg.norm(S2, axis=0)

        iota = jax.random.uniform(
            subkeys[4], (n_samples,), minval=0.1, maxval=jnp.pi - 0.1
        )
        M_c = jax.random.uniform(subkeys[5], (n_samples,), minval=5, maxval=50)
        q = jax.random.uniform(subkeys[6], (n_samples,), minval=0.125, maxval=1)
        phase_c = jax.random.uniform(
            subkeys[7], (n_samples,), minval=0, maxval=2 * jnp.pi
        )

        # Test with different reference frequencies
        for f_ref in [10.0, 20.0, 50.0]:
            # Get bilby results
            m1, m2 = Mc_q_to_m1_m2(M_c, q)
            bilby_results = []
            for i in range(n_samples):
                result = lalsimulation_spins_to_bilby(
                    incl=float(iota[i]),
                    spin1x=float(S1[0, i]),
                    spin1y=float(S1[1, i]),
                    spin1z=float(S1[2, i]),
                    spin2x=float(S2[0, i]),
                    spin2y=float(S2[1, i]),
                    spin2z=float(S2[2, i]),
                    mass_1=float(m1[i]),
                    mass_2=float(m2[i]),
                    reference_frequency=f_ref,
                    phase=float(phase_c[i]),
                )
                bilby_results.append(result)

            bilby_spins = {
                key: jnp.array([r[key] for r in bilby_results])
                for key in bilby_results[0].keys()
            }

            # Get Jim results
            input_dict = {
                "iota": iota,
                "s1_x": S1[0],
                "s1_y": S1[1],
                "s1_z": S1[2],
                "s2_x": S2[0],
                "s2_y": S2[1],
                "s2_z": S2[2],
                "M_c": M_c,
                "q": q,
                "phase_c": phase_c,
            }
            transform = SpinAnglesToCartesianSpinTransform(freq_ref=f_ref)
            jimgw_spins, jacobian = jax.vmap(transform.inverse)(input_dict)

            # Compare results
            assert common_keys_allclose(jimgw_spins, bilby_spins), (
                f"Jim and bilby backward spin transforms disagree at f_ref={f_ref}"
            )
            assert not jnp.isnan(jacobian).any(), "Jacobian contains NaN values"
