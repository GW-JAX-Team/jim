import math

import jax.numpy as jnp
import pytest

from jimgw.samplers.periodic import (
    to_displacement_wrapper,
    to_index_dict,
    to_prior_space_stepper,
    to_unit_cube_stepper,
)


PARAMS_3D = ("alpha", "phase_c", "psi")
# phase_c in [0, 2π], psi in [0, π]
PERIODIC_3D = {"phase_c": (0.0, 2 * math.pi), "psi": (0.0, math.pi)}


# ---------------------------------------------------------------------------
# to_index_dict
# ---------------------------------------------------------------------------


def test_to_index_dict_none_passthrough():
    assert to_index_dict(None, PARAMS_3D) is None


def test_to_index_dict_single_periodic():
    result = to_index_dict({"phase_c": (0.0, 2 * math.pi)}, PARAMS_3D)
    assert result == {1: (0.0, 2 * math.pi)}


def test_to_index_dict_multiple_periodic():
    periodic = {
        "phase_c": (0.0, 2 * math.pi),
        "psi": (0.0, math.pi),
    }
    result = to_index_dict(periodic, PARAMS_3D)
    assert result == {1: (0.0, 2 * math.pi), 2: (0.0, math.pi)}


def test_to_index_dict_first_param():
    result = to_index_dict({"alpha": (-1.0, 1.0)}, PARAMS_3D)
    assert result == {0: (-1.0, 1.0)}


def test_to_index_dict_unknown_name_raises():
    with pytest.raises(ValueError, match="not found in sampling parameters"):
        to_index_dict({"unknown": (0.0, 1.0)}, PARAMS_3D)


def test_to_index_dict_empty_periodic():
    result = to_index_dict({}, PARAMS_3D)
    assert result == {}


def test_to_index_dict_all_params_periodic():
    periodic = {
        "alpha": (0.0, 1.0),
        "phase_c": (0.0, 2 * math.pi),
        "psi": (0.0, math.pi),
    }
    result = to_index_dict(periodic, PARAMS_3D)
    assert result == {0: (0.0, 1.0), 1: (0.0, 2 * math.pi), 2: (0.0, math.pi)}


# ---------------------------------------------------------------------------
# to_unit_cube_stepper
# ---------------------------------------------------------------------------


def _make_pos(alpha=0.3, phase_c=0.6, psi=0.9):
    return {"alpha": jnp.array(alpha), "phase_c": jnp.array(phase_c), "psi": jnp.array(psi)}


def _make_dir(alpha=0.1, phase_c=0.7, psi=-0.5):
    return {"alpha": jnp.array(alpha), "phase_c": jnp.array(phase_c), "psi": jnp.array(psi)}


def test_unit_cube_stepper_none_periodic():
    stepper = to_unit_cube_stepper(None, PARAMS_3D)
    pos = _make_pos()
    direction = _make_dir()
    result = stepper(pos, direction, 1.0)
    # Non-periodic: just add step
    for k in PARAMS_3D:
        assert float(result[k]) == pytest.approx(float(pos[k]) + float(direction[k]))


def test_unit_cube_stepper_wraps_periodic():
    stepper = to_unit_cube_stepper({"phase_c": (0.0, 1.0)}, PARAMS_3D)
    pos = {"alpha": jnp.array(0.3), "phase_c": jnp.array(0.9), "psi": jnp.array(0.1)}
    direction = {"alpha": jnp.array(0.0), "phase_c": jnp.array(0.5), "psi": jnp.array(0.0)}
    # phase_c + step = 0.9 + 0.5 = 1.4 → mod 1.0 = 0.4
    result = stepper(pos, direction, 1.0)
    assert float(result["phase_c"]) == pytest.approx(0.4, abs=1e-6)
    # alpha and psi are not periodic
    assert float(result["alpha"]) == pytest.approx(0.3, abs=1e-6)
    assert float(result["psi"]) == pytest.approx(0.1, abs=1e-6)


def test_unit_cube_stepper_no_wrapping_needed():
    stepper = to_unit_cube_stepper({"phase_c": (0.0, 1.0)}, PARAMS_3D)
    pos = {"alpha": jnp.array(0.0), "phase_c": jnp.array(0.1), "psi": jnp.array(0.0)}
    direction = {"alpha": jnp.array(0.0), "phase_c": jnp.array(0.2), "psi": jnp.array(0.0)}
    result = stepper(pos, direction, 1.0)
    assert float(result["phase_c"]) == pytest.approx(0.3, abs=1e-6)


def test_unit_cube_stepper_unknown_name_raises():
    with pytest.raises(ValueError, match="not found in sampling parameters"):
        to_unit_cube_stepper({"unknown": (0.0, 1.0)}, PARAMS_3D)


# ---------------------------------------------------------------------------
# to_prior_space_stepper
# ---------------------------------------------------------------------------


def test_prior_space_stepper_none_periodic():
    stepper = to_prior_space_stepper(None, PARAMS_3D)
    pos = _make_pos()
    direction = _make_dir()
    result, accepted = stepper(pos, direction, 1.0)
    assert accepted is True
    for k in PARAMS_3D:
        assert float(result[k]) == pytest.approx(float(pos[k]) + float(direction[k]))


def test_prior_space_stepper_wraps_phase_c():
    two_pi = 2 * math.pi
    stepper = to_prior_space_stepper({"phase_c": (0.0, two_pi)}, PARAMS_3D)
    pos = {"alpha": jnp.array(0.0), "phase_c": jnp.array(6.0), "psi": jnp.array(0.0)}
    direction = {"alpha": jnp.array(0.0), "phase_c": jnp.array(1.0), "psi": jnp.array(0.0)}
    result, accepted = stepper(pos, direction, 1.0)
    # 6.0 + 1.0 = 7.0 → mod 2π ≈ 0.717
    expected = float(jnp.mod(7.0, two_pi))
    assert float(result["phase_c"]) == pytest.approx(expected, abs=1e-6)
    assert accepted is True


def test_prior_space_stepper_returns_accepted_bool():
    stepper = to_prior_space_stepper({"phase_c": (0.0, 2 * math.pi)}, PARAMS_3D)
    _, accepted = stepper(_make_pos(), _make_dir(), 0.1)
    assert accepted is True


def test_prior_space_stepper_unknown_name_raises():
    with pytest.raises(ValueError, match="not found in sampling parameters"):
        to_prior_space_stepper({"unknown": (0.0, 1.0)}, PARAMS_3D)


# ---------------------------------------------------------------------------
# to_displacement_wrapper
# ---------------------------------------------------------------------------


def test_displacement_wrapper_none_periodic():
    wrapper = to_displacement_wrapper(None, PARAMS_3D)
    proposed = {"alpha": jnp.array(0.1), "phase_c": jnp.array(0.5), "psi": jnp.array(-0.2)}
    current = {"alpha": jnp.array(0.3), "phase_c": jnp.array(2.0), "psi": jnp.array(1.0)}
    result = wrapper(proposed, current)
    for k in PARAMS_3D:
        assert float(result[k]) == pytest.approx(float(proposed[k]))


def test_displacement_wrapper_wraps_periodic():
    two_pi = 2 * math.pi
    wrapper = to_displacement_wrapper({"phase_c": (0.0, two_pi)}, PARAMS_3D)
    # current = 5.8, displacement = 1.0 → new = 6.8 → wrapped = mod(6.8, 2π) ≈ 0.517
    current_val = 5.8
    disp_val = 1.0
    current = {"alpha": jnp.array(0.0), "phase_c": jnp.array(current_val), "psi": jnp.array(0.0)}
    proposed = {"alpha": jnp.array(0.0), "phase_c": jnp.array(disp_val), "psi": jnp.array(0.0)}
    result = wrapper(proposed, current)
    wrapped_pos = float(jnp.mod(current_val + disp_val, two_pi))
    expected_disp = wrapped_pos - current_val
    assert float(result["phase_c"]) == pytest.approx(expected_disp, abs=1e-6)
    # Non-periodic params unchanged
    assert float(result["alpha"]) == pytest.approx(0.0, abs=1e-6)


def test_displacement_wrapper_unknown_name_raises():
    with pytest.raises(ValueError, match="not found in sampling parameters"):
        to_displacement_wrapper({"unknown": (0.0, 1.0)}, PARAMS_3D)
