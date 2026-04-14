import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from typing import Callable

from jimgw.core.constants import MTSUN
from jimgw.core.utils import safe_arctan2, carte_to_spherical_angles


def apply_fixed_parameters(
    params: dict[str, Float],
    fixed_parameters: dict[
        str, Float | Callable[[dict[str, Float]], Float | dict[str, Float]]
    ],
) -> dict[str, Float]:
    """Merge ``fixed_parameters`` into *params*, resolving callables in-place.

    For each entry in ``fixed_parameters``:

    - If the value is **callable**, it is called with the current *params* dict.
      If the result is a ``dict``, the value stored under the matching key is used;
      otherwise the scalar result is used directly.
    - If the value is **not callable**, it is inserted as-is.

    Args:
        params (dict[str, Float]): Parameter dictionary to update in-place. Callers
            that need to preserve the original should pass a copy.
        fixed_parameters (dict): Fixed overrides. Values may be scalar constants or
            callables ``f(params) -> Float | dict[str, Float]`` applied in insertion
            order.

    Returns:
        dict[str, Float]: The same ``params`` dict, mutated in-place with the fixed
            parameters applied.

    Raises:
        KeyError: If a callable returns a dict that does not contain the expected key.
    """
    for key, value in fixed_parameters.items():
        if callable(value):
            result = value(params)
            if isinstance(result, dict):
                if key not in result:
                    raise KeyError(
                        f"apply_fixed_parameters: callable {value!r} returned a dict "
                        f"that does not contain the expected key {key!r}. "
                        f"Returned keys: {list(result.keys())}"
                    )
                params[key] = result[key]
            else:
                params[key] = result
        else:
            params[key] = value
    return params


def complex_inner_product(
    h1: Float[Array, " n_freq"],
    h2: Float[Array, " n_freq"],
    psd: Float[Array, " n_freq"],
    df: Float,
) -> Complex:
    """Compute the complex noise-weighted inner product of two frequency-domain waveforms.

    The first waveform ``h1`` is complex-conjugated. The result is:

    .. math::

        \\langle h_1, h_2 \\rangle = 4 \\Delta f \\sum_k \\frac{h_1^*(f_k)\\, h_2(f_k)}{S_n(f_k)}

    Args:
        h1 (Float[Array, " n_freq"]): First waveform (complex array).
        h2 (Float[Array, " n_freq"]): Second waveform (complex array).
        psd (Float[Array, " n_freq"]): One-sided power spectral density at each
            frequency bin.
        df (Float): Frequency bin spacing in Hz.

    Returns:
        Complex: Complex noise-weighted inner product. When ``h2`` is the detector
            data, this is the complex match-filtered SNR.
    """
    return 4.0 * jnp.sum(jnp.conj(h1) * h2 / psd) * df


def inner_product(
    h1: Float[Array, " n_freq"],
    h2: Float[Array, " n_freq"],
    psd: Float[Array, " n_freq"],
    df: Float,
) -> Float:
    """Compute the real noise-weighted inner product of two frequency-domain waveforms.

    Returns the real part of :func:`complex_inner_product`:

    .. math::

        (h_1 | h_2) = \\operatorname{Re}\\langle h_1, h_2 \\rangle
            = 4 \\Delta f \\sum_k \\operatorname{Re}\\!\\left[
              \\frac{h_1^*(f_k)\\, h_2(f_k)}{S_n(f_k)} \\right]

    Args:
        h1 (Float[Array, " n_freq"]): First waveform (complex array).
        h2 (Float[Array, " n_freq"]): Second waveform (complex array).
        psd (Float[Array, " n_freq"]): One-sided power spectral density at each
            frequency bin.
        df (Float): Frequency bin spacing in Hz.

    Returns:
        Float: Real noise-weighted inner product. When both waveforms are equal,
            this equals the optimal SNR squared.
    """
    return complex_inner_product(h1, h2, psd, df).real


def m1_m2_to_M_q(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the total mass M
    and mass ratio q.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Total mass (M_tot) and mass ratio (q).
    """
    M_tot = m1 + m2
    q = m2 / m1
    return M_tot, q


def M_q_to_m1_m2(M_tot: Float, q: Float) -> tuple[Float, Float]:
    """
    Transforms the total mass M and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Args:
        M_tot (Float): Total mass.
        q (Float): Mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_Mc_q(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and mass ratio q.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Chirp mass (M_c) and mass ratio (q).
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    M_c = M_tot * eta ** (3.0 / 5)
    q = m2 / m1
    return M_c, q


def Mc_q_to_m1_m2(M_c: Float, q: Float) -> tuple[Float, Float]:
    """
    Transforms the chirp mass M_c and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Args:
        M_c (Float): Chirp mass.
        q (Float): Mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    eta = q / (1 + q) ** 2
    M_tot = M_c / eta ** (3.0 / 5)
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_M_eta(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the total mass M
    and symmetric mass ratio eta.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Total mass (M) and symmetric mass ratio (eta).
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    return M_tot, eta


def M_eta_to_m1_m2(M_tot: Float, eta: Float) -> tuple[Float, Float]:
    """
    Transforms the total mass M and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Args:
        M_tot (Float): Total mass.
        eta (Float): Symmetric mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    m1 = M_tot * (1 + jnp.sqrt(1 - 4 * eta)) / 2
    m2 = M_tot * (1 - jnp.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def m1_m2_to_Mc_eta(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and symmetric mass ratio eta.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Chirp mass (M_c) and symmetric mass ratio (eta).
    """
    M = m1 + m2
    eta = m1 * m2 / M**2
    M_c = M * eta ** (3.0 / 5)
    return M_c, eta


def Mc_eta_to_m1_m2(M_c: Float, eta: Float) -> tuple[Float, Float]:
    """
    Transforming the chirp mass M_c and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Args:
        M_c (Float): Chirp mass.
        eta (Float): Symmetric mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    M = M_c / eta ** (3.0 / 5)
    m1 = M * (1 + jnp.sqrt(1 - 4 * eta)) / 2
    m2 = M * (1 - jnp.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def q_to_eta(q: Float) -> Float:
    """
    Transforming the chirp mass M_c and mass ratio q to the symmetric mass ratio eta.

    Args:
        q (Float): Mass ratio.

    Returns:
        Float: Symmetric mass ratio (eta).
    """
    eta = q / (1 + q) ** 2
    return eta


def eta_to_q(eta: Float) -> Float:
    """
    Transforming the symmetric mass ratio eta to the mass ratio q.

    Copied and modified from bilby/gw/conversion.py

    Args:
        eta (Float): Symmetric mass ratio.

    Returns:
        Float: Mass ratio (q).
    """
    temp = 1 / eta / 2 - 1
    return temp - (temp**2 - 1) ** 0.5


def euler_rotation(delta_x: Float[Array, "3"]) -> Float[Array, "3 3"]:
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Copied and modified from bilby-cython/geometry.pyx
    """
    norm = jnp.linalg.vector_norm(delta_x)

    cos_beta = delta_x[2] / norm
    sin_beta = jnp.sqrt(1 - cos_beta**2)

    alpha = jnp.arctan2(-delta_x[1] * cos_beta, delta_x[0])
    gamma = jnp.arctan2(delta_x[1], delta_x[0])

    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    cos_gamma = jnp.cos(gamma)
    sin_gamma = jnp.sin(gamma)

    rotation = jnp.array(
        [
            [
                cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma,
                -sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_gamma,
                sin_beta * cos_gamma,
            ],
            [
                cos_alpha * cos_beta * sin_gamma + sin_alpha * cos_gamma,
                -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma,
                sin_beta * sin_gamma,
            ],
            [-cos_alpha * sin_beta, sin_alpha * sin_beta, cos_beta],
        ]
    )

    return rotation


def angle_rotation(
    zenith: Float, azimuth: Float, rotation: Float[Array, "3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame
    to the polar angle and azimuthal angle in sky frame.

    Modified from bilby-cython/geometry.pyx.

    Args:
        zenith (Float): Zenith angle.
        azimuth (Float): Azimuthal angle.
        rotation (Float[Array, "3 3"]): The rotation matrix.

    Returns:
        tuple[Float, Float]: Polar angle (theta) and azimuthal angle (phi).
    """
    sky_loc_vec = jnp.array(
        [
            jnp.sin(zenith) * jnp.cos(azimuth),
            jnp.sin(zenith) * jnp.sin(azimuth),
            jnp.cos(zenith),
        ]
    )
    rotated_vec = jnp.einsum("ij,j...->i...", rotation, sky_loc_vec)

    theta = jnp.acos(rotated_vec[2])
    phi = jnp.fmod(
        jnp.arctan2(rotated_vec[1], rotated_vec[0]) + 2 * jnp.pi,
        2 * jnp.pi,
    )
    return theta, phi


def _theta_phi_to_ra_dec(theta: Float, phi: Float, gmst: Float) -> tuple[Float, Float]:
    """
    Transforming the polar angle and azimuthal angle to right ascension and declination.

    Args:
        theta (Float): Polar angle.
        phi (Float): Azimuthal angle.
        gmst (Float): Greenwich mean sidereal time.

    Returns:
        tuple[Float, Float]: Right ascension (ra) and declination (dec).
    """
    ra = phi + gmst
    dec = jnp.pi / 2 - theta
    ra = ra % (2 * jnp.pi)
    return ra, dec


def zenith_azimuth_to_ra_dec(
    zenith: Float, azimuth: Float, gmst: Float, rotation: Float[Array, "3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame to right ascension and declination.

    Copied and modified from bilby/gw/utils.py

    Args:
        zenith (Float): Zenith angle.
        azimuth (Float): Azimuthal angle.
        gmst (Float): Greenwich mean sidereal time.
        rotation (Float[Array, "3 3"]): The rotation matrix.

    Returns:
        tuple[Float, Float]: Right ascension (ra) and declination (dec).
    """
    theta, phi = angle_rotation(zenith, azimuth, rotation)
    ra, dec = _theta_phi_to_ra_dec(theta, phi, gmst)
    return ra, dec


def _ra_dec_to_theta_phi(ra: Float, dec: Float, gmst: Float) -> tuple[Float, Float]:
    """
    Transforming the right ascension ra and declination dec to the polar angle
    theta and azimuthal angle phi.

    Args:
        ra (Float): Right ascension.
        dec (Float): Declination.
        gmst (Float): Greenwich mean sidereal time.

    Returns:
        tuple[Float, Float]: Polar angle (theta) and azimuthal angle (phi).
    """
    phi = ra - gmst
    theta = jnp.pi / 2 - dec
    phi = (phi + 2 * jnp.pi) % (2 * jnp.pi)
    return theta, phi


def ra_dec_to_zenith_azimuth(
    ra: Float, dec: Float, gmst: Float, rotation: Float[Array, "3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the right ascension and declination to the zenith angle and azimuthal angle.

    Args:
        ra (Float): Right ascension.
        dec (Float): Declination.
        gmst (Float): Greenwich mean sidereal time.
        rotation (Float[Array, "3 3"]): The rotation matrix.

    Returns:
        tuple[Float, Float]: Zenith angle and azimuthal angle.
    """
    theta, phi = _ra_dec_to_theta_phi(ra, dec, gmst)
    zenith, azimuth = angle_rotation(theta, phi, rotation)
    return zenith, azimuth


def _rotate_y(angle: Float) -> Float[Array, "3 3"]:
    """Return the 3x3 rotation matrix for a rotation about the y-axis.

    Args:
        angle (Float): Rotation angle in radians.

    Returns:
        Float[Array, "3 3"]: Rotation matrix Ry(angle).
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.array(
        [[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]]
    )


def _rotate_z(angle: Float) -> Float[Array, "3 3"]:
    """Return the 3x3 rotation matrix for a rotation about the z-axis.

    Args:
        angle (Float): Rotation angle in radians.

    Returns:
        Float[Array, "3 3"]: Rotation matrix Rz(angle).
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
    )


def _Lmag_2PN(m1: Float, m2: Float, v0: Float) -> Float:
    """
    Compute the magnitude of the orbital angular momentum
    to 2 post-Newtonian orders.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.
        v0 (Float): Relative velocity at the reference frequency.

    Returns:
        Float: Magnitude of the orbital angular momentum.
    """
    eta = m1 * m2 / (m1 + m2) ** 2
    LN = (m1 + m2) * (m1 + m2) * eta / v0
    L_2PN = 1.5 + eta / 6.0
    return LN * (1.0 + v0 * v0 * L_2PN)


def spin_angles_to_cartesian_spin(
    theta_jn: Float,
    phi_jl: Float,
    tilt_1: Float,
    tilt_2: Float,
    phi_12: Float,
    chi_1: Float,
    chi_2: Float,
    M_c: Float,
    q: Float,
    fRef: Float,
    phiRef: Float,
) -> tuple[Float, Float, Float, Float, Float, Float, Float]:
    """
    Transforming the spin parameters.

    The code is based on the approach used in LALsimulation:
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html

    Args:
        theta_jn (Float): Zenith angle between the total angular momentum and the line of sight.
        phi_jl (Float): Difference between total and orbital angular momentum azimuthal angles.
        tilt_1 (Float): Zenith angle between the spin and orbital angular momenta for the primary object.
        tilt_2 (Float): Zenith angle between the spin and orbital angular momenta for the secondary object.
        phi_12 (Float): Difference between the azimuthal angles of the individual spin vector projections
            onto the orbital plane.
        chi_1 (Float): Primary object aligned spin.
        chi_2 (Float): Secondary object aligned spin.
        M_c (Float): The chirp mass.
        q (Float): The mass ratio.
        fRef (Float): The reference frequency.
        phiRef (Float): Binary phase at a reference frequency.

    Returns:
        tuple[Float, Float, Float, Float, Float, Float, Float]: Tuple of (iota, S1x, S1y, S1z, S2x, S2y, S2z):

            - iota: Zenith angle between the orbital angular momentum and the line of sight.
            - S1x: The x-component of the primary spin.
            - S1y: The y-component of the primary spin.
            - S1z: The z-component of the primary spin.
            - S2x: The x-component of the secondary spin.
            - S2y: The y-component of the secondary spin.
            - S2z: The z-component of the secondary spin.
    """

    # Starting frame: LNh along the z-axis
    # S1hat on the x-z plane
    LNh = jnp.array([0.0, 0.0, 1.0])

    # Define the spin vectors in the LNh frame
    s1hat = jnp.array(
        [
            jnp.sin(tilt_1) * jnp.cos(phiRef),
            jnp.sin(tilt_1) * jnp.sin(phiRef),
            jnp.cos(tilt_1),
        ]
    )
    s2hat = jnp.array(
        [
            jnp.sin(tilt_2) * jnp.cos(phi_12 + phiRef),
            jnp.sin(tilt_2) * jnp.sin(phi_12 + phiRef),
            jnp.cos(tilt_2),
        ]
    )

    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = jnp.cbrt((m1 + m2) * MTSUN * jnp.pi * fRef)

    # Define S1, S2, and J
    Lmag = _Lmag_2PN(m1, m2, v0)
    s1 = m1 * m1 * chi_1 * s1hat
    s2 = m2 * m2 * chi_2 * s2hat
    J = s1 + s2 + jnp.array([0.0, 0.0, Lmag])

    # Normalize J, and find theta0 and phi0 (the angles in starting frame)
    Jhat = J / jnp.linalg.norm(J)
    theta0, phi0 = carte_to_spherical_angles(*Jhat)

    # Rotations 1–3 combined.
    # LNh only needs R2 and R3 (it starts along z, so R1 has no effect).
    # s1hat and s2hat need all three: R3 @ R2 @ R1.
    R_23 = _rotate_z(phi_jl - jnp.pi) @ _rotate_y(-theta0)
    R_123 = R_23 @ _rotate_z(-phi0)
    LNh = R_23 @ LNh
    s1hat = R_123 @ s1hat
    s2hat = R_123 @ s2hat

    # Compute iota
    N = jnp.array([0.0, jnp.sin(theta_jn), jnp.cos(theta_jn)])
    iota = jnp.arccos(jnp.dot(N, LNh))

    thetaLJ, phiL = carte_to_spherical_angles(*LNh)

    # Rotations 4–6 combined.
    # N only needs R4 and R5; s1hat/s2hat need all three: R6 @ R5 @ R4.
    R_45 = _rotate_y(-thetaLJ) @ _rotate_z(-phiL)
    N = R_45 @ N
    phiN = safe_arctan2(N[1], N[0])
    R_456 = _rotate_z(jnp.pi / 2.0 - phiN - phiRef) @ R_45
    s1hat = R_456 @ s1hat
    s2hat = R_456 @ s2hat

    S1 = s1hat * chi_1
    S2 = s2hat * chi_2
    return iota, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2]


def cartesian_spin_to_spin_angles(
    iota: Float,
    S1x: Float,
    S1y: Float,
    S1z: Float,
    S2x: Float,
    S2y: Float,
    S2z: Float,
    M_c: Float,
    q: Float,
    fRef: Float,
    phiRef: Float,
) -> tuple[Float, Float, Float, Float, Float, Float, Float]:
    """
    Transforming the cartesian spin parameters to the spin angles.

    The code is based on the approach used in LALsimulation:
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html

    Args:
        iota (Float): Zenith angle between the orbital angular momentum and the line of sight.
        S1x (Float): The x-component of the primary spin.
        S1y (Float): The y-component of the primary spin.
        S1z (Float): The z-component of the primary spin.
        S2x (Float): The x-component of the secondary spin.
        S2y (Float): The y-component of the secondary spin.
        S2z (Float): The z-component of the secondary spin.
        M_c (Float): The chirp mass.
        q (Float): The mass ratio.
        fRef (Float): The reference frequency.
        phiRef (Float): The binary phase at the reference frequency.

    Returns:
        tuple[Float, ...]: Tuple of (theta_jn, phi_jl, tilt_1, tilt_2, phi_12, chi_1, chi_2):

            - theta_jn: Zenith angle between the total angular momentum and the line of sight.
            - phi_jl: Difference between total and orbital angular momentum azimuthal angles.
            - tilt_1: Zenith angle between the spin and orbital angular momenta for the primary object.
            - tilt_2: Zenith angle between the spin and orbital angular momenta for the secondary object.
            - phi_12: Difference between the azimuthal angles of the individual spin vector projections onto the orbital plane.
            - chi_1: Primary object aligned spin.
            - chi_2: Secondary object aligned spin.
    """
    # Starting frame: LNh along the z-axis
    LNh = jnp.array([0.0, 0.0, 1.0])

    # Define the dimensionless component spin vectors and magnitudes
    s1_vec = jnp.array([S1x, S1y, S1z])
    s2_vec = jnp.array([S2x, S2y, S2z])
    chi_1 = jnp.linalg.norm(s1_vec)
    chi_2 = jnp.linalg.norm(s2_vec)

    # Define the spin unit vectors in the LNh frame
    s1hat = jnp.where(chi_1 > 0, s1_vec / chi_1, jnp.zeros_like(s1_vec))
    s2hat = jnp.where(chi_2 > 0, s2_vec / chi_2, jnp.zeros_like(s2_vec))

    # Azimuthal and polar angles of the spin vectors
    tilt_1, phi1 = carte_to_spherical_angles(*s1hat, default_value=0.0)
    tilt_2, phi2 = carte_to_spherical_angles(*s2hat, default_value=0.0)

    phi_12 = phi2 - phi1
    phi_12 = (phi_12 + 2 * jnp.pi) % (2 * jnp.pi)  # Ensure 0 <= phi_12 < 2pi

    # Get angles in the J-N frame
    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = jnp.cbrt((m1 + m2) * MTSUN * jnp.pi * fRef)

    # Define S1, S2, J
    S1 = m1 * m1 * s1_vec
    S2 = m2 * m2 * s2_vec

    Lmag = _Lmag_2PN(m1, m2, v0)
    J = S1 + S2 + Lmag * LNh

    # Normalize J
    Jhat = J / jnp.linalg.norm(J)
    thetaJL, phiJ = carte_to_spherical_angles(*Jhat)

    # Azimuthal angle from phase angle
    phi0 = 0.5 * jnp.pi - phiRef
    # Line-of-sight vector in L-frame
    N = jnp.array(
        [jnp.sin(iota) * jnp.cos(phi0), jnp.sin(iota) * jnp.sin(phi0), jnp.cos(iota)]
    )

    # Inclination w.r.t. J
    theta_jn = jnp.arccos(jnp.dot(Jhat, N))

    # Rotate from L-frame to J-frame: combine the two rotations into one matrix.
    # N and LNh both need Ry(-thetaJL) @ Rz(-phiJ).
    R_JL = _rotate_y(-thetaJL) @ _rotate_z(-phiJ)
    N = R_JL @ N
    LNh = R_JL @ LNh

    phiN = safe_arctan2(N[1], N[0])
    LNh = _rotate_z(0.5 * jnp.pi - phiN) @ LNh

    phi_jl = safe_arctan2(LNh[1], LNh[0])
    phi_jl = (phi_jl + 2 * jnp.pi) % (2 * jnp.pi)  # Ensure 0 <= phi_jl < 2pi

    return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, chi_1, chi_2
