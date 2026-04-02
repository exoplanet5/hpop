"""
Gravity acceleration computation for cislunar HPOP.

Includes:
- Lunar non-spherical gravity (spherical harmonics up to configurable degree)
- Earth third-body gravity
- Sun third-body gravity
- Optional: Solar radiation pressure

All computations in Moon-centered ICRF frame.
Units: km, km/s, km/s^2
"""

import numpy as np
from lunar_coeffs import GM_MOON, R_MOON, get_coefficients

# Standard gravitational parameters (km^3/s^2)
GM_EARTH = 398600.4418
GM_SUN = 1.32712440018e+11


def lunar_body_fixed_rotation(t_jd_tdb):
    """Compute rotation matrix from ICRF inertial to Moon body-fixed (ME frame).

    Uses IAU 2009 model for the Moon's orientation.

    Parameters
    ----------
    t_jd_tdb : float
        Julian date in TDB scale.

    Returns
    -------
    R : ndarray (3,3)
        Rotation matrix from inertial to body-fixed.
    """
    T = (t_jd_tdb - 2451545.0) / 36525.0  # Julian centuries from J2000
    d = t_jd_tdb - 2451545.0  # Julian days from J2000

    # Fundamental arguments for libration (degrees)
    E1 = 125.045 - 0.0529921 * d
    E2 = 250.089 - 0.1059842 * d
    E3 = 260.008 + 13.0120009 * d
    E4 = 176.625 + 13.3407154 * d
    E5 = 357.529 + 0.9856003 * d
    E6 = 311.589 + 26.4057084 * d
    E7 = 134.963 + 13.0649930 * d
    E8 = 276.617 + 0.3287146 * d
    E9 = 34.226 + 1.7484877 * d
    E10 = 15.134 - 0.1589763 * d
    E11 = 119.743 + 0.0036096 * d
    E12 = 239.961 + 0.1643573 * d
    E13 = 25.053 + 12.9590088 * d

    # Convert to radians for trig
    E = np.radians([0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13])

    # Right ascension of Moon's north pole (degrees)
    alpha0 = (269.9949
              + 0.0031 * T
              - 3.8787 * np.sin(E[1])
              - 0.1204 * np.sin(E[2])
              + 0.0700 * np.sin(E[3])
              - 0.0172 * np.sin(E[4])
              + 0.0072 * np.sin(E[6])
              - 0.0052 * np.sin(E[10])
              + 0.0043 * np.sin(E[13]))

    # Declination of Moon's north pole (degrees)
    delta0 = (66.5392
              + 0.0130 * T
              + 1.5419 * np.cos(E[1])
              + 0.0239 * np.cos(E[2])
              - 0.0278 * np.cos(E[3])
              + 0.0068 * np.cos(E[4])
              - 0.0029 * np.cos(E[6])
              + 0.0009 * np.cos(E[7])
              + 0.0008 * np.cos(E[10])
              - 0.0009 * np.cos(E[13]))

    # Prime meridian angle W (degrees)
    W = (38.3213
         + 13.17635815 * d
         - 1.4e-12 * d * d
         + 3.5610 * np.sin(E[1])
         + 0.1208 * np.sin(E[2])
         - 0.0642 * np.sin(E[3])
         + 0.0158 * np.sin(E[4])
         + 0.0252 * np.sin(E[5])
         - 0.0066 * np.sin(E[6])
         - 0.0047 * np.sin(E[7])
         - 0.0046 * np.sin(E[8])
         + 0.0028 * np.sin(E[9])
         + 0.0052 * np.sin(E[10])
         + 0.0040 * np.sin(E[11])
         + 0.0019 * np.sin(E[12])
         - 0.0044 * np.sin(E[13]))

    # Convert to radians
    a = np.radians(alpha0)
    d_dec = np.radians(delta0)
    w = np.radians(W)

    # Rotation matrix: inertial -> body-fixed
    # R = Rz(W) * Rx(90-delta0) * Rz(90+alpha0)
    sa, ca = np.sin(a + np.pi / 2), np.cos(a + np.pi / 2)
    sd, cd = np.sin(np.pi / 2 - d_dec), np.cos(np.pi / 2 - d_dec)
    sw, cw = np.sin(w), np.cos(w)

    # Rz(90+alpha0)
    Rz1 = np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]])
    # Rx(90-delta0)
    Rx = np.array([[1, 0, 0], [0, cd, sd], [0, -sd, cd]])
    # Rz(W)
    Rz2 = np.array([[cw, sw, 0], [-sw, cw, 0], [0, 0, 1]])

    R = Rz2 @ Rx @ Rz1
    return R


def _legendre_and_deriv(max_degree, sin_phi, cos_phi):
    """Compute fully normalized associated Legendre functions and derivatives.

    Uses recursive algorithm for numerical stability.

    Parameters
    ----------
    max_degree : int
    sin_phi : float
        Sine of geocentric latitude.
    cos_phi : float
        Cosine of geocentric latitude.

    Returns
    -------
    P : ndarray (max_degree+1, max_degree+1)
        Fully normalized Legendre functions P_nm(sin_phi).
    dP : ndarray (max_degree+1, max_degree+1)
        Derivative of P_nm with respect to phi.
    """
    N = max_degree + 1
    P = np.zeros((N, N))
    dP = np.zeros((N, N))

    # Sectorial seed values
    P[0, 0] = 1.0
    dP[0, 0] = 0.0

    if max_degree >= 1:
        P[1, 0] = np.sqrt(3.0) * sin_phi
        P[1, 1] = np.sqrt(3.0) * cos_phi
        dP[1, 0] = np.sqrt(3.0) * cos_phi
        dP[1, 1] = -np.sqrt(3.0) * sin_phi

    for n in range(2, N):
        # Sectorial term P(n,n)
        fn = np.sqrt((2.0 * n + 1.0) / (2.0 * n))
        P[n, n] = fn * cos_phi * P[n - 1, n - 1]
        dP[n, n] = fn * (cos_phi * dP[n - 1, n - 1] - sin_phi * P[n - 1, n - 1])

        # Sub-sectorial P(n, n-1)
        gn = np.sqrt(2.0 * n + 1.0)
        P[n, n - 1] = gn * sin_phi * P[n - 1, n - 1]
        dP[n, n - 1] = gn * (sin_phi * dP[n - 1, n - 1] + cos_phi * P[n - 1, n - 1])

    # Zonal and tesseral terms via recursion
    for m in range(0, max_degree - 1):
        for n in range(m + 2, N):
            a_nm = np.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
            b_nm = np.sqrt(((2.0 * n + 1.0) * ((n - 1.0) ** 2 - m * m)) /
                           ((2.0 * n - 3.0) * (n * n - m * m)))
            P[n, m] = a_nm * sin_phi * P[n - 1, m] - b_nm * P[n - 2, m]
            dP[n, m] = a_nm * (sin_phi * dP[n - 1, m] + cos_phi * P[n - 1, m]) - b_nm * dP[n - 2, m]

    return P, dP


def spherical_harmonic_accel(pos_bf, C, S, max_degree, gm=GM_MOON, r_ref=R_MOON):
    """Compute gravitational acceleration from spherical harmonic expansion.

    Parameters
    ----------
    pos_bf : ndarray (3,)
        Position in body-fixed frame [km].
    C, S : ndarray
        Normalized Stokes coefficients.
    max_degree : int
        Maximum degree of expansion.
    gm : float
        Gravitational parameter [km^3/s^2].
    r_ref : float
        Reference radius [km].

    Returns
    -------
    accel_bf : ndarray (3,)
        Acceleration in body-fixed frame [km/s^2].
    """
    x, y, z = pos_bf
    r = np.sqrt(x * x + y * y + z * z)

    if r < r_ref * 0.5:
        return np.zeros(3)

    # Spherical coordinates
    xy = np.sqrt(x * x + y * y)
    sin_phi = z / r
    cos_phi = xy / r if xy > 1e-15 else 1e-15
    lam = np.arctan2(y, x)

    # Compute Legendre functions
    P, dP = _legendre_and_deriv(max_degree, sin_phi, cos_phi)

    # Precompute cos(m*lambda) and sin(m*lambda)
    cos_ml = np.zeros(max_degree + 1)
    sin_ml = np.zeros(max_degree + 1)
    for m in range(max_degree + 1):
        cos_ml[m] = np.cos(m * lam)
        sin_ml[m] = np.sin(m * lam)

    # Acceleration components in spherical coordinates
    # a_r = -dU/dr, a_phi = (1/r)*dU/dphi, a_lam = (1/(r*cos_phi))*dU/dlam
    a_r = 0.0
    a_phi = 0.0
    a_lam = 0.0

    rn = r_ref / r  # (R/r) accumulator
    for n in range(2, max_degree + 1):
        rn *= (r_ref / r)
        for m in range(0, n + 1):
            Cnm = C[n, m]
            Snm = S[n, m]
            cs = Cnm * cos_ml[m] + Snm * sin_ml[m]
            sc = -Cnm * sin_ml[m] + Snm * cos_ml[m]

            a_r += (n + 1) * rn * P[n, m] * cs
            a_phi += rn * dP[n, m] * cs
            a_lam += rn * m * P[n, m] * sc

    factor = gm / (r * r)
    a_r = -factor * a_r
    a_phi = factor * a_phi
    a_lam = factor * a_lam / cos_phi if cos_phi > 1e-15 else 0.0

    # Convert from spherical to Cartesian (body-fixed)
    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)

    # Unit vectors in Cartesian
    # e_r = [cos_phi*cos_lam, cos_phi*sin_lam, sin_phi]
    # e_phi = [-sin_phi*cos_lam, -sin_phi*sin_lam, cos_phi]
    # e_lam = [-sin_lam, cos_lam, 0]
    ax = (a_r * cos_phi * cos_lam
          + a_phi * (-sin_phi * cos_lam)
          + a_lam * (-sin_lam))
    ay = (a_r * cos_phi * sin_lam
          + a_phi * (-sin_phi * sin_lam)
          + a_lam * cos_lam)
    az = (a_r * sin_phi
          + a_phi * cos_phi)

    return np.array([ax, ay, az])


def central_body_accel(pos, gm=GM_MOON):
    """Point-mass gravitational acceleration.

    Parameters
    ----------
    pos : ndarray (3,)
        Position relative to central body [km].
    gm : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    accel : ndarray (3,)
        Acceleration [km/s^2].
    """
    r = np.linalg.norm(pos)
    if r < 1.0:
        return np.zeros(3)
    return -gm / (r ** 3) * pos


def third_body_accel(pos_sc, pos_body, gm_body):
    """Third-body gravitational perturbation acceleration.

    Parameters
    ----------
    pos_sc : ndarray (3,)
        Position of spacecraft relative to central body [km].
    pos_body : ndarray (3,)
        Position of perturbing body relative to central body [km].
    gm_body : float
        Gravitational parameter of perturbing body [km^3/s^2].

    Returns
    -------
    accel : ndarray (3,)
        Perturbation acceleration [km/s^2].
    """
    d = pos_body - pos_sc  # vector from s/c to perturbing body
    r_body = np.linalg.norm(pos_body)
    d_norm = np.linalg.norm(d)

    if d_norm < 1.0 or r_body < 1.0:
        return np.zeros(3)

    return gm_body * (d / d_norm ** 3 - pos_body / r_body ** 3)


def srp_accel(pos_sc_helio, cr=1.5, area_mass_ratio=0.02):
    """Solar radiation pressure acceleration (simplified).

    Parameters
    ----------
    pos_sc_helio : ndarray (3,)
        Position of spacecraft relative to Sun [km].
    cr : float
        Radiation pressure coefficient.
    area_mass_ratio : float
        Area-to-mass ratio [m^2/kg].

    Returns
    -------
    accel : ndarray (3,)
        SRP acceleration [km/s^2].
    """
    P_sun = 4.56e-6  # Solar radiation pressure at 1 AU [N/m^2]
    AU_km = 1.495978707e8  # 1 AU in km

    r = np.linalg.norm(pos_sc_helio)
    if r < 1.0:
        return np.zeros(3)

    # Scale with distance squared
    factor = P_sun * (AU_km / r) ** 2 * cr * area_mass_ratio
    # Convert N/m^2 * m^2/kg = m/s^2 -> km/s^2
    factor *= 1e-3

    return factor * pos_sc_helio / r
