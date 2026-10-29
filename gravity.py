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


# ---------------------------------------------------------------------------
# Spherical harmonics
# ---------------------------------------------------------------------------

_DEGREE_CACHE = {}


def _degree_tables(max_degree):
    """Return cached, degree-dependent constant arrays for the recursions and sums.

    Returns
    -------
    a, b : ndarray (N, N)
        Recursion coefficients for P(n, m) from P(n-1, m) and P(n-2, m).
    fn, gn : ndarray (N,)
        Sectorial / sub-sectorial seed coefficients.
    n_col : ndarray (N, 1), m_row : ndarray (1, N)
        Degree and order index arrays for broadcasting.
    mask : ndarray (N, N)
        1.0 where 2 <= n and m <= n, else 0.0.
    """
    tables = _DEGREE_CACHE.get(max_degree)
    if tables is None:
        N = max_degree + 1
        n_col = np.arange(N, dtype=float)[:, None]
        m_row = np.arange(N, dtype=float)[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            a = np.sqrt((4.0 * n_col * n_col - 1.0) / (n_col * n_col - m_row * m_row))
            b = np.sqrt(((2.0 * n_col + 1.0) * ((n_col - 1.0) ** 2 - m_row * m_row))
                        / ((2.0 * n_col - 3.0) * (n_col * n_col - m_row * m_row)))
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        n_vec = np.arange(N, dtype=float)
        fn = np.sqrt((2.0 * n_vec + 1.0) / (2.0 * np.maximum(n_vec, 1.0)))
        gn = np.sqrt(2.0 * n_vec + 1.0)
        mask = np.tril(np.ones((N, N)))
        mask[:2, :] = 0.0
        tables = (a, b, fn, gn, n_col, m_row, mask)
        _DEGREE_CACHE[max_degree] = tables
    return tables


def _legendre_and_deriv(max_degree, sin_phi, cos_phi):
    """Compute fully normalized associated Legendre functions and derivatives.

    Standard forward recursion in degree n; each degree is evaluated for all
    orders m at once, so the Python loop runs over n only.

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
    a, b, fn, gn, _, _, _ = _degree_tables(max_degree)

    P[0, 0] = 1.0
    if max_degree >= 1:
        s3 = np.sqrt(3.0)
        P[1, 0] = s3 * sin_phi
        P[1, 1] = s3 * cos_phi
        dP[1, 0] = s3 * cos_phi
        dP[1, 1] = -s3 * sin_phi

    for n in range(2, N):
        # Sectorial term P(n,n)
        P[n, n] = fn[n] * cos_phi * P[n - 1, n - 1]
        dP[n, n] = fn[n] * (cos_phi * dP[n - 1, n - 1] - sin_phi * P[n - 1, n - 1])
        # Sub-sectorial P(n, n-1)
        P[n, n - 1] = gn[n] * sin_phi * P[n - 1, n - 1]
        dP[n, n - 1] = gn[n] * (sin_phi * dP[n - 1, n - 1] + cos_phi * P[n - 1, n - 1])
        # Zonal and tesseral terms, m = 0 .. n-2, all at once
        mm = slice(0, n - 1)
        an = a[n, mm]
        bn = b[n, mm]
        P[n, mm] = an * sin_phi * P[n - 1, mm] - bn * P[n - 2, mm]
        dP[n, mm] = an * (sin_phi * dP[n - 1, mm] + cos_phi * P[n - 1, mm]) - bn * dP[n - 2, mm]

    return P, dP


def spherical_harmonic_accel(pos_bf, C, S, max_degree, gm=GM_MOON, r_ref=R_MOON):
    """Compute gravitational acceleration from spherical harmonic expansion.

    Only the non-spherical part (degree >= 2) is returned; the central
    point-mass term is applied separately.

    Parameters
    ----------
    pos_bf : ndarray (3,)
        Position in body-fixed frame [km].
    C, S : ndarray
        Normalized Stokes coefficients, at least (max_degree+1, max_degree+1).
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

    N = max_degree + 1
    P, dP = _legendre_and_deriv(max_degree, sin_phi, cos_phi)
    _, _, _, _, n_col, m_row, mask = _degree_tables(max_degree)

    ml = m_row * lam
    cos_ml = np.cos(ml)
    sin_ml = np.sin(ml)

    Cn = C[:N, :N]
    Sn = S[:N, :N]
    cs = Cn * cos_ml + Sn * sin_ml
    sc = Sn * cos_ml - Cn * sin_ml

    # (R/r)^n weights, zero for n < 2 and m > n
    W = mask * (r_ref / r) ** n_col

    # a_r = -dU/dr, a_phi = (1/r)*dU/dphi, a_lam = (1/(r*cos_phi))*dU/dlam
    a_r = np.sum(W * (n_col + 1.0) * P * cs)
    a_phi = np.sum(W * dP * cs)
    a_lam = np.sum(W * m_row * P * sc)

    factor = gm / (r * r)
    a_r = -factor * a_r
    a_phi = factor * a_phi
    a_lam = factor * a_lam / cos_phi if cos_phi > 1e-15 else 0.0

    # Convert from spherical to Cartesian (body-fixed)
    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)
    ax = a_r * cos_phi * cos_lam - a_phi * sin_phi * cos_lam - a_lam * sin_lam
    ay = a_r * cos_phi * sin_lam - a_phi * sin_phi * sin_lam + a_lam * cos_lam
    az = a_r * sin_phi + a_phi * cos_phi

    return np.array([ax, ay, az])


# ---------------------------------------------------------------------------
# Point-mass, third-body and SRP terms
# ---------------------------------------------------------------------------

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

    Includes the indirect term (acceleration of the central body toward the
    perturbing body), as required in a central-body-centred frame.

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
    """Solar radiation pressure acceleration (cannonball model, no eclipses).

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
