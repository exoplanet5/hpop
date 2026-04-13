"""
High Precision Orbit Propagator (HPOP) for cislunar spacecraft.

Propagates spacecraft state in Moon-centered ICRF frame using
numerical integration with high-precision perturbation models.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from astropy.time import Time, TimeDelta
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric_posvel,
)
import astropy.units as u

from gravity import (
    GM_MOON, GM_EARTH, GM_SUN,
    lunar_body_fixed_rotation,
    spherical_harmonic_accel,
    central_body_accel,
    third_body_accel,
    srp_accel,
)
from lunar_coeffs import get_coefficients, R_MOON
from time_utils import parse_epoch, time_to_jd_tdb


def convert_earth_to_moon(state_earth_icrf, epoch):
    """Convert state from Earth-centered ICRF to Moon-centered ICRF.

    Parameters
    ----------
    state_earth_icrf : ndarray (6,)
        [x, y, z, vx, vy, vz] in Earth-centered ICRF, km and km/s.
    epoch : astropy.time.Time

    Returns
    -------
    state_moon_icrf : ndarray (6,)
        [x, y, z, vx, vy, vz] in Moon-centered ICRF, km and km/s.
    """
    solar_system_ephemeris.set('builtin')
    moon_pos, moon_vel = get_body_barycentric_posvel('moon', epoch)
    earth_pos, earth_vel = get_body_barycentric_posvel('earth', epoch)

    # Moon position/velocity relative to Earth (ICRS/ICRF)
    moon_geo_pos = (moon_pos.xyz - earth_pos.xyz).to(u.km).value
    moon_geo_vel = (moon_vel.xyz - earth_vel.xyz).to(u.km / u.s).value

    state_moon = np.zeros(6)
    state_moon[:3] = state_earth_icrf[:3] - moon_geo_pos
    state_moon[3:] = state_earth_icrf[3:] - moon_geo_vel
    return state_moon


def convert_moon_to_earth(state_moon_icrf, epoch):
    """Convert state from Moon-centered ICRF to Earth-centered ICRF.

    Parameters
    ----------
    state_moon_icrf : ndarray (6,)
    epoch : astropy.time.Time

    Returns
    -------
    state_earth_icrf : ndarray (6,)
    """
    solar_system_ephemeris.set('builtin')
    moon_pos, moon_vel = get_body_barycentric_posvel('moon', epoch)
    earth_pos, earth_vel = get_body_barycentric_posvel('earth', epoch)

    moon_geo_pos = (moon_pos.xyz - earth_pos.xyz).to(u.km).value
    moon_geo_vel = (moon_vel.xyz - earth_vel.xyz).to(u.km / u.s).value

    state_earth = np.zeros(6)
    state_earth[:3] = state_moon_icrf[:3] + moon_geo_pos
    state_earth[3:] = state_moon_icrf[3:] + moon_geo_vel
    return state_earth


def mean_to_true_anomaly(M_deg, e, tol=1e-14, max_iter=50):
    """Convert mean anomaly to true anomaly via Kepler's equation.

    Parameters
    ----------
    M_deg : float
        Mean anomaly [deg].
    e : float
        Eccentricity.

    Returns
    -------
    nu_deg : float
        True anomaly [deg].
    """
    M = np.radians(M_deg) % (2 * np.pi)
    # Solve Kepler's equation M = E - e*sin(E) with Newton-Raphson
    E = M + e * np.sin(M)  # initial guess
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    # E -> true anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                         np.sqrt(1 - e) * np.cos(E / 2))
    return np.degrees(nu) % 360


def keplerian_to_cartesian(a, e, i, raan, argp, nu, gm=GM_MOON):
    """Convert Keplerian elements to Cartesian state vector.

    Parameters
    ----------
    a : float
        Semi-major axis [km].
    e : float
        Eccentricity.
    i : float
        Inclination [deg].
    raan : float
        Right ascension of ascending node [deg].
    argp : float
        Argument of periapsis [deg].
    nu : float
        True anomaly [deg].
    gm : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    state : ndarray (6,)
        [x, y, z, vx, vy, vz] in km, km/s.
    """
    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)
    nu = np.radians(nu)

    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(nu))

    # Position and velocity in perifocal frame
    r_pf = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(gm / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix from perifocal to inertial
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R = np.array([
        [cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
         -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
         -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
         -cos_raan * sin_i],
        [sin_argp * sin_i,
         cos_argp * sin_i,
         cos_i]
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf

    return np.concatenate([r_eci, v_eci])


def cartesian_to_keplerian(state, gm=GM_MOON):
    """Convert Cartesian state vector to Keplerian elements.

    Parameters
    ----------
    state : ndarray (6,)
        [x, y, z, vx, vy, vz] in km, km/s.
    gm : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    elements : dict
        Keys: a, e, i, raan, argp, nu (angles in degrees).
    """
    r_vec = state[:3]
    v_vec = state[3:6]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = ((v ** 2 - gm / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / gm
    e = np.linalg.norm(e_vec)

    energy = v ** 2 / 2 - gm / r

    if abs(e - 1.0) > 1e-10:
        a = -gm / (2 * energy)
    else:
        a = float('inf')

    inc = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))

    if n > 1e-10:
        raan = np.degrees(np.arccos(np.clip(n_vec[0] / n, -1, 1)))
        if n_vec[1] < 0:
            raan = 360 - raan
    else:
        raan = 0.0

    if n > 1e-10 and e > 1e-10:
        argp = np.degrees(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1)))
        if e_vec[2] < 0:
            argp = 360 - argp
    else:
        argp = 0.0

    if e > 1e-10:
        nu = np.degrees(np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1)))
        if np.dot(r_vec, v_vec) < 0:
            nu = 360 - nu
    else:
        nu = 0.0

    return {'a': a, 'e': e, 'i': inc, 'raan': raan, 'argp': argp, 'nu': nu}


def _precompute_third_body_positions(t_start_jd_tdb, duration_sec, body_interval=600.0):
    """Precompute Earth and Sun positions relative to Moon at regular intervals.

    Returns cubic spline interpolators for each body.
    """
    solar_system_ephemeris.set('builtin')

    n_pts = max(int(duration_sec / body_interval) + 2, 10)
    dt_array = np.linspace(0, duration_sec, n_pts)
    jd_array = t_start_jd_tdb + dt_array / 86400.0

    earth_pos = np.zeros((n_pts, 3))
    sun_pos = np.zeros((n_pts, 3))

    for idx, jd in enumerate(jd_array):
        t = Time(jd, format='jd', scale='tdb')
        moon_bc = get_body_barycentric_posvel('moon', t)[0].xyz.to(u.km).value
        earth_bc = get_body_barycentric_posvel('earth', t)[0].xyz.to(u.km).value
        sun_bc = get_body_barycentric_posvel('sun', t)[0].xyz.to(u.km).value

        earth_pos[idx] = earth_bc - moon_bc
        sun_pos[idx] = sun_bc - moon_bc

    # Build cubic splines (parameter: elapsed seconds from t_start)
    earth_interp = [CubicSpline(dt_array, earth_pos[:, k]) for k in range(3)]
    sun_interp = [CubicSpline(dt_array, sun_pos[:, k]) for k in range(3)]

    return earth_interp, sun_interp


def _get_interp_pos(interp_list, t_elapsed):
    """Evaluate interpolated position."""
    return np.array([interp_list[k](t_elapsed) for k in range(3)])


class HPOPConfig:
    """Configuration for HPOP propagation."""

    def __init__(self):
        self.gravity_degree = 10
        self.lunar_harmonics = True
        self.earth_gravity = True
        self.sun_gravity = True
        self.srp = False
        self.cr = 1.5
        self.area_mass_ratio = 0.02  # m^2/kg
        self.integrator = 'DOP853'
        self.rtol = 1e-12
        self.atol = 1e-14


def propagate(state0, epoch, duration_sec, step_sec=60.0, config=None,
              progress_callback=None):
    """Propagate spacecraft state using HPOP.

    Parameters
    ----------
    state0 : ndarray (6,)
        Initial state [x, y, z, vx, vy, vz] in Moon-centered ICRF.
        Units: km, km/s.
    epoch : astropy.time.Time
        Initial epoch.
    duration_sec : float
        Propagation duration [seconds].
    step_sec : float
        Output step size [seconds].
    config : HPOPConfig or None
        Propagation configuration.
    progress_callback : callable or None
        Called with (fraction_complete,) for progress updates.

    Returns
    -------
    result : dict
        'times': list of astropy.time.Time
        'states': ndarray (N, 6) in km, km/s
        'keplerian': list of dicts with orbital elements
        'epoch': initial epoch
    """
    if config is None:
        config = HPOPConfig()

    # Precompute gravity coefficients
    C, S = get_coefficients(config.gravity_degree)

    # Get initial JD TDB
    t0_jd_tdb = time_to_jd_tdb(epoch)

    # Precompute third-body ephemeris
    earth_interp = sun_interp = None
    if config.earth_gravity or config.sun_gravity or config.srp:
        earth_interp, sun_interp = _precompute_third_body_positions(
            t0_jd_tdb, duration_sec
        )

    def equations_of_motion(t_elapsed, y):
        """Right-hand side of the ODE system.

        t_elapsed is seconds since epoch.
        y = [x, y, z, vx, vy, vz]
        """
        pos = y[:3]
        vel = y[3:6]
        r = np.linalg.norm(pos)

        # Central body (Moon point-mass)
        accel = central_body_accel(pos, GM_MOON)

        # Lunar non-spherical harmonics
        if config.lunar_harmonics and r < 50000.0:
            jd_tdb = t0_jd_tdb + t_elapsed / 86400.0
            R_bf = lunar_body_fixed_rotation(jd_tdb)
            pos_bf = R_bf @ pos
            a_harm_bf = spherical_harmonic_accel(pos_bf, C, S, config.gravity_degree)
            accel += R_bf.T @ a_harm_bf

        # Earth third-body
        if config.earth_gravity and earth_interp is not None:
            earth_pos = _get_interp_pos(earth_interp, t_elapsed)
            accel += third_body_accel(pos, earth_pos, GM_EARTH)

        # Sun third-body
        if config.sun_gravity and sun_interp is not None:
            sun_pos = _get_interp_pos(sun_interp, t_elapsed)
            accel += third_body_accel(pos, sun_pos, GM_SUN)

        # Solar radiation pressure
        if config.srp and sun_interp is not None:
            sun_pos = _get_interp_pos(sun_interp, t_elapsed)
            pos_sc_helio = pos - sun_pos
            accel += srp_accel(pos_sc_helio, config.cr, config.area_mass_ratio)

        return np.concatenate([vel, accel])

    # Integration time span
    t_span = (0.0, duration_sec)
    t_eval = np.arange(0, duration_sec + step_sec * 0.5, step_sec)

    # Solve ODE
    sol = solve_ivp(
        equations_of_motion,
        t_span,
        state0,
        method=config.integrator,
        t_eval=t_eval,
        rtol=config.rtol,
        atol=config.atol,
        max_step=step_sec * 10,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Build output
    states = sol.y.T  # (N, 6)
    times = epoch + TimeDelta(sol.t, format='sec')

    keplerian_list = []
    for s in states:
        try:
            kep = cartesian_to_keplerian(s, GM_MOON)
            keplerian_list.append(kep)
        except Exception:
            keplerian_list.append({
                'a': float('nan'), 'e': float('nan'), 'i': float('nan'),
                'raan': float('nan'), 'argp': float('nan'), 'nu': float('nan')
            })

    return {
        'times': times,
        'states': states,
        'keplerian': keplerian_list,
        'epoch': epoch,
        'sol': sol,
    }
