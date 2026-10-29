"""
High Precision Orbit Propagator (HPOP) for cislunar spacecraft.

Propagates spacecraft state in Moon-centered ICRF frame using
numerical integration with high-precision perturbation models.

Solar-system body positions come from a JPL DE kernel when one is available
(see `_configure_ephemeris`), otherwise from astropy's analytic 'builtin'
model, which differs from DE440 by up to ~18 km (~9 arcsec) for the Moon.
"""

import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from astropy.time import Time, TimeDelta
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric,
    get_body_barycentric_posvel,
    HeliocentricMeanEcliptic,
    GCRS,
    SkyCoord,
    CartesianRepresentation,
    CartesianDifferential,
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


# J2000 mean obliquity of the ecliptic (IAU 1976), degrees
OBLIQUITY_J2000_DEG = 23.4392911
AU_KM = 149597870.7

# Lunar harmonics are negligible beyond this Moon-centred distance
HARMONICS_CUTOFF_KM = 50000.0


# ---------------------------------------------------------------------------
# Solar-system ephemeris source
# ---------------------------------------------------------------------------

def _configure_ephemeris():
    """Select the solar-system ephemeris used by every astropy body lookup.

    Search order:
      1. $HPOP_EPHEMERIS (path to an SPK kernel, or the word 'builtin')
      2. ~/JPLDE/de440s.bsp  (shared local kernel store)
      3. <project dir>/de440s.bsp
      4. astropy 'builtin' analytic model
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    env = os.environ.get('HPOP_EPHEMERIS', '').strip()
    if env:
        candidates.append(env)
    candidates.append(os.path.expanduser('~/JPLDE/de440s.bsp'))
    candidates.append(os.path.join(here, 'de440s.bsp'))

    for cand in candidates:
        if cand.lower() == 'builtin':
            break
        if os.path.isfile(cand):
            try:
                solar_system_ephemeris.set('file://' + os.path.abspath(cand))
                print(f"[HPOP] solar-system ephemeris: {cand}")
                return os.path.basename(cand)
            except Exception as exc:  # pragma: no cover - depends on local files
                print(f"[HPOP] could not load ephemeris kernel {cand}: {exc}")

    solar_system_ephemeris.set('builtin')
    print("[HPOP] solar-system ephemeris: astropy builtin (analytic; Moon error up to ~18 km)")
    return 'builtin'


EPHEMERIS_NAME = _configure_ephemeris()


# ---------------------------------------------------------------------------
# Frame conversions
# ---------------------------------------------------------------------------

def _moon_relative_to_earth(epoch):
    """Moon position and velocity relative to Earth (ICRF), km and km/s.

    Works for a scalar Time (returns (3,) arrays) or a Time array of length N
    (returns (N, 3) arrays).
    """
    moon_pos, moon_vel = get_body_barycentric_posvel('moon', epoch)
    earth_pos, earth_vel = get_body_barycentric_posvel('earth', epoch)
    pos = (moon_pos.xyz - earth_pos.xyz).to_value(u.km)
    vel = (moon_vel.xyz - earth_vel.xyz).to_value(u.km / u.s)
    return np.moveaxis(pos, 0, -1), np.moveaxis(vel, 0, -1)


def convert_earth_to_moon(state_earth_icrf, epoch):
    """Convert state from Earth-centered ICRF to Moon-centered ICRF.

    Parameters
    ----------
    state_earth_icrf : ndarray (6,) or (N, 6)
        [x, y, z, vx, vy, vz] in Earth-centered ICRF, km and km/s.
    epoch : astropy.time.Time
        Scalar for a single state, or length-N array matching the rows.

    Returns
    -------
    state_moon_icrf : ndarray, same shape as input
    """
    moon_pos, moon_vel = _moon_relative_to_earth(epoch)
    out = np.array(state_earth_icrf, dtype=float, copy=True)
    out[..., :3] -= moon_pos
    out[..., 3:] -= moon_vel
    return out


def convert_moon_to_earth(state_moon_icrf, epoch):
    """Convert state from Moon-centered ICRF to Earth-centered ICRF.

    Accepts a single state with a scalar epoch, or an (N, 6) array with a
    length-N Time array (one ephemeris lookup for all rows).
    """
    moon_pos, moon_vel = _moon_relative_to_earth(epoch)
    out = np.array(state_moon_icrf, dtype=float, copy=True)
    out[..., :3] += moon_pos
    out[..., 3:] += moon_vel
    return out


def convert_helio_ecliptic_to_earth_icrf(state_helio_ecl, epoch):
    """Convert heliocentric J2000 mean ecliptic state to Earth-centered ICRF.

    Uses astropy's HeliocentricMeanEcliptic(equinox='J2000') -> GCRS
    transformation, which accounts for the J2000 mean obliquity (IAU 2006),
    the FK5/ICRS frame bias, and the heliocentric -> geocentric translation
    via the configured solar-system ephemeris.

    Parameters
    ----------
    state_helio_ecl : ndarray (6,)
        [x, y, z, vx, vy, vz] in heliocentric J2000 ecliptic, km and km/s.
    epoch : astropy.time.Time

    Returns
    -------
    state_earth_icrf : ndarray (6,)
        [x, y, z, vx, vy, vz] in Earth-centered ICRF (GCRS), km, km/s.
    """
    r = state_helio_ecl[:3]
    v = state_helio_ecl[3:]
    helio = SkyCoord(
        CartesianRepresentation(r[0]*u.km, r[1]*u.km, r[2]*u.km).with_differentials(
            CartesianDifferential(v[0]*u.km/u.s, v[1]*u.km/u.s, v[2]*u.km/u.s)),
        frame=HeliocentricMeanEcliptic(equinox='J2000', obstime=epoch),
    )
    gcrs = helio.transform_to(GCRS(obstime=epoch))
    r_gcrs = gcrs.cartesian.xyz.to_value(u.km)
    v_gcrs = gcrs.velocity.d_xyz.to_value(u.km / u.s)
    return np.concatenate([r_gcrs, v_gcrs])


# ---------------------------------------------------------------------------
# Orbital elements
# ---------------------------------------------------------------------------

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

    return {'a': float(a), 'e': float(e), 'i': float(inc),
            'raan': float(raan), 'argp': float(argp), 'nu': float(nu)}


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def _precompute_third_body_positions(t_start_jd_tdb, duration_sec, body_interval=600.0):
    """Tabulate Earth and Sun positions relative to the Moon and spline them.

    One vectorized ephemeris lookup per body; returns a single CubicSpline
    over the (N, 6) array [earth_xyz, sun_xyz] parameterised by elapsed
    seconds, so the ODE right-hand side needs one spline evaluation.
    """
    n_pts = max(int(duration_sec / body_interval) + 2, 10)
    dt_array = np.linspace(0.0, duration_sec, n_pts)
    t = Time(t_start_jd_tdb + dt_array / 86400.0, format='jd', scale='tdb')

    moon = get_body_barycentric('moon', t).xyz.to_value(u.km).T
    earth = get_body_barycentric('earth', t).xyz.to_value(u.km).T
    sun = get_body_barycentric('sun', t).xyz.to_value(u.km).T

    rel = np.hstack([earth - moon, sun - moon])  # (N, 6)
    return CubicSpline(dt_array, rel, axis=0)


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
        Propagation duration [seconds], must be positive.
    step_sec : float
        Output grid step [seconds]. The integrator chooses its own internal
        steps; the dense output ('sol') can be evaluated at any time within
        the span.
    config : HPOPConfig or None
        Propagation configuration.
    progress_callback : callable or None
        Unused; kept for API compatibility.

    Returns
    -------
    result : dict
        'times'        astropy Time array (N,)
        'states'       ndarray (N, 6) in km, km/s (Moon-centered ICRF)
        't_sec'        ndarray (N,) seconds since epoch
        'epoch'        initial epoch
        'duration_sec' total span in seconds
        'sol'          scipy OdeResult (use result['sol'].sol(t) for dense output)
        'n_rhs_evals'  number of right-hand-side evaluations
    """
    if config is None:
        config = HPOPConfig()
    if duration_sec <= 0:
        raise ValueError("Propagation duration must be positive")

    # Gravity coefficients
    C, S = get_coefficients(config.gravity_degree)
    degree = config.gravity_degree

    # Initial JD TDB
    t0_jd_tdb = time_to_jd_tdb(epoch)

    # Third-body ephemeris (single spline over [earth_xyz, sun_xyz])
    need_bodies = config.earth_gravity or config.sun_gravity or config.srp
    body_interp = _precompute_third_body_positions(t0_jd_tdb, duration_sec) if need_bodies else None

    use_harm = config.lunar_harmonics
    use_earth = config.earth_gravity
    use_sun = config.sun_gravity
    use_srp = config.srp
    cr = config.cr
    area_mass = config.area_mass_ratio

    def equations_of_motion(t_elapsed, y):
        """Right-hand side: t_elapsed in seconds since epoch, y = [r, v]."""
        pos = y[:3]
        r = np.sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])

        # Central body (Moon point-mass)
        accel = central_body_accel(pos, GM_MOON)

        # Lunar non-spherical harmonics
        if use_harm and r < HARMONICS_CUTOFF_KM:
            jd_tdb = t0_jd_tdb + t_elapsed / 86400.0
            R_bf = lunar_body_fixed_rotation(jd_tdb)
            a_harm_bf = spherical_harmonic_accel(R_bf @ pos, C, S, degree)
            accel = accel + R_bf.T @ a_harm_bf

        if body_interp is not None:
            bodies = body_interp(t_elapsed)
            if use_earth:
                accel = accel + third_body_accel(pos, bodies[:3], GM_EARTH)
            if use_sun:
                accel = accel + third_body_accel(pos, bodies[3:], GM_SUN)
            if use_srp:
                accel = accel + srp_accel(pos - bodies[3:], cr, area_mass)

        out = np.empty(6)
        out[:3] = y[3:6]
        out[3:] = accel
        return out

    # Output grid: multiples of step_sec, always ending exactly at duration_sec
    t_eval = np.arange(0.0, duration_sec, step_sec)
    if duration_sec - t_eval[-1] > 1e-6:
        t_eval = np.append(t_eval, duration_sec)
    else:
        t_eval[-1] = duration_sec

    sol = solve_ivp(
        equations_of_motion,
        (0.0, duration_sec),
        np.asarray(state0, dtype=float),
        method=config.integrator,
        t_eval=t_eval,
        rtol=config.rtol,
        atol=config.atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    states = sol.y.T  # (N, 6)
    times = epoch + TimeDelta(sol.t, format='sec')

    return {
        'times': times,
        'states': states,
        't_sec': sol.t,
        'epoch': epoch,
        'duration_sec': float(duration_sec),
        'sol': sol,
        'n_rhs_evals': int(sol.nfev),
    }
