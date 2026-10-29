"""
Observation ephemeris computation.

Computes topocentric RA/DEC (plus rates, Alt/Az, lunar elongation and a
day/night flag) of a cislunar spacecraft as seen from a ground-based observer,
given the observer's geodetic coordinates or MPC observatory code.

Positions are instantaneous geometric topocentric vectors: no light-time,
aberration or refraction corrections are applied.  All epochs of one
ephemeris are processed in a single vectorized pass.
"""

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from astropy.time import TimeDelta
from astropy.coordinates import (
    EarthLocation,
    GCRS,
    AltAz,
    SkyCoord,
    CartesianRepresentation,
    get_body_barycentric,
    get_body,
)
import astropy.units as u

# Importing propagator selects the solar-system ephemeris source (DE kernel
# or astropy builtin) for every body lookup made below.
from propagator import EPHEMERIS_NAME  # noqa: F401


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _observer_location_cached(kind, a, b, c):
    if kind == 'mpc':
        code = a
        try:
            return EarthLocation.of_site(code)
        except Exception:
            pass
        try:
            from astroquery.mpc import MPC
            obs_data = MPC.get_observatory_location(code)
            lon = obs_data[0]
            cos_phi = obs_data[1]
            sin_phi = obs_data[2]
            R_earth = 6378.137
            rho = np.sqrt(cos_phi**2 + sin_phi**2) * R_earth
            lat_gc = np.degrees(np.arctan2(sin_phi, cos_phi))
            return EarthLocation.from_geocentric(
                rho * np.cos(np.radians(lat_gc)) * np.cos(np.radians(lon)) * u.km,
                rho * np.cos(np.radians(lat_gc)) * np.sin(np.radians(lon)) * u.km,
                rho * np.sin(np.radians(lat_gc)) * u.km,
            )
        except Exception as e:
            raise ValueError(f"Cannot resolve MPC code '{code}': {e}")
    return EarthLocation.from_geodetic(lon=b * u.deg, lat=a * u.deg, height=c * u.m)


def get_observer_location(observer_config):
    """Get EarthLocation from observer configuration.

    Parameters
    ----------
    observer_config : dict
        Either {'type': 'geodetic', 'lat': deg, 'lon': deg, 'alt': m}
        or {'type': 'mpc', 'mpc_code': 'XXX'}

    Returns
    -------
    location : astropy.coordinates.EarthLocation
    """
    if observer_config.get('type') == 'mpc':
        code = str(observer_config.get('mpc_code', '')).strip()
        if not code:
            raise ValueError("MPC observatory code is empty")
        return _observer_location_cached('mpc', code, None, None)
    lat = float(observer_config.get('lat', 0.0))
    lon = float(observer_config.get('lon', 0.0))
    alt = float(observer_config.get('alt', 0.0))
    return _observer_location_cached('geodetic', lat, lon, alt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_hms(ra_deg):
    """RA in degrees -> 'HH:MM:SS.ss' with correct carry at 60 s."""
    tot = int(round(ra_deg / 15.0 * 360000.0)) % (24 * 360000)
    h, rem = divmod(tot, 360000)
    m, cs = divmod(rem, 6000)
    return f"{h:02d}:{m:02d}:{cs / 100.0:05.2f}"


def _fmt_dms(dec_deg):
    """DEC in degrees -> '+DD:MM:SS.s' with correct carry at 60 arcsec."""
    sign = '-' if dec_deg < 0 else '+'
    tot = int(round(abs(dec_deg) * 36000.0))
    d, rem = divmod(tot, 36000)
    m, ds = divmod(rem, 600)
    return f"{sign}{d:02d}:{m:02d}:{ds / 10.0:04.1f}"


def _sky_condition(sun_alt):
    """Classify sky condition from Sun altitude (deg) -> str."""
    if sun_alt is None or not np.isfinite(sun_alt):
        return 'unknown'
    if sun_alt > 0:
        return 'day'
    elif sun_alt > -6:
        return 'civil_tw'
    elif sun_alt > -12:
        return 'nautical_tw'
    elif sun_alt > -18:
        return 'astro_tw'
    return 'night'


def _sky_rates(ra_deg, dec_deg, step_sec):
    """Finite-difference sky-motion rates on a uniform time grid.

    Returns (ra_rate, dec_rate, total_rate, pa):
      ra_rate   d(RA)/dt      arcsec/min (coordinate rate, no cos(dec))
      dec_rate  d(DEC)/dt     arcsec/min
      total     great-circle rate, arcsec/min (RA term scaled by cos(dec))
      pa        position angle of motion, deg, N through E
    Central differences inside, one-sided at the ends.
    """
    n = len(ra_deg)
    if n < 2 or step_sec <= 0:
        z = np.zeros(n)
        return z, z, z, z
    dt_min = step_sec / 60.0
    ra_unwrapped = np.unwrap(ra_deg, period=360.0)
    dra = np.gradient(ra_unwrapped, dt_min)      # deg/min
    ddec = np.gradient(dec_deg, dt_min)          # deg/min
    ra_rate = dra * 3600.0
    dec_rate = ddec * 3600.0
    dra_sky = dra * np.cos(np.radians(dec_deg)) * 3600.0
    total = np.hypot(dra_sky, dec_rate)
    pa = np.degrees(np.arctan2(dra_sky, dec_rate)) % 360.0
    return ra_rate, dec_rate, total, pa


def _spacecraft_states(prop_result, dt_array):
    """Moon-centred states at elapsed seconds dt_array, shape (N, 6).

    Prefers the integrator's dense output (7th-order DOP853 interpolant);
    falls back to a cubic spline through the output grid.
    """
    sol = prop_result.get('sol')
    if sol is not None and getattr(sol, 'sol', None) is not None:
        return sol.sol(dt_array).T
    prop_times = prop_result['times']
    t_sec = prop_result.get('t_sec')
    if t_sec is None:
        t_sec = (prop_times - prop_times[0]).sec
    states = prop_result['states']
    if len(t_sec) > 3:
        return CubicSpline(t_sec, states, axis=0)(dt_array)
    return interp1d(t_sec, states, axis=0, fill_value='extrapolate')(dt_array)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_observation_ephemeris(prop_result, observer_config, step_sec=60.0,
                                  t_start_offset=0.0, t_duration=None):
    """Compute RA/DEC ephemeris for an observer.

    Parameters
    ----------
    prop_result : dict
        Output from propagator.propagate().
    observer_config : dict
        Observer location configuration.
    step_sec : float
        Output step in seconds (default 60s = 1 min).
    t_start_offset : float
        Offset in seconds from propagation epoch to start ephemeris output.
    t_duration : float or None
        Duration in seconds for ephemeris output. If None, uses full propagation range.

    Returns
    -------
    ephemeris : dict of lists (one entry per output epoch)
    """
    location = get_observer_location(observer_config)
    prop_times = prop_result['times']
    prop_epoch = prop_times[0]
    prop_total_sec = float(prop_result.get('duration_sec',
                                           (prop_times[-1] - prop_epoch).sec))

    # Output window inside the propagated span
    step_sec = float(step_sec)
    if step_sec <= 0:
        raise ValueError("Ephemeris step must be positive")
    ephem_start_sec = max(0.0, float(t_start_offset))
    if t_duration is not None:
        ephem_end_sec = min(ephem_start_sec + float(t_duration), prop_total_sec)
    else:
        ephem_end_sec = prop_total_sec
    n_steps = int(np.floor((ephem_end_sec - ephem_start_sec) / step_sec + 1e-9)) + 1
    dt_array = ephem_start_sec + np.arange(max(n_steps, 1)) * step_sec
    dt_array = dt_array[dt_array <= prop_total_sec + 1e-6]
    n = len(dt_array)
    out_times = prop_epoch + TimeDelta(dt_array, format='sec')

    # Spacecraft Moon-centred positions (N, 3)
    sc_moon = _spacecraft_states(prop_result, dt_array)[:, :3]

    # Moon relative to Earth (ICRF) at all epochs, one lookup per body
    moon_bc = get_body_barycentric('moon', out_times).xyz.to_value(u.km).T
    earth_bc = get_body_barycentric('earth', out_times).xyz.to_value(u.km).T
    moon_geo = moon_bc - earth_bc                       # (N, 3)
    sc_gcrs = moon_geo + sc_moon                        # (N, 3)

    # Observer in GCRS (includes Earth rotation, precession/nutation, polar motion)
    obs_gcrs = location.get_gcrs(out_times).cartesian.xyz.to_value(u.km).T

    # Topocentric geometric vector -> RA/DEC
    topo = sc_gcrs - obs_gcrs
    dist = np.linalg.norm(topo, axis=1)
    topo_unit = topo / dist[:, None]
    dec = np.degrees(np.arcsin(np.clip(topo_unit[:, 2], -1.0, 1.0)))
    ra = np.degrees(np.arctan2(topo_unit[:, 1], topo_unit[:, 0])) % 360.0

    # Alt/Az from the GCRS Cartesian position (topocentric transform)
    altaz_frame = AltAz(obstime=out_times, location=location)
    try:
        sc_coord = SkyCoord(
            CartesianRepresentation(sc_gcrs[:, 0] * u.km,
                                    sc_gcrs[:, 1] * u.km,
                                    sc_gcrs[:, 2] * u.km),
            frame=GCRS(obstime=out_times),
        )
        sc_altaz = sc_coord.transform_to(altaz_frame)
        alt = np.round(np.atleast_1d(sc_altaz.alt.deg), 4).tolist()
        az = np.round(np.atleast_1d(sc_altaz.az.deg), 4).tolist()
    except Exception:
        alt = [None] * n
        az = [None] * n

    # Sun altitude -> sky condition flag
    try:
        sun_alt = np.atleast_1d(
            get_body('sun', out_times, location).transform_to(altaz_frame).alt.deg)
        sky_flags = [_sky_condition(float(s)) for s in sun_alt]
    except Exception:
        sky_flags = ['unknown'] * n

    # Lunar elongation: angle between target and Moon centre as seen by observer
    topo_moon = moon_geo - obs_gcrs
    dist_moon = np.linalg.norm(topo_moon, axis=1)
    cos_elong = np.sum(topo_unit * topo_moon / dist_moon[:, None], axis=1)
    lunar_elong = np.degrees(np.arccos(np.clip(cos_elong, -1.0, 1.0)))

    # Sky-motion rates and position angle
    ra_rate, dec_rate, total_rate, pa = _sky_rates(ra, dec, step_sec)

    return {
        'times_utc': [t for t in np.atleast_1d(out_times.utc.isot)],
        'ra_deg': np.round(ra, 6).tolist(),
        'dec_deg': np.round(dec, 6).tolist(),
        'ra_hms': [_fmt_hms(x) for x in ra],
        'dec_dms': [_fmt_dms(x) for x in dec],
        'ra_rate': np.round(ra_rate, 4).tolist(),
        'dec_rate': np.round(dec_rate, 4).tolist(),
        'total_rate': np.round(total_rate, 4).tolist(),
        'pa': np.round(pa, 2).tolist(),
        'lunar_elong': np.round(lunar_elong, 4).tolist(),
        'distance_km': np.round(dist, 3).tolist(),
        'altitude_deg': alt,
        'azimuth_deg': az,
        'sky_flag': sky_flags,
    }
