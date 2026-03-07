"""
Observation ephemeris computation.

Computes RA/DEC of a cislunar spacecraft as seen from a ground-based observer,
given the observer's geodetic coordinates or MPC observatory code.
"""

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import (
    EarthLocation,
    GCRS,
    ICRS,
    CartesianRepresentation,
    solar_system_ephemeris,
    get_body_barycentric_posvel,
)
import astropy.units as u


# Known MPC observatory codes (a selection of major ones)
# Full list can be queried from astropy/MPC, but we include common ones
MPC_OBSERVATORIES = {
    '500': {'name': 'Geocenter', 'lat': 0.0, 'lon': 0.0, 'alt': 0.0},
}


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
        code = observer_config['mpc_code']
        try:
            return EarthLocation.of_site(code)
        except Exception:
            pass
        # Try as MPC observatory code
        try:
            from astroquery.mpc import MPC
            obs_data = MPC.get_observatory_location(code)
            # obs_data returns (lon, cos_phi, sin_phi) in specific format
            lon = obs_data[0]  # degrees
            cos_phi = obs_data[1]
            sin_phi = obs_data[2]
            # Convert parallax constants to geodetic lat/alt
            # cos_phi and sin_phi are geocentric parallax constants in Earth radii
            R_earth = 6378.137  # km, equatorial radius
            rho = np.sqrt(cos_phi**2 + sin_phi**2) * R_earth
            lat_gc = np.degrees(np.arctan2(sin_phi, cos_phi))
            return EarthLocation.from_geocentric(
                rho * np.cos(np.radians(lat_gc)) * np.cos(np.radians(lon)) * u.km,
                rho * np.cos(np.radians(lat_gc)) * np.sin(np.radians(lon)) * u.km,
                rho * np.sin(np.radians(lat_gc)) * u.km,
            )
        except Exception as e:
            raise ValueError(f"Cannot resolve MPC code '{code}': {e}")
    else:
        lat = observer_config.get('lat', 0.0)
        lon = observer_config.get('lon', 0.0)
        alt = observer_config.get('alt', 0.0)
        return EarthLocation.from_geodetic(
            lon=lon * u.deg, lat=lat * u.deg, height=alt * u.m
        )


def compute_observation_ephemeris(prop_result, observer_config, step_sec=60.0):
    """Compute RA/DEC ephemeris for an observer.

    Parameters
    ----------
    prop_result : dict
        Output from propagator.propagate().
    observer_config : dict
        Observer location configuration.
    step_sec : float
        Output step in seconds (default 60s = 1 min).

    Returns
    -------
    ephemeris : dict
        'times_utc': list of ISO strings
        'ra_deg': list of RA in degrees
        'dec_deg': list of DEC in degrees
        'ra_hms': list of RA in HH:MM:SS.ss
        'dec_dms': list of DEC in +DD:MM:SS.s
        'distance_km': list of observer-to-s/c distances
        'altitude_deg': list of altitude above horizon (if applicable)
        'azimuth_deg': list of azimuth
    """
    solar_system_ephemeris.set('builtin')

    location = get_observer_location(observer_config)
    prop_times = prop_result['times']
    prop_states = prop_result['states']  # Moon-centered J2000, km

    # Determine output times at step_sec intervals
    t_start = prop_times[0]
    t_end = prop_times[-1]
    total_sec = (t_end - t_start).sec
    n_steps = int(total_sec / step_sec) + 1
    dt_array = np.arange(n_steps) * step_sec
    out_times = t_start + TimeDelta(dt_array, format='sec')

    # Interpolate spacecraft states at output times
    prop_t_sec = (prop_times - t_start).sec
    out_t_sec = dt_array

    # Linear interpolation of states
    from scipy.interpolate import CubicSpline
    if len(prop_t_sec) > 3:
        state_interp = CubicSpline(prop_t_sec, prop_states, axis=0)
    else:
        from scipy.interpolate import interp1d
        state_interp = interp1d(prop_t_sec, prop_states, axis=0, fill_value='extrapolate')

    ra_list = []
    dec_list = []
    ra_hms_list = []
    dec_dms_list = []
    dist_list = []
    alt_list = []
    az_list = []
    time_str_list = []

    for idx, t in enumerate(out_times):
        t_sec = out_t_sec[idx]
        sc_moon = state_interp(t_sec)[:3]  # Moon-centered J2000 position [km]

        # Get Moon position in GCRS (Earth-centered)
        moon_bc = get_body_barycentric_posvel('moon', t)[0]
        earth_bc = get_body_barycentric_posvel('earth', t)[0]
        moon_geo = (moon_bc.xyz - earth_bc.xyz).to(u.km).value  # km, ICRS

        # Spacecraft position in GCRS (Earth-centered ICRS ≈ GCRS for this purpose)
        sc_gcrs_km = moon_geo + sc_moon  # km

        # Observer position in GCRS
        obs_gcrs = location.get_gcrs(t)
        obs_xyz_km = np.array([
            obs_gcrs.cartesian.x.to(u.km).value,
            obs_gcrs.cartesian.y.to(u.km).value,
            obs_gcrs.cartesian.z.to(u.km).value,
        ])

        # Topocentric vector (GCRS)
        topo = sc_gcrs_km - obs_xyz_km
        dist = np.linalg.norm(topo)

        # Convert to RA/DEC
        topo_unit = topo / dist
        dec = np.degrees(np.arcsin(np.clip(topo_unit[2], -1, 1)))
        ra = np.degrees(np.arctan2(topo_unit[1], topo_unit[0])) % 360

        # Format RA as HH:MM:SS.ss
        ra_h = ra / 15.0
        ra_hh = int(ra_h)
        ra_mm = int((ra_h - ra_hh) * 60)
        ra_ss = (ra_h - ra_hh - ra_mm / 60.0) * 3600
        ra_hms = f"{ra_hh:02d}:{ra_mm:02d}:{ra_ss:05.2f}"

        # Format DEC as +DD:MM:SS.s
        dec_sign = '+' if dec >= 0 else '-'
        dec_abs = abs(dec)
        dec_dd = int(dec_abs)
        dec_mm = int((dec_abs - dec_dd) * 60)
        dec_ss = (dec_abs - dec_dd - dec_mm / 60.0) * 3600
        dec_dms = f"{dec_sign}{dec_dd:02d}:{dec_mm:02d}:{dec_ss:04.1f}"

        # Compute altitude/azimuth (approximate)
        from astropy.coordinates import SkyCoord, AltAz
        try:
            sc_sky = SkyCoord(
                ra=ra * u.deg, dec=dec * u.deg,
                distance=dist * u.km, frame='icrs'
            )
            altaz = sc_sky.transform_to(AltAz(obstime=t, location=location))
            alt_list.append(round(altaz.alt.deg, 4))
            az_list.append(round(altaz.az.deg, 4))
        except Exception:
            alt_list.append(None)
            az_list.append(None)

        ra_list.append(round(ra, 6))
        dec_list.append(round(dec, 6))
        ra_hms_list.append(ra_hms)
        dec_dms_list.append(dec_dms)
        dist_list.append(round(dist, 3))
        time_str_list.append(t.utc.isot)

    return {
        'times_utc': time_str_list,
        'ra_deg': ra_list,
        'dec_deg': dec_list,
        'ra_hms': ra_hms_list,
        'dec_dms': dec_dms_list,
        'distance_km': dist_list,
        'altitude_deg': alt_list,
        'azimuth_deg': az_list,
    }
