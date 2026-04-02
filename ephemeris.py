"""
Observation ephemeris computation.

Computes RA/DEC of a cislunar spacecraft as seen from a ground-based observer,
given the observer's geodetic coordinates or MPC observatory code.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from astropy.time import Time, TimeDelta
from astropy.coordinates import (
    EarthLocation,
    GCRS,
    AltAz,
    SkyCoord,
    CartesianRepresentation,
    solar_system_ephemeris,
    get_body_barycentric_posvel,
    get_body,
)
import astropy.units as u


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
    else:
        lat = observer_config.get('lat', 0.0)
        lon = observer_config.get('lon', 0.0)
        alt = observer_config.get('alt', 0.0)
        return EarthLocation.from_geodetic(
            lon=lon * u.deg, lat=lat * u.deg, height=alt * u.m
        )


def _sun_altitude(t, location):
    """Compute Sun altitude in degrees at given time and location."""
    try:
        altaz_frame = AltAz(obstime=t, location=location)
        sun_altaz = get_body('sun', t, location).transform_to(altaz_frame)
        return sun_altaz.alt.deg
    except Exception:
        return None


def _sky_condition(sun_alt):
    """Classify sky condition based on Sun altitude.

    Returns
    -------
    str : 'day', 'civil_tw', 'nautical_tw', 'astro_tw', or 'night'
    """
    if sun_alt is None:
        return 'unknown'
    if sun_alt > 0:
        return 'day'
    elif sun_alt > -6:
        return 'civil_tw'
    elif sun_alt > -12:
        return 'nautical_tw'
    elif sun_alt > -18:
        return 'astro_tw'
    else:
        return 'night'


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
    ephemeris : dict
    """
    solar_system_ephemeris.set('builtin')

    location = get_observer_location(observer_config)
    prop_times = prop_result['times']
    prop_states = prop_result['states']  # Moon-centered ICRF, km

    # Determine output times at step_sec intervals
    prop_epoch = prop_times[0]
    prop_end = prop_times[-1]
    prop_total_sec = (prop_end - prop_epoch).sec

    # Ephemeris output window
    ephem_start_sec = max(0.0, t_start_offset)
    if t_duration is not None:
        ephem_end_sec = min(ephem_start_sec + t_duration, prop_total_sec)
    else:
        ephem_end_sec = prop_total_sec
    ephem_duration = ephem_end_sec - ephem_start_sec

    n_steps = int(ephem_duration / step_sec) + 1
    dt_array = ephem_start_sec + np.arange(n_steps) * step_sec
    # Clip to not exceed propagation end
    dt_array = dt_array[dt_array <= prop_total_sec]
    out_times = prop_epoch + TimeDelta(dt_array, format='sec')

    # Interpolate spacecraft states at output times
    prop_t_sec = (prop_times - prop_epoch).sec
    if len(prop_t_sec) > 3:
        state_interp = CubicSpline(prop_t_sec, prop_states, axis=0)
    else:
        state_interp = interp1d(prop_t_sec, prop_states, axis=0,
                                fill_value='extrapolate')

    ra_list = []
    dec_list = []
    ra_hms_list = []
    dec_dms_list = []
    dist_list = []
    alt_list = []
    az_list = []
    time_str_list = []
    sky_flag_list = []
    ra_rate_list = []
    dec_rate_list = []
    total_rate_list = []
    pa_list = []
    lunar_elong_list = []

    for idx, t in enumerate(out_times):
        t_sec = dt_array[idx]
        sc_moon = state_interp(t_sec)[:3]  # Moon-centered ICRF position [km]

        # Get Moon position in GCRS (Earth-centered)
        moon_bc = get_body_barycentric_posvel('moon', t)[0]
        earth_bc = get_body_barycentric_posvel('earth', t)[0]
        moon_geo = (moon_bc.xyz - earth_bc.xyz).to(u.km).value  # km, ICRS

        # Spacecraft position in GCRS
        sc_gcrs_km = moon_geo + sc_moon

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

        # Topocentric RA/DEC from GCRS topocentric vector
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

        # Compute Alt/Az using GCRS Cartesian position directly
        try:
            sc_coord = SkyCoord(
                CartesianRepresentation(
                    sc_gcrs_km[0] * u.km,
                    sc_gcrs_km[1] * u.km,
                    sc_gcrs_km[2] * u.km
                ),
                frame=GCRS(obstime=t)
            )
            altaz_frame = AltAz(obstime=t, location=location)
            sc_altaz = sc_coord.transform_to(altaz_frame)
            alt_list.append(round(float(sc_altaz.alt.deg), 4))
            az_list.append(round(float(sc_altaz.az.deg), 4))
        except Exception:
            alt_list.append(None)
            az_list.append(None)

        # Sun altitude -> sky condition flag
        sun_alt = _sun_altitude(t, location)
        sky_flag_list.append(_sky_condition(sun_alt))

        # Lunar elongation: angular separation between target and Moon center
        topo_moon = moon_geo - obs_xyz_km
        dist_moon = np.linalg.norm(topo_moon)
        if dist > 0 and dist_moon > 0:
            cos_elong = np.dot(topo_unit, topo_moon / dist_moon)
            lunar_elong = np.degrees(np.arccos(np.clip(cos_elong, -1, 1)))
        else:
            lunar_elong = 0.0
        lunar_elong_list.append(round(lunar_elong, 4))

        ra_list.append(round(ra, 6))
        dec_list.append(round(dec, 6))
        ra_hms_list.append(ra_hms)
        dec_dms_list.append(dec_dms)
        dist_list.append(round(dist, 3))
        time_str_list.append(t.utc.isot)

    # Compute RA/DEC rates via finite differences (arcmin/min)
    ra_arr = np.array(ra_list)
    dec_arr = np.array(dec_list)
    n = len(ra_arr)
    for idx in range(n):
        if n < 2:
            ra_rate_list.append(0.0)
            dec_rate_list.append(0.0)
            total_rate_list.append(0.0)
            pa_list.append(0.0)
            continue

        if idx == 0:
            i0, i1 = 0, 1
        elif idx == n - 1:
            i0, i1 = n - 2, n - 1
        else:
            i0, i1 = idx - 1, idx + 1

        dt_min = (dt_array[i1] - dt_array[i0]) / 60.0  # minutes
        if dt_min == 0:
            ra_rate_list.append(0.0)
            dec_rate_list.append(0.0)
            total_rate_list.append(0.0)
            pa_list.append(0.0)
            continue

        # RA difference handling wrap-around at 360 deg
        dra = ra_arr[i1] - ra_arr[i0]
        if dra > 180:
            dra -= 360
        elif dra < -180:
            dra += 360
        ddec = dec_arr[i1] - dec_arr[i0]

        # Rates in arcsec/min (deg -> arcsec = *3600)
        ra_rate = dra * 3600.0 / dt_min   # arcsec/min (coordinate rate, not cos(dec) corrected)
        dec_rate = ddec * 3600.0 / dt_min  # arcsec/min

        # Total rate: great-circle rate on sky (cos(dec) correction on RA)
        cos_dec = np.cos(np.radians(dec_arr[idx]))
        dra_sky = dra * cos_dec * 3600.0 / dt_min  # arcsec/min projected on sky
        total_rate = np.sqrt(dra_sky**2 + dec_rate**2)

        # Position angle: measured N through E (standard)
        pa = np.degrees(np.arctan2(dra_sky, dec_rate)) % 360

        ra_rate_list.append(round(ra_rate, 4))
        dec_rate_list.append(round(dec_rate, 4))
        total_rate_list.append(round(total_rate, 4))
        pa_list.append(round(pa, 2))

    return {
        'times_utc': time_str_list,
        'ra_deg': ra_list,
        'dec_deg': dec_list,
        'ra_hms': ra_hms_list,
        'dec_dms': dec_dms_list,
        'ra_rate': ra_rate_list,
        'dec_rate': dec_rate_list,
        'total_rate': total_rate_list,
        'pa': pa_list,
        'lunar_elong': lunar_elong_list,
        'distance_km': dist_list,
        'altitude_deg': alt_list,
        'azimuth_deg': az_list,
        'sky_flag': sky_flag_list,
    }
