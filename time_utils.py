"""
Time utilities for HPOP.

Handles parsing of various epoch formats and delta-T corrections.
Supports: ISOT, JD, MJD, GPS, Unix, calendar formats, etc.
"""

from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np


def parse_epoch(epoch_str, epoch_format=None, scale='utc'):
    """Parse epoch string into astropy Time object.

    Parameters
    ----------
    epoch_str : str or float
        Epoch value. For JD/MJD can be numeric.
    epoch_format : str or None
        Format hint: 'isot', 'iso', 'jd', 'mjd', 'gps', 'unix',
        'datetime', 'yday', 'tt', 'tdb'.
        If None, auto-detect.
    scale : str
        Time scale: 'utc', 'tt', 'tdb', 'tai', 'tcg', 'tcb', 'gps'.

    Returns
    -------
    t : astropy.time.Time
        Parsed time object.
    """
    # Handle format aliases
    fmt_map = {
        'isot': 'isot',
        'iso': 'iso',
        'jd': 'jd',
        'mjd': 'mjd',
        'gps': 'gps',
        'unix': 'unix',
        'yday': 'yday',
        'datetime': 'datetime',
    }

    # Handle scale-as-format cases
    scale_map = {
        'tt': 'tt',
        'tdb': 'tdb',
        'tai': 'tai',
        'tcg': 'tcg',
        'tcb': 'tcb',
    }

    actual_scale = scale
    actual_format = epoch_format

    if epoch_format in scale_map:
        actual_scale = scale_map[epoch_format]
        actual_format = None  # auto-detect format

    if actual_format and actual_format in fmt_map:
        actual_format = fmt_map[actual_format]

    # Try to convert numeric input
    try:
        val = float(epoch_str)
        if actual_format is None:
            # Auto-detect: JD is ~2.4M+, MJD is ~50000-70000
            if val > 2400000:
                actual_format = 'jd'
            elif 40000 < val < 100000:
                actual_format = 'mjd'
            elif val > 1e9:
                actual_format = 'unix'
            else:
                actual_format = 'jd'
        return Time(val, format=actual_format, scale=actual_scale)
    except (ValueError, TypeError):
        pass

    # String input
    if actual_format:
        return Time(epoch_str, format=actual_format, scale=actual_scale)

    # Auto-detect string format
    try:
        return Time(epoch_str, scale=actual_scale)
    except ValueError:
        pass

    # Try common formats
    for fmt in ['isot', 'iso', 'yday']:
        try:
            return Time(epoch_str, format=fmt, scale=actual_scale)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse epoch: {epoch_str} with format={epoch_format}")


def get_delta_t(t):
    """Get delta-T (TT - UT1) for a given time.

    Parameters
    ----------
    t : astropy.time.Time

    Returns
    -------
    dt : float
        Delta-T in seconds.
    """
    # TT - UT1 = (TT - TAI) + (TAI - UTC) + (UTC - UT1)
    # TT - TAI = 32.184 s (constant)
    # TAI - UTC = leap seconds
    # UTC - UT1 = -DUT1 (from IERS data)
    try:
        tt_jd = t.tt.jd
        ut1_jd = t.ut1.jd
        delta_t = (tt_jd - ut1_jd) * 86400.0  # days to seconds
        return delta_t
    except Exception:
        # Fallback: approximate delta-T
        year = t.jy  # Julian year
        # Polynomial approximation for 2020-2030 era
        dt = 69.36 + 0.3 * (year - 2020.0)
        return dt


def time_range(t_start, duration_seconds, step_seconds):
    """Generate array of Time objects.

    Parameters
    ----------
    t_start : astropy.time.Time
    duration_seconds : float
    step_seconds : float

    Returns
    -------
    times : astropy.time.Time
        Array of times.
    """
    n_steps = int(duration_seconds / step_seconds) + 1
    dt_array = np.arange(n_steps) * step_seconds
    times = t_start + TimeDelta(dt_array, format='sec')
    return times


def time_to_jd_tdb(t):
    """Convert astropy Time to JD in TDB scale.

    Parameters
    ----------
    t : astropy.time.Time

    Returns
    -------
    jd_tdb : float or ndarray
    """
    return t.tdb.jd
