"""
HPOP Web Application - Cislunar High Precision Orbit Propagator.

Flask web app accessible on LAN for orbit propagation and
observation ephemeris computation.
"""

import sys
import os
import json
import traceback

import numpy as np
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))

import re

from propagator import (
    propagate, keplerian_to_cartesian, cartesian_to_keplerian,
    convert_earth_to_moon, convert_moon_to_earth,
    mean_to_true_anomaly,
    HPOPConfig,
)
from ephemeris import compute_observation_ephemeris
from time_utils import parse_epoch
from astropy.time import TimeDelta
from lunar_coeffs import GM_MOON

app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


app.json_encoder = NumpyEncoder


@app.route('/')
def index():
    return render_template('index.html')


def _parse_state_from_request(data, epoch):
    """Parse initial state from request data, handling frame conversion.

    Returns state in Moon-centered ICRF (km, km/s).
    """
    input_type = data.get('input_type', 'state_vector')
    ref_frame = data.get('ref_frame', 'earth_icrf')

    if input_type == 'keplerian':
        kep = data['keplerian']
        gm = GM_MOON if ref_frame == 'moon_icrf' else 398600.4418
        ecc = float(kep['e'])
        anomaly_val = float(kep.get('anomaly', kep.get('nu', 0.0)))
        anomaly_type = kep.get('anomaly_type', 'true')
        if anomaly_type == 'mean':
            nu = mean_to_true_anomaly(anomaly_val, ecc)
        else:
            nu = anomaly_val
        state0 = keplerian_to_cartesian(
            float(kep['a']), ecc,
            float(kep['i']), float(kep['raan']),
            float(kep['argp']), nu,
            gm=gm
        )
    else:
        state0 = np.array([float(x) for x in data['state']])

    # Convert to Moon-centered ICRF if needed
    if ref_frame != 'moon_icrf':
        state0 = convert_earth_to_moon(state0, epoch)

    return state0


@app.route('/api/parse_text', methods=['POST'])
def api_parse_text():
    """Parse plain text into state vector or Keplerian elements.

    Supports:
    1. XML-tagged state vectors: <X>...</X> <Y>...</Y> ...
    2. Find_Orb style Keplerian elements block
    """
    try:
        data = request.get_json()
        text = data.get('text', '')

        # Detect Find_Orb format by looking for characteristic patterns
        if re.search(r'(?:Find_Orb|Epoch\s+\d{4}\s+\w+)', text) and \
           re.search(r'^\s*[Mae]\s', text, re.MULTILINE):
            return _parse_findorb(text)

        # Fall back to XML tag parsing
        pos_unit = data.get('pos_unit', 'km')
        vel_unit = data.get('vel_unit', 'km/s')

        tag_map = {
            'X': 0, 'Y': 1, 'Z': 2,
            'X_DOT': 3, 'Y_DOT': 4, 'Z_DOT': 5,
            'VX': 3, 'VY': 4, 'VZ': 5,
        }

        values = [0.0] * 6
        for tag, idx in tag_map.items():
            pattern = rf'<{tag}>\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*</{tag}>'
            match = re.search(pattern, text)
            if match:
                values[idx] = float(match.group(1))

        # Unit conversion
        pos_scale = 1e-3 if pos_unit == 'm' else 1.0
        vel_scale = 1e-3 if vel_unit == 'm/s' else 1.0
        for i in range(3):
            values[i] *= pos_scale
        for i in range(3, 6):
            values[i] *= vel_scale

        return jsonify({'success': True, 'state': values})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def _parse_findorb(text):
    """Parse Find_Orb style orbital elements block.

    Example:
        Epoch 2026 Apr  9.0 TT = JDT 2461139.5                 Find_Orb
        M  91.93181287 +/- 0.040            (J2000 equator)
        n  11.34293126 +/- 0.00563          Peri.   68.99438 +/- 0.026
        a423434.889 +/- 140                 Node     4.54045 +/- 0.00026
        e   0.4862783 +/- 6.5e-5            Incl.   28.58676 +/- 0.00029
    """
    NUM = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'

    # --- Epoch ---
    # Try JDT first (most reliable)
    epoch_str = None
    jdt_m = re.search(r'JDT\s+(' + NUM + r')', text)
    if jdt_m:
        from astropy.time import Time
        jd_val = float(jdt_m.group(1))
        epoch_str = Time(jd_val, format='jd', scale='tt').utc.isot

    if not epoch_str:
        # Try "Epoch YYYY Mon DD.D TT"
        epoch_m = re.search(
            r'Epoch\s+(\d{4})\s+(\w+)\s+(\d+\.?\d*)\s+TT', text)
        if epoch_m:
            from astropy.time import Time
            year = int(epoch_m.group(1))
            mon_str = epoch_m.group(2)
            day_frac = float(epoch_m.group(3))
            months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                      'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                      'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            month = months.get(mon_str[:3], 1)
            day_int = int(day_frac)
            day_rem = day_frac - day_int
            iso = f"{year:04d}-{month:02d}-{day_int:02d}T00:00:00"
            from astropy.time import TimeDelta
            t = Time(iso, scale='tt') + TimeDelta(day_rem * 86400, format='sec')
            epoch_str = t.utc.isot

    if not epoch_str:
        return jsonify({'success': False,
                        'error': 'Cannot parse epoch from Find_Orb text'}), 400

    # --- Orbital elements ---
    # M (mean anomaly) - line starts with M followed by spaces and number
    M_m = re.search(r'^\s*M\s+(' + NUM + r')', text, re.MULTILINE)
    # a (semi-major axis) - "a" immediately followed by number or spaces then number
    a_m = re.search(r'^\s*a\s*(' + NUM + r')', text, re.MULTILINE)
    # e (eccentricity)
    e_m = re.search(r'^\s*e\s+(' + NUM + r')', text, re.MULTILINE)
    # Peri. (argument of periapsis)
    peri_m = re.search(r'Peri\.\s+(' + NUM + r')', text)
    # Node (RAAN)
    node_m = re.search(r'Node\s+(' + NUM + r')', text)
    # Incl. (inclination)
    incl_m = re.search(r'Incl\.\s+(' + NUM + r')', text)

    missing = []
    if not M_m: missing.append('M')
    if not a_m: missing.append('a')
    if not e_m: missing.append('e')
    if not peri_m: missing.append('Peri')
    if not node_m: missing.append('Node')
    if not incl_m: missing.append('Incl')
    if missing:
        return jsonify({'success': False,
                        'error': f'Cannot parse: {", ".join(missing)}'}), 400

    return jsonify({
        'success': True,
        'format': 'findorb',
        'epoch': epoch_str,
        'keplerian': {
            'a': float(a_m.group(1)),
            'e': float(e_m.group(1)),
            'i': float(incl_m.group(1)),
            'raan': float(node_m.group(1)),
            'argp': float(peri_m.group(1)),
            'anomaly': float(M_m.group(1)),
            'anomaly_type': 'mean',
        },
    })


@app.route('/api/propagate', methods=['POST'])
def api_propagate():
    """Run orbit propagation.

    Expects JSON body with:
    - input_type: 'state_vector' or 'keplerian'
    - state: [x, y, z, vx, vy, vz] (km, km/s) if state_vector
    - keplerian: {a, e, i, raan, argp, nu} (km, deg) if keplerian
    - ref_frame: 'earth_icrf', 'moon_icrf' (default 'earth_icrf')
    - epoch: epoch string
    - epoch_format: 'isot', 'jd', 'mjd', etc.
    - epoch_scale: 'utc', 'tt', 'tdb', etc. (default 'utc')
    - duration_days: propagation duration in days
    - step: output step in seconds (default 60)
    - gravity_degree: max spherical harmonic degree (default 10)
    - perturbations: {lunar_harmonics, earth_gravity, sun_gravity, srp}
    - srp_cr: radiation pressure coefficient (default 1.5)
    - srp_area_mass: area-to-mass ratio m^2/kg (default 0.02)
    """
    try:
        data = request.get_json()

        # Parse epoch
        epoch_str = str(data['epoch'])
        epoch_format = data.get('epoch_format', None)
        epoch_scale = data.get('epoch_scale', 'utc')
        epoch = parse_epoch(epoch_str, epoch_format, epoch_scale)

        # Parse initial state (returns Moon-centered ICRF)
        state0 = _parse_state_from_request(data, epoch)

        # Configuration
        config = HPOPConfig()
        config.gravity_degree = int(data.get('gravity_degree', 10))
        perturbs = data.get('perturbations', {})
        config.lunar_harmonics = perturbs.get('lunar_harmonics', True)
        config.earth_gravity = perturbs.get('earth_gravity', True)
        config.sun_gravity = perturbs.get('sun_gravity', True)
        config.srp = perturbs.get('srp', False)
        config.cr = float(data.get('srp_cr', 1.5))
        config.area_mass_ratio = float(data.get('srp_area_mass', 0.02))

        duration_days = float(data.get('duration_days', data.get('duration', 1)))
        duration = duration_days * 86400.0  # convert days to seconds
        step = float(data.get('step', 60))

        # Run propagation (always in Moon-centered ICRF internally)
        result = propagate(state0, epoch, duration, step, config)

        # Output frame conversion
        output_frame = data.get('output_frame', 'moon_icrf')
        times = result['times']
        states_moon = result['states']  # Moon-centered ICRF

        if output_frame == 'earth_icrf':
            # Convert each state to Earth-centered ICRF
            states_out = np.zeros_like(states_moon)
            for i, (t, s) in enumerate(zip(times, states_moon)):
                states_out[i] = convert_moon_to_earth(s, t)
            gm_out = 398600.4418  # GM_EARTH for Keplerian elements
        else:
            states_out = states_moon
            gm_out = GM_MOON

        # Recompute Keplerian elements in output frame
        keplerian = []
        for s in states_out:
            try:
                keplerian.append(cartesian_to_keplerian(s, gm_out))
            except Exception:
                keplerian.append({
                    'a': float('nan'), 'e': float('nan'), 'i': float('nan'),
                    'raan': float('nan'), 'argp': float('nan'), 'nu': float('nan')
                })

        times_iso = [t.utc.isot for t in times]
        initial_state_out = states_out[0] if len(states_out) > 0 else state0

        return jsonify({
            'success': True,
            'times': times_iso,
            'states': states_out.tolist(),
            'keplerian': keplerian,
            'epoch': epoch.utc.isot,
            'n_steps': len(times_iso),
            'initial_state': initial_state_out.tolist(),
            'initial_keplerian': cartesian_to_keplerian(initial_state_out, gm_out),
            'output_frame': output_frame,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ephemeris', methods=['POST'])
def api_ephemeris():
    """Compute observation ephemeris.

    Same input as /api/propagate, plus:
    - observer: {type: 'geodetic', lat, lon, alt} or {type: 'mpc', mpc_code: 'XXX'}
    - ephemeris_step: step in seconds for ephemeris output (default 60)
    """
    try:
        data = request.get_json()

        # Parse epoch
        epoch_str = str(data['epoch'])
        epoch_format = data.get('epoch_format', None)
        epoch_scale = data.get('epoch_scale', 'utc')
        epoch = parse_epoch(epoch_str, epoch_format, epoch_scale)

        # Parse initial state (returns Moon-centered ICRF)
        state0 = _parse_state_from_request(data, epoch)

        # Configuration
        config = HPOPConfig()
        config.gravity_degree = int(data.get('gravity_degree', 10))
        perturbs = data.get('perturbations', {})
        config.lunar_harmonics = perturbs.get('lunar_harmonics', True)
        config.earth_gravity = perturbs.get('earth_gravity', True)
        config.sun_gravity = perturbs.get('sun_gravity', True)
        config.srp = perturbs.get('srp', False)

        # Ephemeris time range: ephem_start + ephem_duration_days
        # Orbit propagation covers: epoch to epoch + duration_days
        ephem_start_str = data.get('ephem_start', None)
        ephem_duration_days = float(data.get('ephem_duration_days',
                                             data.get('duration_days',
                                                      data.get('duration', 1))))
        prop_duration_days = float(data.get('duration_days', data.get('duration', 1)))

        if ephem_start_str:
            ephem_start = parse_epoch(ephem_start_str, 'isot', 'utc')
        else:
            ephem_start = epoch

        # Propagation must cover from epoch to ephem_start + ephem_duration
        ephem_end = ephem_start + TimeDelta(ephem_duration_days * 86400.0, format='sec')
        needed_sec = (ephem_end - epoch).sec
        prop_duration = max(prop_duration_days * 86400.0, needed_sec)
        step = float(data.get('step', 60))

        # Propagate for the full needed range
        result = propagate(state0, epoch, prop_duration, step, config)

        # Observer
        observer_config = data.get('observer', {
            'type': 'geodetic', 'lat': 0.0, 'lon': 0.0, 'alt': 0.0
        })
        ephem_step = float(data.get('ephemeris_step', 60))

        # Compute ephemeris with optional start/end clipping
        ephem_start_offset = (ephem_start - epoch).sec
        ephem = compute_observation_ephemeris(
            result, observer_config, ephem_step,
            t_start_offset=ephem_start_offset,
            t_duration=ephem_duration_days * 86400.0
        )

        return jsonify({
            'success': True,
            **ephem,
            'n_points': len(ephem['times_utc']),
            'observer': observer_config,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/convert', methods=['POST'])
def api_convert():
    """Convert between state vector and Keplerian elements."""
    try:
        data = request.get_json()
        direction = data.get('direction', 'kep_to_cart')

        if direction == 'kep_to_cart':
            kep = data['keplerian']
            ecc = float(kep['e'])
            anomaly_val = float(kep.get('anomaly', kep.get('nu', 0.0)))
            anomaly_type = kep.get('anomaly_type', 'true')
            if anomaly_type == 'mean':
                nu = mean_to_true_anomaly(anomaly_val, ecc)
            else:
                nu = anomaly_val
            state = keplerian_to_cartesian(
                float(kep['a']), ecc,
                float(kep['i']), float(kep['raan']),
                float(kep['argp']), nu,
                gm=GM_MOON
            )
            return jsonify({
                'success': True,
                'state': state.tolist(),
            })
        else:
            state = np.array([float(x) for x in data['state']])
            kep = cartesian_to_keplerian(state, GM_MOON)
            return jsonify({
                'success': True,
                'keplerian': kep,
            })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HPOP Cislunar Orbit Propagator')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0 for LAN)')
    parser.add_argument('--port', type=int, default=5100, help='Port (default: 5100)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    print(f"Starting HPOP server on http://{args.host}:{args.port}")
    print(f"Access from LAN: http://<your-ip>:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
