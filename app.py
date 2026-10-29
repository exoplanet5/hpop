"""
HPOP Web Application - Cislunar High Precision Orbit Propagator.

Flask web app accessible on LAN. One request parses the initial state,
propagates it, and returns an observation ephemeris; the propagated state
vectors are an optional extra output.
"""

import sys
import os
import re
import time
import traceback

import numpy as np
from flask import Flask, render_template, request, jsonify
from flask.json.provider import DefaultJSONProvider

sys.path.insert(0, os.path.dirname(__file__))

from propagator import (  # noqa: E402
    propagate, keplerian_to_cartesian, cartesian_to_keplerian,
    convert_earth_to_moon, convert_moon_to_earth,
    convert_helio_ecliptic_to_earth_icrf,
    mean_to_true_anomaly,
    AU_KM, EPHEMERIS_NAME,
    HPOPConfig,
)
from gravity import GM_SUN, GM_EARTH  # noqa: E402
from ephemeris import compute_observation_ephemeris  # noqa: E402
from time_utils import parse_epoch  # noqa: E402
from astropy.time import Time, TimeDelta  # noqa: E402
from lunar_coeffs import GM_MOON  # noqa: E402


class NumpyJSONProvider(DefaultJSONProvider):
    """JSON provider that understands numpy scalars and arrays."""

    @staticmethod
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return DefaultJSONProvider.default(o)


app = Flask(__name__)
app.json = NumpyJSONProvider(app)


@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------

def _epoch_from_request(data):
    epoch_str = str(data['epoch']).strip()
    if not epoch_str:
        raise ValueError("Epoch is empty")
    return parse_epoch(epoch_str, data.get('epoch_format', None),
                       data.get('epoch_scale', 'utc'))


def _config_from_request(data):
    config = HPOPConfig()
    config.gravity_degree = int(data.get('gravity_degree', 10))
    perturbs = data.get('perturbations', {}) or {}
    config.lunar_harmonics = bool(perturbs.get('lunar_harmonics', True))
    config.earth_gravity = bool(perturbs.get('earth_gravity', True))
    config.sun_gravity = bool(perturbs.get('sun_gravity', True))
    config.srp = bool(perturbs.get('srp', False))
    config.cr = float(data.get('srp_cr', 1.5))
    config.area_mass_ratio = float(data.get('srp_area_mass', 0.02))
    return config


def _parse_state_from_request(data, epoch):
    """Parse initial state from request data, handling frame conversion.

    Returns state in Moon-centered ICRF (km, km/s).
    """
    input_type = data.get('input_type', 'state_vector')
    ref_frame = data.get('ref_frame', 'earth_icrf')

    if input_type == 'keplerian':
        kep = data['keplerian']
        if ref_frame == 'moon_icrf':
            gm = GM_MOON
        elif ref_frame == 'sun_ecliptic':
            gm = GM_SUN
        else:
            gm = GM_EARTH

        ecc = float(kep['e'])
        anomaly_val = float(kep.get('anomaly', kep.get('nu', 0.0)))
        anomaly_type = kep.get('anomaly_type', 'true')
        nu = mean_to_true_anomaly(anomaly_val, ecc) if anomaly_type == 'mean' else anomaly_val

        a_in = float(kep['a'])
        # For sun_ecliptic, semi-major axis is in AU; convert to km
        a_km = a_in * AU_KM if ref_frame == 'sun_ecliptic' else a_in

        state0 = keplerian_to_cartesian(
            a_km, ecc,
            float(kep['i']), float(kep['raan']),
            float(kep['argp']), nu,
            gm=gm
        )
    else:
        state0 = np.array([float(x) for x in data['state']], dtype=float)
        if state0.shape != (6,) or not np.all(np.isfinite(state0)):
            raise ValueError("State vector must contain six finite numbers")
        # State vector input: if sun_ecliptic, positions in AU, velocities in AU/day
        if ref_frame == 'sun_ecliptic':
            state0[:3] *= AU_KM
            state0[3:] *= AU_KM / 86400.0

    # Convert to Earth ICRF first (if needed), then to Moon-centered ICRF
    if ref_frame == 'sun_ecliptic':
        state0 = convert_helio_ecliptic_to_earth_icrf(state0, epoch)
        state0 = convert_earth_to_moon(state0, epoch)
    elif ref_frame != 'moon_icrf':
        state0 = convert_earth_to_moon(state0, epoch)

    return state0


def _observer_from_request(data):
    return data.get('observer') or {'type': 'geodetic', 'lat': 0.0, 'lon': 0.0, 'alt': 0.0}


def _target_name(data):
    name = str(data.get('target_name', '') or '').strip()
    if not name:
        name = 'obj_' + Time.now().utc.isot[:19]
    return name


def _finite_elements(kep):
    """Replace non-finite element values with None so JSON stays valid."""
    return {k: (float(v) if np.isfinite(v) else None) for k, v in kep.items()}


# ---------------------------------------------------------------------------
# Core computation shared by all endpoints
# ---------------------------------------------------------------------------

def _run_computation(data, want_states, want_ephemeris):
    """Propagate once and build the requested outputs.

    Window logic
    ------------
    The ephemeris window is [ephem_start, ephem_start + ephem_duration_days].
    Propagation always starts at the state epoch and covers at least the end
    of the ephemeris window (and `duration_days` if given).  Starting an
    ephemeris before the state epoch is rejected: backward propagation is not
    performed.
    """
    epoch = _epoch_from_request(data)
    state0 = _parse_state_from_request(data, epoch)
    config = _config_from_request(data)
    name = _target_name(data)

    step = float(data.get('step', 60))
    if step <= 0:
        raise ValueError("Output step must be positive")
    ephem_step = float(data.get('ephemeris_step', 60))

    prop_days = float(data.get('duration_days', data.get('duration', 0)) or 0)
    ephem_days = float(data.get('ephem_duration_days', prop_days or 1))
    ephem_start_str = str(data.get('ephem_start', '') or '').strip()
    if not ephem_start_str or ephem_start_str == str(data['epoch']).strip():
        ephem_start = epoch          # default: the state epoch, whatever its format/scale
    else:
        ephem_start = parse_epoch(ephem_start_str, None, 'utc')
    start_offset = float((ephem_start - epoch).sec)

    needed_sec = 0.0
    if want_ephemeris:
        if start_offset < -1e-6:
            raise ValueError(
                f"Ephemeris start {ephem_start.utc.isot} is before the state epoch "
                f"{epoch.utc.isot}; backward propagation is not supported")
        if ephem_days <= 0:
            raise ValueError("Ephemeris duration must be positive")
        needed_sec = start_offset + ephem_days * 86400.0
    prop_duration = max(prop_days * 86400.0, needed_sec)
    if prop_duration <= 0:
        raise ValueError("Propagation duration must be positive")

    t0 = time.perf_counter()
    result = propagate(state0, epoch, prop_duration, step, config)
    t_prop = time.perf_counter() - t0

    out = {
        'success': True,
        'target_name': name,
        'epoch': epoch.utc.isot,
        'prop_start': result['times'][0].utc.isot,
        'prop_end': result['times'][-1].utc.isot,
        'ephemeris_source': EPHEMERIS_NAME,
        'timing': {'propagate_s': round(t_prop, 3), 'n_rhs_evals': result['n_rhs_evals']},
    }

    if want_ephemeris:
        observer = _observer_from_request(data)
        t0 = time.perf_counter()
        eph = compute_observation_ephemeris(
            result, observer, ephem_step,
            t_start_offset=max(0.0, start_offset),
            t_duration=ephem_days * 86400.0,
        )
        out['timing']['ephemeris_s'] = round(time.perf_counter() - t0, 3)
        out['ephemeris'] = {**eph, 'n_points': len(eph['times_utc']), 'observer': observer}

    if want_states:
        output_frame = data.get('output_frame', 'earth_icrf')
        times = result['times']
        states_moon = result['states']
        if output_frame == 'earth_icrf':
            states_out = convert_moon_to_earth(states_moon, times)
            gm_out = GM_EARTH
        else:
            output_frame = 'moon_icrf'
            states_out = states_moon
            gm_out = GM_MOON

        keplerian = []
        for s in states_out:
            try:
                keplerian.append(_finite_elements(cartesian_to_keplerian(s, gm_out)))
            except Exception:
                keplerian.append({k: None for k in ('a', 'e', 'i', 'raan', 'argp', 'nu')})

        out['propagation'] = {
            'times': [t for t in np.atleast_1d(times.utc.isot)],
            'states': states_out.tolist(),
            'keplerian': keplerian,
            'n_steps': len(times),
            'initial_state': states_out[0].tolist(),
            'initial_keplerian': keplerian[0],
            'output_frame': output_frame,
        }

    return out


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

_NUM = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'


@app.route('/api/parse_text', methods=['POST'])
def api_parse_text():
    """Parse plain text into state vector or Keplerian elements.

    Supports:
    1. XML-tagged state vectors: <EPOCH>..</EPOCH> <X>..</X> <Y>..</Y> ... <X_DOT>..</X_DOT>
       (pos_unit 'm' or 'km', vel_unit 'm/s' or 'km/s'; epoch is optional, taken as UTC)
    2. Find_Orb style Keplerian elements block
    """
    try:
        data = request.get_json()
        text = data.get('text', '') or ''

        # Detect Find_Orb format by looking for characteristic patterns
        if re.search(r'(?:Find_Orb|Epoch\s+\d{4}\s+\w+)', text) and \
           re.search(r'^\s*[Mae]\s', text, re.MULTILINE):
            return _parse_findorb(text)

        pos_unit = data.get('pos_unit', 'm')
        vel_unit = data.get('vel_unit', 'm/s')

        tag_map = {
            'X': 0, 'Y': 1, 'Z': 2,
            'X_DOT': 3, 'Y_DOT': 4, 'Z_DOT': 5,
            'VX': 3, 'VY': 4, 'VZ': 5,
        }

        values = [0.0] * 6
        found = set()
        for tag, idx in tag_map.items():
            match = re.search(rf'<{tag}>\s*({_NUM})\s*</{tag}>', text, re.IGNORECASE)
            if match:
                values[idx] = float(match.group(1))
                found.add(idx)
        if len(found) < 6:
            return jsonify({'success': False,
                            'error': 'Expected <X> <Y> <Z> <X_DOT> <Y_DOT> <Z_DOT> tags '
                                     '(or VX/VY/VZ); found only '
                                     f'{len(found)} of 6'}), 400

        # Unit conversion to km, km/s
        pos_scale = 1e-3 if pos_unit == 'm' else 1.0
        vel_scale = 1e-3 if vel_unit == 'm/s' else 1.0
        for i in range(3):
            values[i] *= pos_scale
        for i in range(3, 6):
            values[i] *= vel_scale

        out = {'success': True, 'format': 'xml', 'state': values}

        # Optional epoch (CCSDS-style ISO string, taken as UTC)
        epoch_m = re.search(r'<EPOCH>\s*([^<]+?)\s*</EPOCH>', text, re.IGNORECASE)
        if epoch_m:
            raw = epoch_m.group(1).strip().rstrip('Zz')
            try:
                out['epoch'] = Time(raw, scale='utc').utc.isot
            except Exception:
                out['epoch_unparsed'] = raw

        return jsonify(out)

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

    "(J2000 equator)" -> Earth ICRF with a in km;
    "(J2000 ecliptic)" -> heliocentric J2000 ecliptic with a in AU.
    """
    # --- Epoch ---
    epoch_str = None
    jdt_m = re.search(r'JDT\s+(' + _NUM + r')', text)
    if jdt_m:
        epoch_str = Time(float(jdt_m.group(1)), format='jd', scale='tt').utc.isot

    if not epoch_str:
        epoch_m = re.search(r'Epoch\s+(\d{4})\s+(\w+)\s+(\d+\.?\d*)\s+TT', text)
        if epoch_m:
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
            t = Time(iso, scale='tt') + TimeDelta(day_rem * 86400, format='sec')
            epoch_str = t.utc.isot

    if not epoch_str:
        return jsonify({'success': False,
                        'error': 'Cannot parse epoch from Find_Orb text'}), 400

    # --- Orbital elements ---
    M_m = re.search(r'^\s*M\s+(' + _NUM + r')', text, re.MULTILINE)
    a_m = re.search(r'^\s*a\s*(' + _NUM + r')', text, re.MULTILINE)
    e_m = re.search(r'^\s*e\s+(' + _NUM + r')', text, re.MULTILINE)
    peri_m = re.search(r'Peri\.\s+(' + _NUM + r')', text)
    node_m = re.search(r'Node\s+(' + _NUM + r')', text)
    incl_m = re.search(r'Incl\.\s+(' + _NUM + r')', text)

    missing = [lab for lab, m in (('M', M_m), ('a', a_m), ('e', e_m),
                                  ('Peri', peri_m), ('Node', node_m), ('Incl', incl_m)) if not m]
    if missing:
        return jsonify({'success': False,
                        'error': f'Cannot parse: {", ".join(missing)}'}), 400

    if re.search(r'\(\s*J2000\s+ecliptic\s*\)', text, re.IGNORECASE):
        frame = 'sun_ecliptic'
    else:
        frame = 'earth_icrf'

    return jsonify({
        'success': True,
        'format': 'findorb',
        'epoch': epoch_str,
        'frame': frame,
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


# ---------------------------------------------------------------------------
# Computation endpoints
# ---------------------------------------------------------------------------

@app.route('/api/compute', methods=['POST'])
def api_compute():
    """Propagate once; return the observation ephemeris and, optionally, states.

    JSON body (all frames/units as in the web form):
    - target_name: optional label (default 'obj_<UTC now>')
    - input_type: 'state_vector' or 'keplerian'; state or keplerian payload
    - ref_frame: 'earth_icrf' | 'moon_icrf' | 'sun_ecliptic'
    - epoch, epoch_format, epoch_scale
    - observer: {type:'geodetic', lat, lon, alt} or {type:'mpc', mpc_code}
    - ephem_start (UTC ISOT, blank = epoch), ephem_duration_days, ephemeris_step (s)
    - gravity_degree, perturbations{...}, srp_cr, srp_area_mass
    - output_states: bool (default false); step (s) and output_frame for the states
    """
    try:
        data = request.get_json()
        want_states = bool(data.get('output_states', False))
        return jsonify(_run_computation(data, want_states=want_states, want_ephemeris=True))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/propagate', methods=['POST'])
def api_propagate():
    """Run orbit propagation only (states + osculating elements)."""
    try:
        data = request.get_json()
        if not data.get('duration_days') and not data.get('duration'):
            data['duration_days'] = 1
        out = _run_computation(data, want_states=True, want_ephemeris=False)
        prop = out.pop('propagation')
        return jsonify({**out, **prop})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ephemeris', methods=['POST'])
def api_ephemeris():
    """Compute observation ephemeris only."""
    try:
        data = request.get_json()
        out = _run_computation(data, want_states=False, want_ephemeris=True)
        eph = out.pop('ephemeris')
        return jsonify({**out, **eph})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/convert', methods=['POST'])
def api_convert():
    """Convert between state vector and Keplerian elements (Moon GM)."""
    try:
        data = request.get_json()
        direction = data.get('direction', 'kep_to_cart')

        if direction == 'kep_to_cart':
            kep = data['keplerian']
            ecc = float(kep['e'])
            anomaly_val = float(kep.get('anomaly', kep.get('nu', 0.0)))
            anomaly_type = kep.get('anomaly_type', 'true')
            nu = mean_to_true_anomaly(anomaly_val, ecc) if anomaly_type == 'mean' else anomaly_val
            state = keplerian_to_cartesian(
                float(kep['a']), ecc,
                float(kep['i']), float(kep['raan']),
                float(kep['argp']), nu,
                gm=GM_MOON
            )
            return jsonify({'success': True, 'state': state.tolist()})

        state = np.array([float(x) for x in data['state']])
        return jsonify({'success': True,
                        'keplerian': _finite_elements(cartesian_to_keplerian(state, GM_MOON))})

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
