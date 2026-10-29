---
title: HPOP
emoji: 🌙
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
---

# HPOP — Cislunar High Precision Orbit Propagator

A web-based high precision orbit propagator and observation-ephemeris generator for spacecraft in the cislunar region (and, via a heliocentric input frame, Sun–Earth L2 objects). Built with Python and Flask, accessible over LAN or as a Hugging Face Space.

## Workflow

One page, one button. Enter the initial state (paste text, state vector, or Keplerian elements), the observer, and the ephemeris window, then press **Compute Ephemeris**. The server propagates the orbit from the state epoch to the end of the ephemeris window and returns the topocentric ephemeris. Propagated state vectors and osculating elements are an optional extra output (off by default).

Propagation always happens internally: the ephemeris at time *t* needs the spacecraft state at *t*, which only numerical integration provides. What the optional checkbox controls is whether the state tables are built and sent to the browser.

## Dynamical Model

The propagator integrates equations of motion in the **Moon-centered ICRF** frame using SciPy's DOP853 (8th-order Runge-Kutta) integrator with tolerances `rtol=1e-12`, `atol=1e-14`. The integrator chooses its own step size (no artificial cap); outputs at arbitrary times come from the integrator's 7th-order dense output.

### Perturbation Sources

| Source | Description |
|--------|-------------|
| **Lunar non-spherical gravity** | Spherical harmonic expansion up to configurable degree/order (max 20). Fully normalized Stokes coefficients from the GRAIL GRGM660PRIM model. Moon body-fixed frame orientation uses the IAU 2009 libration model. Applied within 50,000 km of the Moon. |
| **Earth third-body gravity** | Point-mass perturbation with indirect term. |
| **Sun third-body gravity** | Point-mass perturbation with indirect term. |
| **Solar radiation pressure** | Optional. Cannonball model with configurable reflectivity coefficient (Cr) and area-to-mass ratio. Scales with inverse-square heliocentric distance. No eclipse model. |

### Solar-System Ephemeris

Body positions (Moon, Earth, Sun) come from a JPL DE kernel when one is available, otherwise from astropy's analytic `builtin` model. The analytic Moon differs from DE440 by up to ~18 km (~9 arcsec as seen from Earth), so a kernel is recommended for comparison with astrometry. Search order at startup:

1. `$HPOP_EPHEMERIS` — path to an SPK kernel, or the word `builtin`
2. `~/JPLDE/de440s.bsp`
3. `./de440s.bsp` in the project directory
4. astropy `builtin`

The source in use is printed at startup and reported in every result. The Docker image downloads `de440s.bsp` (33 MB) from NAIF at build time.

### Constants

- GM_Moon = 4902.800066 km³/s²
- GM_Earth = 398600.4418 km³/s²
- GM_Sun = 1.32712440018×10¹¹ km³/s²
- R_Moon (reference radius) = 1738.0 km
- 1 AU = 149597870.7 km

### Time Handling

- Epoch input in: ISOT, ISO, JD, MJD, TT, TDB, TAI, GPS, Unix
- Internal propagation uses TDB scale
- Earth orientation (UT1, polar motion) via Astropy/IERS data

### Reference Frames

The project uses **ICRF** consistently (not J2000; the two differ by ~0.01 arcsec frame tie rotation).

Input states can be given in:
- **Earth ICRF** (Earth-centered inertial, km, km/s) — converted to Moon-centered at the input epoch
- **Moon ICRF** (Moon-centered inertial) — used directly
- **Sun J2000 ecliptic** (heliocentric, AU and AU/day; Keplerian *a* in AU) — converted via astropy `HeliocentricMeanEcliptic(equinox='J2000') → GCRS`, then to Moon-centered

Propagated states can be output in Earth ICRF or Moon ICRF (osculating elements use the matching GM).

### Observation Ephemeris

Topocentric **geometric** positions: observer and spacecraft are both evaluated at the output instant. Earth rotation, precession/nutation and topocentric parallax are included; light-time, stellar aberration and refraction are not. Plate-solved astrometry (Gaia frame) is "astrometric", i.e. already de-aberrated, so the only correction missing for a direct comparison is light-time (≈ v⊥/c ≈ 20″ for an object moving ~30 km/s in the barycentric frame; ≲1″ for lunar orbiters).

## Features

### Initial State Input
- **Paste Text** (default):
  - XML-tagged state vector `<EPOCH>…</EPOCH> <X>…</X> <Y>…</Y> <Z>…</Z> <X_DOT>…</X_DOT> …`, default units m and m/s (switchable to km, km/s). The epoch, if present, fills the epoch field (UTC).
  - Find_Orb elements block (`Epoch … TT = JDT …`, `M`, `a`, `e`, `Peri.`, `Node`, `Incl.`). `(J2000 equator)` → Earth ICRF with *a* in km; `(J2000 ecliptic)` → heliocentric ecliptic with *a* in AU. Mean anomaly and the TT epoch are handled automatically.
- **State Vector** fields
- **Keplerian Elements** with True or Mean anomaly
- **Target name**; blank defaults to `obj_<UTC time when Compute is pressed>`, used in CSV file names

### Ephemeris Output (1-minute default step)
- RA (HMS), DEC (DMS), RA/DEC rates (arcsec/min), total sky-plane rate and position angle (N through E)
- Lunar elongation (angular separation from the Moon's centre as seen by the observer)
- Topocentric distance, altitude, azimuth
- Sky condition flag: night / astronomical / nautical / civil twilight / day
- Observer by geodetic coordinates or MPC observatory code
- CSV download

### Propagated States (optional)
- Time-stamped state vectors and osculating Keplerian elements in Earth ICRF or Moon ICRF at a chosen step
- CSV download

## Requirements

- Python 3.10+
- Packages: `numpy`, `scipy`, `astropy`, `jplephem`, `flask`, `astroquery`

```bash
pip install numpy scipy astropy jplephem flask astroquery
```

Optional but recommended: a JPL DE kernel such as `de440s.bsp` (see *Solar-System Ephemeris*).

## Running

```bash
cd /path/to/hpop
python app.py
```

The server starts on `0.0.0.0:5100`, accessible from any device on the LAN at `http://<your-ip>:5100`.

### Options

```
python app.py --host 0.0.0.0 --port 5100 --debug
HPOP_EPHEMERIS=/path/to/de440s.bsp python app.py
```

| Flag / variable | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address (0.0.0.0 for LAN access) |
| `--port` | `5100` | Port number |
| `--debug` | off | Enable Flask debug/auto-reload |
| `HPOP_EPHEMERIS` | auto | SPK kernel path, or `builtin` |

### Docker / Hugging Face Spaces

The `Dockerfile` installs the dependencies, downloads `de440s.bsp` and serves on port 7860. A GitHub Action (`.github/workflows/sync-to-hf.yml`) pushes `main` to the Hugging Face Space on every push.

## Project Structure

```
hpop/
├── app.py              Flask web app and API routes
├── propagator.py        HPOP core: ephemeris source selection, ODE integration, frame conversion, Keplerian ↔ Cartesian
├── gravity.py           Lunar spherical harmonics, third-body perturbations, SRP
├── lunar_coeffs.py      GRAIL normalized Stokes coefficients (C_nm, S_nm)
├── ephemeris.py         Vectorized topocentric ephemeris for ground observers
├── time_utils.py        Multi-format epoch parsing, delta-T
├── Dockerfile           Container for Hugging Face Spaces
└── templates/
    └── index.html       Single-page web interface
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web interface |
| POST | `/api/compute` | Propagate once; return ephemeris and, with `output_states: true`, the states |
| POST | `/api/propagate` | States and osculating elements only |
| POST | `/api/ephemeris` | Observation ephemeris only |
| POST | `/api/parse_text` | Parse XML-tagged state vector or Find_Orb elements |
| POST | `/api/convert` | Convert between state vector and Keplerian elements |

`/api/compute` body fields: `target_name`, `input_type` (`state_vector` | `keplerian`) with `state` or `keplerian{a,e,i,raan,argp,anomaly,anomaly_type}`, `ref_frame` (`earth_icrf` | `moon_icrf` | `sun_ecliptic`), `epoch`, `epoch_format`, `epoch_scale`, `observer` (`{type:'geodetic',lat,lon,alt}` or `{type:'mpc',mpc_code}`), `ephem_start` (UTC ISOT, blank = epoch), `ephem_duration_days`, `ephemeris_step` (s), `gravity_degree`, `perturbations{lunar_harmonics,earth_gravity,sun_gravity,srp}`, `srp_cr`, `srp_area_mass`, `output_states`, `step` (s), `output_frame`.
