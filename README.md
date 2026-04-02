# HPOP — Cislunar High Precision Orbit Propagator

A web-based high precision orbit propagator for spacecraft in the cislunar region. Built with Python and Flask, accessible over LAN.

## Dynamical Model

The propagator integrates equations of motion in **Moon-centered ICRF** frame using SciPy's DOP853 (8th-order Runge-Kutta) integrator with tolerances of `rtol=1e-12`, `atol=1e-14`.

### Perturbation Sources

| Source | Description |
|--------|-------------|
| **Lunar non-spherical gravity** | Spherical harmonic expansion up to configurable degree/order (max 20). Fully normalized Stokes coefficients from the GRAIL GRGM660PRIM model. Moon body-fixed frame orientation uses the IAU 2009 libration model. |
| **Earth third-body gravity** | Point-mass perturbation. Earth position from JPL ephemeris via Astropy, pre-interpolated with cubic splines for integration efficiency. |
| **Sun third-body gravity** | Point-mass perturbation, same ephemeris approach as Earth. |
| **Solar radiation pressure** | Optional. Cannonball model with configurable reflectivity coefficient (Cr) and area-to-mass ratio. Scales with inverse-square heliocentric distance. |

### Constants

- GM_Moon = 4902.800066 km³/s²
- GM_Earth = 398600.4418 km³/s²
- GM_Sun = 1.32712440018×10¹¹ km³/s²
- R_Moon (reference radius) = 1738.0 km

### Time Handling

- Supports epoch input in: ISOT, ISO, JD, MJD, TT, TDB, TAI, GPS, Unix
- Internal propagation uses TDB scale
- Delta-T (TT − UT1) corrections applied via Astropy/IERS data

### Reference Frames

The project uses **ICRF** consistently (not J2000; the two differ by ~0.01 arcsec frame tie rotation).

Input state vectors can be provided in:
- **Earth ICRF** (Earth-centered inertial) — converted to Moon-centered using JPL ephemeris at the input epoch
- **Moon ICRF** (Moon-centered inertial) — used directly

Output reference frame is selectable between Earth ICRF and Moon ICRF. Internal propagation always runs in Moon-centered ICRF; frame conversion is applied at the output boundary.

## Features

### Part 1: Orbit Propagation
- Input: state vector (km, km/s) or Keplerian elements (km, deg)
- Supports pasting XML-tagged text (e.g., `<X>...</X> <X_DOT>...</X_DOT>`) with automatic unit conversion (m↔km, m/s↔km/s)
- Selectable input and output reference frames (Earth ICRF / Moon ICRF)
- Propagation duration in days, output step in seconds
- Output: time-stamped state vectors and osculating Keplerian elements (with correct GM for the selected output frame)
- Downloadable as CSV

### Part 2: Observation Ephemeris
- Computes topocentric RA/DEC of the spacecraft as seen from a ground observer
- Observer specified by geodetic coordinates (lat, lon, alt) or MPC observatory code
- Configurable start time and duration (sub-range of the propagation window)
- Output at 1-minute intervals (configurable):
  - RA (HMS), DEC (DMS)
  - RA rate, DEC rate (arcsec/min)
  - Total sky-plane motion rate (arcsec/min) and position angle (PA, N through E)
  - Lunar elongation (angular separation between target and Moon center)
  - Topocentric distance, altitude, azimuth
  - Sky condition flag (night / astronomical twilight / nautical twilight / civil twilight / day)
- Downloadable as CSV

## Requirements

- Python 3.10+
- Packages: `numpy`, `scipy`, `astropy`, `jplephem`, `flask`, `astroquery`

Install into a virtual environment:

```bash
pip install numpy scipy astropy jplephem flask astroquery
```

## Running

```bash
cd /path/to/hpop
python app.py
```

The server starts on `0.0.0.0:5100` by default, accessible from any device on the LAN at `http://<your-ip>:5100`.

### Options

```
python app.py --host 0.0.0.0 --port 5100 --debug
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address (0.0.0.0 for LAN access) |
| `--port` | `5100` | Port number |
| `--debug` | off | Enable Flask debug/auto-reload |

## Project Structure

```
hpop/
├── app.py              Flask web app and API routes
├── propagator.py        HPOP core: ODE integration, frame conversion, Keplerian ↔ Cartesian
├── gravity.py           Lunar spherical harmonics, third-body perturbations, SRP
├── lunar_coeffs.py      GRAIL normalized Stokes coefficients (C_nm, S_nm)
├── ephemeris.py         Topocentric RA/DEC computation for ground observers
├── time_utils.py        Multi-format epoch parsing, delta-T
└── templates/
    └── index.html       Web interface
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web interface |
| POST | `/api/propagate` | Run orbit propagation |
| POST | `/api/ephemeris` | Compute observation ephemeris |
| POST | `/api/parse_text` | Parse XML-tagged state vector text |
| POST | `/api/convert` | Convert between state vector and Keplerian elements |
