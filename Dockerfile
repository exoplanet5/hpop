FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy scipy astropy jplephem flask astroquery

# JPL DE440s planetary/lunar ephemeris (33 MB). Cached as its own layer so
# code changes do not re-download it. Falls back to astropy 'builtin' if absent.
ADD https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp /app/de440s.bsp
ENV HPOP_EPHEMERIS=/app/de440s.bsp

COPY . .

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
