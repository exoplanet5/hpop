FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy scipy astropy jplephem flask astroquery erfa

COPY . .

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
