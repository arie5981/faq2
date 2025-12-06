FROM python:3.11-slim as build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git && curl --proto '=https' --tlsv1.2 -sSf https://www.google.com/search?q=https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
FROM python:3.11-slim
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY app.py .
COPY Procfile .
ENV PORT 8080
EXPOSE 8080
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
