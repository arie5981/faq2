FROM heroku/heroku:22-build

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM heroku/heroku:22-runtime

WORKDIR /app

COPY --from=0 /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages

COPY app.py .
COPY Procfile .

COPY fly.toml .

EXPOSE 8080

CMD ["gunicorn", "app:app"]
