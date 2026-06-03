FROM python:3.11-slim

WORKDIR /app

# התקנת הספריות הבסיסיות
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת קבצי האפליקציה
COPY app.py .
COPY templates/ ./templates/

# הגדרת משתנה סביבה לפורט
ENV PORT=8080
EXPOSE 8080

# הרצה ישירה של פייתון ללא Gunicorn
CMD ["python", "app.py"]
