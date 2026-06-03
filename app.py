# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os

# ספריות ה-AI נשארות מיובאות כדי לוודא יציבות
from rapidfuzz import fuzz
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False 

# רשימת השאלות הנפוצות להצגה בדף הבית (מהקוד המקורי שלך)
POPULAR_FAQ_LIST = [
    "איך מוסיפים משתמש חדש באתר מייצגים.",
    "מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.",
    "איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.",
    "רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.",
]

@app.route('/')
def index():
    return render_template('index.html', popular_questions=POPULAR_FAQ_LIST)

@app.route('/search', methods=['POST'])
def search():
    # קבלת השאלה מה-HTML
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"success": False, "answer_html": "שאילתה ריקה."})
    
    # זמני לבדיקה: השרת מחזיר את אותה השאלה כתוכי כדי לוודא שהצינור הוויזואלי עובד
    mock_answer = f"קיבלתי את השאלה שלך: <b>{query}</b>. השרת עובד ומגיב בהצלחה!"
    
    return jsonify({
        "success": True,
        "answer_html": mock_answer,
        "similar_questions": ["שאלה לדוגמה 1", "שאלה לדוגמה 2"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
