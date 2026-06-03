# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

POPULAR_FAQ_LIST = [
    "איך מוסיפים משתמש חדש באתר מייצגים.",
    "מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.",
    "איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.",
    "רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.",
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # ה-Imports רצים רק פה פנימית, כשיש דרישה, ולא מדליקים את השרת בקריסה
    try:
        from langchain_openai import OpenAIEmbeddings
        import tiktoken
        status = "הספריות נטענו בהצלחה בתוך פונקציית החיפוש!"
    except Exception as e:
        status = f"שגיאה בטעינת הספריות: {str(e)}"

    data = request.get_json() or {}
    query = data.get('query', '')
    
    mock_answer = f"שרת הבדיקה קיבל: {query}. סטטוס ספריות: {status}"
    return jsonify({
        "success": True,
        "answer_html": mock_answer,
        "similar_questions": []
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
