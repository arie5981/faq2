# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import re
import unicodedata

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

FAQ_PATH = "faq.txt"

# פונקציות עזר לנירמול טקסט (מהקוד המקורי שלך)
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r'[^\w\s\u0590-\u05fe]', ' ', text)
    return " ".join(text.split())

def load_faq_file(path: str):
    """קריאת קובץ ה-FAQ ופירוק לשאלות ותשובות"""
    if not os.path.exists(path):
        return [], f"קובץ {path} לא נמצא בשרת!"
    
    qa_pairs = []
    current_q = None
    current_a = []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                if line_str.endswith("?"):
                    if current_q and current_a:
                        qa_pairs.append({"question": current_q, "answer": "\n".join(current_a)})
                        current_a = []
                    current_q = line_str
                else:
                    if current_q:
                        current_a.append(line_str)
            
            if current_q and current_a:
                qa_pairs.append({"question": current_q, "answer": "\n".join(current_a)})
                
        return qa_pairs, f"נטענו בהצלחה {len(qa_pairs)} שאלות ותשובות."
    except Exception as e:
        return [], f"שגיאה בקריאת הקובץ: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # טעינת הקובץ בצורה עצלנית כדי לשמור על יציבות הזיכרון
    pairs, file_status = load_faq_file(FAQ_PATH)
    
    data = request.get_json() or {}
    query = data.get('query', '')
    
    mock_answer = f"שרת הבדיקה קיבל: {query}. <br>סטטוס קובץ: {file_status}"
    return jsonify({
        "success": True,
        "answer_html": mock_answer,
        "similar_questions": []
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
