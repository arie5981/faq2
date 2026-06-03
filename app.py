# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import re
import unicodedata
from rapidfuzz import fuzz

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

FAQ_PATH = "faq.txt"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r'[^\w\s\u0590-\u05fe]', ' ', text)
    return " ".join(text.split())

def load_faq_file(path: str):
    if not os.path.exists(path):
        return []
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
        return qa_pairs
    except:
        return []

# פונקציית החיפוש הפאזי המקורית שלך
def find_best_match_fuzzy(query: str, qa_pairs: list):
    cleaned_query = clean_text(query)
    best_match = None
    best_score = 0
    similar_questions = []
    
    for pair in qa_pairs:
        cleaned_q = clean_text(pair["question"])
        score = fuzz.token_set_ratio(cleaned_query, cleaned_q)
        
        if score > 50:
            similar_questions.append((pair["question"], pair["answer"], score))
            
        if score > best_score:
            best_score = score
            best_match = pair

    similar_questions.sort(key=lambda x: x[2], reverse=True)
    top_similar = [q[0] for q in similar_questions[:3]]
    
    # אם הציון גבוה מספיק, נחזיר את התשובה
    if best_match and best_score > 70:
        return best_match["answer"], top_similar
    return None, top_similar

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json() or {}
    query = data.get('query', '')
    
    if not query:
        return jsonify({"success": False, "answer_html": "שאילתה ריקה."})
        
    qa_pairs = load_faq_file(FAQ_PATH)
    fuzzy_answer, similar = find_best_match_fuzzy(query, qa_pairs)
    
    if fuzzy_answer:
        answer_html = f"<b>נמצאה תשובה בחיפוש מהיר:</b><br>{fuzzy_answer}"
    else:
        answer_html = "לא נמצאה תשובה מדויקת בחיפוש מהיר. (בשלב הבא נחבר את ה-AI)."
        
    return jsonify({
        "success": True,
        "answer_html": answer_html,
        "similar_questions": similar
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
