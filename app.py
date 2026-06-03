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
    
    if best_match and best_score > 85:  # רף גבוה לחיפוש מהיר
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
    
    # 1. ניסיון ראשון: חיפוש פאזי מהיר (חוסך פניות ל-AI)
    fuzzy_answer, similar = find_best_match_fuzzy(query, qa_pairs)
    if fuzzy_answer:
        return jsonify({
            "success": True,
            "answer_html": f"<b>נמצאה תשובה במאגר:</b><br>{fuzzy_answer}",
            "similar_questions": similar
        })
        
    # 2. ניסיון שני: חיפוש סמנטי מבוסס AI (טעינה עצלנית למניעת חנק זיכרון)
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return jsonify({
                "success": True,
                "answer_html": "לא נמצאה תשובה מדויקת, ומפתח ה-OPENAI_API_KEY לא מוגדר במערכת.",
                "similar_questions": similar
            })
            
        # יצירת המסמכים עבור ה-Vector Store
        documents = [Document(page_content=f"שגיאה: {p['question']}\nתשובה: {p['answer']}", 
                              metadata={"answer": p["answer"], "question": p["question"]}) for p in qa_pairs]
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
        db = FAISS.from_documents(documents, embeddings)
        
        # חיפוש סמנטי
        docs_and_scores = db.similarity_search_with_score(query, k=1)
        
        if docs_and_scores:
            doc, score = docs_and_scores[0]
            # ב-FAISS, ככל שהציון (L2 distance) נמוך יותר, ההתאמה טובה יותר
            if score < 1.2: 
                ai_answer = doc.metadata["answer"]
                return jsonify({
                    "success": True,
                    "answer_html": f"<b>תשובה סמנטית (AI):</b><br>{ai_answer}",
                    "similar_questions": similar
                })
                
        return jsonify({
            "success": True,
            "answer_html": "מצטער, לא הצלחתי למצוא תשובה מתאימה לשאלה זו במאגר המייצגים.",
            "similar_questions": similar
        })
        
    except Exception as e:
        return jsonify({
            "success": True,
            "answer_html": f"לא נמצאה תשובה פאזית, ונכשלה ריצת ה-AI עקב שגיאה פנימית: {str(e)}",
            "similar_questions": similar
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
