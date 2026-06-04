# -*- coding: utf-8 -*-
# app.py
# קובץ Flask ראשי המכיל את לוגיקת החיפוש הסמנטי, הפאזי ומיפוי קישורים דינמי.

import os
import re
import unicodedata
import copy
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ייבוא ספריות Flask
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False 

# הגדרת משתנים גלובליים
FAQ_PATH = "faq.txt"
faq_items = []
faq_store = None
embeddings_ready = False
url_mapping = {}  # מילון גלובלי שיחזיק את המיפוי: {'שם הקישור': 'הכתובת'}

POPULAR_FAQ_LIST = [
    "איך מוסיפים משתמש חדש באתר מייצגים.",
    "מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.",
    "איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.",
    "רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.",
]

@dataclass
class FAQItem:
    question: str
    variants: List[str]
    answer: str

def normalize_he(s: str) -> str:
    """מנרמל טקסט עברי לחיפוש (מוסר ניקוד, סמלים, מאחד רווחים וכו')."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()

    patterns = [
        (r"^רוצה ", "איך "),
        (r"^אני רוצה ", "איך "),
        (r"^אפשר ", "איך "),
        (r"^בא לי ", "איך "),
        (r"^מבקש ", "איך "),
    ]
    for p, repl in patterns:
        if re.match(p, s):
            s = re.sub(p, repl, s)
            break

    s = s.replace("עמדה", "עמדה למחשב")
    s = s.replace("להוסיף עמדה", "להוסיף משתמש חדש")
    return s

def parse_faq_new(text: str) -> List[FAQItem]:
    """מפרק את קובץ ה-FAQ לפי מבנה שאלה/ניסוחים/תשובה."""
    items = []
    blocks = re.split(r"(?=שאלה\s*:)", text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        
        q_match = re.search(r"שאלה\s*:\s*(.+)", b)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nניסוחים דומים\s*:|\Z)", b)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", b)
        
        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match and not v_match else ""
        
        if not a_match and not v_match:
             a_match = re.search(r"(?s)תשובה\s*:\s*(.+)", b)
             answer = a_match.group(1).strip() if a_match else ""

        if v_match:
            a_match = re.search(r"(?s)תשובה\s*:\s*(.+)", b)
            answer = a_match.group(1).strip() if a_match else ""
        
        variants = []
        if v_match:
            raw = v_match.group(1)
            variants = [s.strip(" -\t") for s in raw.split("\n") if s.strip()]

        items.append(FAQItem(question, variants, answer))
    return items

def parse_url_mappings(text: str) -> Dict[str, str]:
    """סורק את ראש הקובץ ומחלץ את הגדרות הקישורים שבין >> ל-<<"""
    mapping = {}
    # מוצא את כל המופעים שנראים כך: >>טקסט: קישור<<
    matches = re.findall(r">>\s*(.*?)\s*:\s*(.*?)\s*<<", text)
    for key, url in matches:
        mapping[key.strip()] = url.strip()
    return mapping

def format_answer_for_html(text: str) -> str:
    """מעבד טקסט גולמי לתצוגת HTML ומחליף סוגריים מרובעים בהיפר-קישורים אמיתיים בכחול עם קו תחתון."""
    global url_mapping
    
    # 1. טיפול במעברי שורה: החלפת \n ב-<br>
    formatted_text = text.replace('\n', '<br>')
    
    # הגדרת סטייל קבוע לקישורים
    link_style = 'style="color: #0000ee; text-decoration: underline; cursor: pointer;"'
    
    # 2. החלפת סוגריים מרובעים [שם הקישור]
    def replace_link(match):
        link_name = match.group(1).strip()
        
        if link_name in url_mapping:
            url = url_mapping[link_name]
            if "@" in url and "://" not in url:
                return f'<a href="mailto:{url}" {link_style}>{link_name}</a>'
            
            # --- השינוי פה: קריאה לפונקציה מיוחדת במקום href ישיר ---
            return f'<a onclick="safeOpen(\'{url}\')" {link_style}>{link_name}</a>'
        
        return f'<a href="#" {link_style}>{link_name}</a>'
        
    # מחפש כל תבנית של [טקסט]
    formatted_text = re.sub(r'\[([^\]]+)\]', replace_link, formatted_text)
    
    # 3. טיפול בכותרת המטא-דאטה שמופיעה בתחתית התשובה
    formatted_text = formatted_text.replace("--- מטא דאטה ---", "<hr><code>--- מטא דאטה ---</code>")
    return formatted_text

# ============================================================
# פונקציית טעינה עצלנית (Lazy Loading)
# ============================================================
def ensure_data_loaded():
    global faq_items, faq_store, embeddings_ready, url_mapping
    
    if faq_items and embeddings_ready:
        return

    # 1. טעינת קובץ הטקסט של ה-FAQ
    if not faq_items:
        try:
            if os.path.exists(FAQ_PATH):
                with open(FAQ_PATH, "r", encoding="utf-8") as f:
                    raw_faq = f.read()
                
                # חילוץ מילון הקישורים הגלובלי מראש הקובץ
                url_mapping = parse_url_mappings(raw_faq)
                print(f"✅ נטענו {len(url_mapping)} חוקי מיפוי קישורים מה-FAQ.")
                
                # ניקוי הגדרות הקישורים מגוף הטקסט כדי שלא יפריעו לפארסר השאלות
                clean_faq_text = re.sub(r">>.*?<<", "", raw_faq)
                
                faq_items = parse_faq_new(clean_faq_text)
                print(f"✅ נטענו {len(faq_items)} שאלות מה-FAQ במצב עצלני.")
            else:
                print(f"❌ שגיאה: קובץ {FAQ_PATH} לא נמצא.")
                return
        except Exception as e:
            print(f"❌ שגיאה בקריאת הקובץ: {e}")
            return

    # 2. בניית ה-Embeddings וה-FAISS
    if faq_items and not embeddings_ready:
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                print("🚨 אזהרה: OPENAI_API_KEY לא מוגדר במערכת.")
                return

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
            docs = [Document(page_content=" | ".join([item.question] + item.variants), metadata={"idx": i})
                    for i, item in enumerate(faq_items)]
            
            faq_store = FAISS.from_documents(docs, embeddings)
            embeddings_ready = True
            print("✅ אינדקס FAISS ו-Embeddings נוצרו בהצלחה בזיכרון!")
        except Exception as e:
            print(f"❌ בניית ה-Embeddings נכשלה: {e}. המערכת תתבסס על חיפוש פאזי בלבד.")
            embeddings_ready = False

def search_faq(query: str) -> Dict[str, Any]:
    """מבצע את לוגיקת החיפוש המקורית והחכמה שלך."""
    global faq_store, faq_items, embeddings_ready
    
    ensure_data_loaded()
    
    if not faq_items:
        return {"success": False, "answer": "מאגר השאלות ריק או שלא נטען כראוי.", "similar_questions": []}

    from rapidfuzz import fuzz

    nq = normalize_he(query)

    verbs = {
        "add": ["הוסף", "להוסיף", "הוספה", "מוסיף", "מוסיפים", "לצרף", "צירוף", "פתיחה", "פתיחת", "רישום", "להירשם"],
        "delete": ["מחק", "מחיקה", "להסיר", "הסר", "הסרה", "ביטול", "לבטל", "סגור", "לסגור", "ביטול משתמש"],
        "update": ["עדכן", "לעדכן", "עדכון", "שינוי", "לשנות", "עריכה", "ערוך", "לתקן", "תיקון"]
    }
    intent = None
    for k, words in verbs.items():
        if any(w in nq for w in words):
            intent = k
            break
    
    scored = []
    for i, item in enumerate(faq_items):
        all_texts = [item.question] + item.variants
        for t in all_texts:
            score = fuzz.token_sort_ratio(nq, normalize_he(t))

            t_intent = None
            for k, words in verbs.items():
                if any(w in t for w in words):
                    t_intent = k
                    break

            if intent and t_intent and intent != t_intent:
                score -= 50
            if intent and t_intent and intent == t_intent:
                score += 25

            scored.append((score, i, t.strip(), t_intent))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:5]

    best_fuzzy_score = top[0][0] if top else 0
    result_item = None
    similar_questions = []

    if embeddings_ready and faq_store:
        try:
            hits = faq_store.similarity_search_with_score(nq, k=8)
        except Exception as e:
            print(f"Error during similarity search: {e}")
            hits = []

        key_words = ["יפוי", "כוח", "הרשאה", "ייצוג", "מייצג", "מעסיק", "מבוטח"]
        boosted_hits = []

        for doc, score in hits:
            idx = doc.metadata["idx"]
            question_text = faq_items[idx].question
            text_norm = normalize_he(question_text + " " + " ".join(faq_items[idx].variants))

            for kw in key_words:
                if kw in nq and kw in text_norm:
                    score -= 0.15 
            boosted_hits.append((doc, score))

        boosted_hits.sort(key=lambda x: x[1])
        best_embed_score = boosted_hits[0][1] if boosted_hits else 999
        
        if best_fuzzy_score < 55 and best_embed_score > 1.2:
             return {"success": False, "answer": "לא נמצאה תשובה, נסה לנסח את השאלה מחדש.", "similar_questions": []}

        if best_fuzzy_score >= 55:
            result_item = copy.deepcopy(faq_items[top[0][1]])
        elif boosted_hits and best_embed_score <= 1.2:
            result_item = copy.deepcopy(faq_items[boosted_hits[0][0].metadata["idx"]])
            
        if result_item:
            similar_questions = [
                faq_items[d.metadata["idx"]].question
                for d, s in boosted_hits[1:4]
                if s <= 1.3 and faq_items[d.metadata["idx"]].question.strip() != result_item.question.strip()
            ][:3]
    
    elif best_fuzzy_score >= 55:
        result_item = copy.deepcopy(faq_items[top[0][1]])
        
    if not result_item:
        return {"success": False, "answer": "לא נמצאה תשובה, נסה לנסח את השאלה מחדש.", "similar_questions": []}

    answer_text = result_item.answer.strip()
    answer_text += f"\n\n--- מטא דאטה ---\nמקור: faq\nשאלה מזוהה: {result_item.question}"
    
    return {
        "success": True, 
        "answer_html": format_answer_for_html(answer_text),
        "similar_questions": similar_questions
    }

# ============================================
# ניתובים (Routes) של Flask
# ============================================

@app.route('/')
def index():
    return render_template('index.html', popular_questions=POPULAR_FAQ_LIST)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json() or {}
    query = data.get('query', '')
    
    if not query:
        return jsonify({"success": False, "answer_html": "שאילתה ריקה."})
    
    result = search_faq(query)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
