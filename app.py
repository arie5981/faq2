# app.py
# קובץ Flask ראשי עם לוגיקת Embeddings וחיפוש

import os
import re
import unicodedata
import copy
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ייבוא ספריות Flask ו-Jinja2
from flask import Flask, render_template, request, jsonify

# ייבוא ספריות ה-AI והחיפוש שלך
from rapidfuzz import fuzz
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================
# הגדרות Flask וטעינת נתונים גלובליים
# ============================================

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # חשוב לתצוגה תקינה של עברית ב-JSON

# הגדרת משתנים גלובליים (יטענו פעם אחת)
FAQ_PATH = "faq.txt" # קובץ זה צריך להיות בתיקיית הבסיס
faq_items = []
faq_store = None
embeddings = None
embeddings_ready = False
openai_api_key = None

# הגדרת שאלות נפוצות להצגה בדף הבית
POPULAR_FAQ_LIST = [
    "איך מוסיפים משתמש חדש באתר מייצגים.",
    "מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.",
    "איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.",
    "רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.",
]

# ============================================
# לוגיקת עיבוד וחיפוש (מועתקת מהקוד שלך)
# ============================================

@dataclass
class FAQItem:
    question: str
    variants: List[str]
    answer: str

def normalize_he(s: str) -> str:
    """מנקה, מאחד, ומרחיב ניסוחים חלקיים לשפה טבעית."""
    if not s:
        return ""
    # ... כאן נכנס הקוד המלא של normalize_he ...
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
    """מפרק את הקובץ לפי מבנה שאלה/ניסוחים/תשובה"""
    items = []
    blocks = re.split(r"(?=שאלה\s*:)", text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # ... כאן נכנס קוד הפארסינג המלא שלך ...
        q_match = re.search(r"שאלה\s*:\s*(.+)", b)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nהוראה\s*:|\Z)", b)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", b)

        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        variants = []

        if v_match:
            raw = v_match.group(1)
            variants = [s.strip(" -\t") for s in raw.split("\n") if s.strip()]

        items.append(FAQItem(question, variants, answer))
    return items

def search_faq(query: str) -> Dict[str, Any]:
    """
    מבצע חיפוש פאזי וסמנטי. 
    מחזיר מילון עם התשובה והשאלות הקשורות.
    """
    global faq_store, faq_items, embeddings_ready, embeddings
    nq = normalize_he(query)

    # ... כל לוגיקת החיפוש הפאזי, זיהוי כוונה ו-Embeddings ...
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

    # ============================
    # Embeddings (רק אם טעינה הצליחה)
    # ============================
    if embeddings_ready and faq_store:
        hits = faq_store.similarity_search_with_score(nq, k=8)
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
        
        # תנאי סף רלוונטיות
        if best_fuzzy_score < 55 and best_embed_score > 1.2:
             return {"success": False, "answer": "לא נמצאה תשובה, נסה לנסח את השאלה מחדש.", "similar_questions": []}

        if best_fuzzy_score >= 55:
            result_item = copy.deepcopy(faq_items[top[0][1]])
        elif boosted_hits:
            result_item = copy.deepcopy(faq_items[boosted_hits[0][0].metadata["idx"]])

        if result_item:
            similar_questions = [
                faq_items[d.metadata["idx"]].question
                for d, s in boosted_hits[1:4]
                if s <= 1.3 and faq_items[d.metadata["idx"]].question.strip() != result_item.question.strip()
            ][:3]
    
    # אם ה-embeddings לא עבדו (בגלל שגיאת טעינה) נסתמך רק על פאזי
    elif best_fuzzy_score >= 55:
        result_item = copy.deepcopy(faq_items[top[0][1]])
        
    if not result_item:
        return {"success": False, "answer": "לא נמצאה תשובה, נסה לנסח את השאלה מחדש.", "similar_questions": []}

    # עיצוב התוצאה הסופית
    answer_text = result_item.answer.strip()
    
    # הוספת כותרות מטא דאטה לשם בדיקה (כדי שתוכל לראות מה זוהה)
    answer_text += f"\n\n--- מטא דאטה ---\nמקור: faq\nשאלה מזוהה: {result_item.question}"
    
    return {
        "success": True, 
        "answer": answer_text,
        "similar_questions": similar_questions
    }

# ============================================
# טעינה ראשונית של המודלים (פונקציה רצה פעם אחת!)
# ============================================

def load_initial_data():
    global faq_items, faq_store, embeddings_ready, embeddings, openai_api_key
    
    # 1. מפתח API: Render מכניס את זה כמשתנה סביבה
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("🚨 אזהרה: משתנה OPENAI_API_KEY לא הוגדר. החיפוש הסמנטי (Embeddings) לא יעבוד.")
    
    # 2. טעינת קובץ ה-FAQ
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            raw_faq = f.read()
    except FileNotFoundError:
        print(f"❌ שגיאה חמורה: קובץ FAQ לא נמצא בנתיב: {FAQ_PATH}.")
        return

    faq_items = parse_faq_new(raw_faq)
    print(f"✅ נטענו {len(faq_items)} שאלות מה-FAQ.")
    
    # 3. יצירת Embeddings ו-FAISS
    if openai_api_key and faq_items:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
            docs = [Document(page_content=" | ".join([item.question] + item.variants), metadata={"idx": i})
                    for i, item in enumerate(faq_items)]
            faq_store = FAISS.from_documents(docs, embeddings)
            embeddings_ready = True
            print("✅ אינדקס embeddings נוצר בהצלחה.")
        except Exception as e:
            print(f"❌ שגיאה: טעינת FAISS/Embeddings נכשלה: {e}. רץ רק במצב פאזי.")
            embeddings_ready = False

# הרצת הטעינה פעם אחת לפני הניתובים
with app.app_context():
    load_initial_data()

# ============================================
# ניתובים (Routes) של Flask
# ============================================

@app.route('/')
def index():
    # ניתוב לדף הבית - מציג את תבנית HTML
    
    # מעביר את רשימת השאלות הנפוצות הקבועה לתבנית Jinja2
    return render_template('index.html', popular_questions=POPULAR_FAQ_LIST)

@app.route('/search', methods=['POST'])
def search():
    # ניתוב לטיפול בבקשות חיפוש באמצעות AJAX / JavaScript
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"success": False, "answer": "שאילתה ריקה."})
    
    # הפעלת לוגיקת החיפוש
    result = search_faq(query)
    
    # החזרת התוצאה כ-JSON
    return jsonify(result)
