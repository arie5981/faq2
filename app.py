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

    # --- שינוי 2: נירמול גמיש ורחב של מונחי עמדות לטובת זיהוי משתמשים ---
    s = s.replace("עמדת מחשב", "משתמש חדש")
    s = s.replace("עמדה למחשב", "משתמש חדש")
    s = s.replace("עמדה", "משתמש חדש")
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
    matches = re.findall(r">>\s*(.*?)\s*:\s*(.*?)\s*<<", text)
    for key, url in matches:
        mapping[key.strip()] = url.strip()
    return mapping

def format_answer_for_html(text: str) -> str:
    """מעבד טקסט גולמי לתצוגת HTML ומחליף סוגריים מרובעים בהיפר-קישורים אמיתיים בכחול עם קו תחתון."""
    global url_mapping
    
    formatted_text = text.replace('\n', '<br>')
    link_style = 'style="color: #0000ee; text-decoration: underline; cursor: pointer;"'
    
    def replace_link(match):
        link_name = match.group(1).strip()
        
        if link_name in url_mapping:
            url = url_mapping[link_name]
            if "@" in url and "://" not in url:
                return f'<a href="mailto:{url}" {link_style}>{link_name}</a>'
            
            return f'<a onclick="safeOpen(\'{url}\')" {link_style}>{link_name}</a>'
        
        return f'<a href="#" {link_style}>{link_name}</a>'
        
    formatted_text = re.sub(r'\[([^\]]+)\]', replace_link, formatted_text)
    formatted_text = formatted_text.replace("--- מטא דאטה ---", "<hr><code>--- מטא דאטה ---</code>")
    return formatted_text

# ============================================================
# פונקציית טעינה עצלנית (Lazy Loading)
# ============================================================
def ensure_data_loaded():
    global faq_items, faq_store, embeddings_ready, url_mapping
    
    if faq_items and embeddings_ready:
        return

    if not faq_items:
        try:
            if os.path.exists(FAQ_PATH):
                with open(FAQ_PATH, "r", encoding="utf-8") as f:
                    raw_faq = f.read()
                
                url_mapping = parse_url_mappings(raw_faq)
                print(f"✅ נטענו {len(url_mapping)} חוקי מיפוי קישורים מה-FAQ.")
                
                clean_faq_text = re.sub(r">>.*?<<", "", raw_faq)
                
                faq_items = parse_faq_new(clean_faq_text)
                print(f"✅ נטענו {len(faq_items)} שאלות מה-FAQ במצב עצלני.")
            else:
                print(f"❌ שגיאה: קובץ {FAQ_PATH} לא נמצא.")
                return
        except Exception as e:
            print(f"❌ שגיאה בקריאת הקובץ: {e}")
            return

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
            
            # --- שינוי 1 חלק א': פירוק כל וריאנט למסמך עצמאי למניעת מיהול סמנטי ---
            docs = []
            for i, item in enumerate(faq_items):
                docs.append(Document(page_content=item.question, metadata={"idx": i}))
                for var in item.variants:
                    if var.strip():
                        docs.append(Document(page_content=var.strip(), metadata={"idx": i}))
            
            faq_store = FAISS.from_documents(docs, embeddings)
            embeddings_ready = True
            print("✅ אינדקס FAISS ו-Embeddings נוצרו בהצלחה בזיכרון!")
        except Exception as e:
            print(f"❌ בניית ה-Embeddings נכשלה: {e}. המערכת תתבסס על חיפוש פאזי בלבד.")
            embeddings_ready = False

def search_faq(query: str) -> Dict[str, Any]:
    """מבצע את לוגיקת החיפוש המקורית, עם הגנה מפני כפילויות ונטרול עיוות אינטנטים."""
    global faq_store, faq_items, embeddings_ready
    
    ensure_data_loaded()
    
    friendly_no_answer = """
    מצטער, לא מצאתי תשובה מדויקת לשאלה זו במאגר המידע של אתר המייצגים.<br><br>
    מה ניתן לעשות?
    <ul style="list-style-type: disc; margin-right: 20px; margin-top: 5px;">
        <li>נסה לנסח את השאלה במילים אחרות או קצרות יותר.</li>
        <li>ודא שאין שגיאות כתיב במונחי החיפוש.</li>
        <li>בחר באחת מהשאלות הנפוצות המופיעות בתפריט הצד.</li>
    </ul>
    """
    
    if not faq_items:
        return {"success": True, "answer_html": "מאגר השאלות ריק או שלא נטען כראוי.", "similar_questions": []}

    from rapidfuzz import fuzz

    nq = normalize_he(query)

    verbs = {
        "add": ["הוסף", "להוסיף", "הוספה", "מוסיף", "מוסיפים", "לצרף", "צירוף", "פתיחה", "פתיחת", "רישום", "להירשם"],
        "delete": ["מחק", "מחיקה", "להסיר", "הסר", "הסרה", "ביטול", "לבטל", "סגור", "לסגור", "ביטול משתמש"],
        "update": ["עדכן", "לעדכן", "עדכון", "עריכה", "ערוך", "לתקן", "תיקון"]
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
                score -= 5
            if intent and t_intent and intent == t_intent:
                score += 2

            scored.append((score, i, t.strip(), t_intent))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:5]

    best_fuzzy_score = top[0][0] if top else 0
    result_item = None
    similar_questions = []

    # --- שינוי 1 חלק ב': לוגיקת בחירה מעודכנת המבוססת על הלהיטים המפורקים ---
    if embeddings_ready and faq_store:
        try:
            hits = faq_store.similarity_search_with_score(nq, k=12)
        except Exception as e:
            print(f"Error during similarity search: {e}")
            hits = []

        key_words = ["יפוי", "כוח", "הרשאה", "ייצוג", "מייצג", "מעסיק", "מבוטח"]
        
        seen_indices = set()
        unique_hits = []

        for doc, score in hits:
            idx = doc.metadata["idx"]
            
            question_text = faq_items[idx].question
            text_norm = normalize_he(question_text + " " + " ".join(faq_items[idx].variants))
            for kw in key_words:
                if kw in nq and kw in text_norm:
                    score -= 0.15
            
            if idx not in seen_indices:
                unique_hits.append((doc, score, idx))
                seen_indices.add(idx)

        unique_hits.sort(key=lambda x: x[1])
        best_embed_score = unique_hits[0][1] if unique_hits else 999
        
        if best_fuzzy_score >= 85:
            result_item = copy.deepcopy(faq_items[top[0][1]])
        elif unique_hits and best_embed_score <= 1.15:
            result_item = copy.deepcopy(faq_items[unique_hits[0][2]])
        elif best_fuzzy_score >= 60:
            result_item = copy.deepcopy(faq_items[top[0][1]])
        else:
            return {"success": True, "answer_html": friendly_no_answer, "similar_questions": []}

        if result_item:
            seen_questions = set()
            seen_questions.add(result_item.question.strip())
            
            for doc, score, idx in unique_hits:
                q_text = faq_items[idx].question.strip()
