# -*- coding: utf-8 -*-
# app.py
# קובץ Flask ראשי המכיל את לוגיקת החיפוש הסמנטי, הפאזי ומיפוי קישורים עם קלאס CSS נקי.

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
url_mapping = {}  # מחזיק את המיפוי מראש הקובץ: {'שם הקישור': 'הכתובת'}

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
    """מפרק את קובץ ה-FAQ לפי מבנה שאלה/ניסוחים/תשובה האוריגינלי שלך."""
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
    """מעבד טקסט גולמי ומחליף סוגריים מרובעים בהיפר-קישור עם קלאס CSS נקי."""
    global url_mapping
    
    # 1. טיפול במעברי שורה: החלפת \n ב-<br>
    formatted_text = text.replace('\n', '<br>')
    
    # 2. החלפת [שם הקישור] בתג HTML נקי עם המחלקה faq-link
    def replace_link(match):
        link_name = match.group(1).strip()
        if link_name in url_mapping:
            url = url_mapping[link_name]
            # אם מדובר בכתובת מייל
            if "@" in url and "://" not in url:
                return f'<a href="mailto:{url}" target="_blank" class="faq-link">{link_name}</a>'
            # קישור אינטרנט רגיל
            return f'<a href="{url}" target="_blank" class="faq-link">{link_name}</a>'
        return f'<span class="highlight">{link_name}</span>'

    formatted_text = re.sub(r'\[([^\]]+)\]', replace_link, formatted_text)
    
    # 3. טיפול בכותרת המטא-דאטה שמופיעה בתחתית התשובה
    formatted_text = formatted_text.replace("--- מטא דאטה ---", "<hr><code>--- מטא דאטה ---</code>")
    return formatted_text

# ============================================================
# פונקציית טעינה עצלנית (Lazy Loading) - מונעת קריסות של השרת
# ============================================================
def ensure_data_loaded():
    global faq_items, faq_store, embeddings_
