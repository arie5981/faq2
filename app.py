# -*- coding: utf-8 -*-
from flask import Flask, render_template
import os

# בדיקת ייבוא של הספריות מהקוד המקורי
from rapidfuzz import fuzz
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
