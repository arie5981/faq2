/* static/style.css */

/* הגדרות בסיסיות לטיפול בכוון הטקסט ובגלילה */
body {
    font-family: 'Inter', sans-serif;
    background-color: #f7f9fc;
}

/* הגדרות גלילה לצ'אט - חשוב ל-direction: rtl */
#chat-history {
    display: flex;
    flex-direction: column-reverse; /* הופך את סדר הופעת הבועות (החדשות למטה) */
    overflow-y: auto;
    flex-grow: 1; /* תופס את כל השטח הפנוי */
    padding: 1rem;
    max-height: calc(100vh - 18rem); /* גובה מקסימלי עם מקום לכותרת ולשדה קלט */
    min-height: 200px;
}

/* עיצוב בועות הצ'אט */
.chat-item {
    margin-bottom: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    max-width: 85%;
    width: fit-content;
}

.user-bubble {
    background-color: #0070a7; /* כחול הביטוח הלאומי */
    color: white;
    align-self: flex-start; /* מיושר לימין בגלל rtl */
    margin-right: auto;
}

.assistant-text {
    background-color: #ffffff;
    border: 1px solid #e0e7ff;
    color: #1f2937;
    align-self: flex-end; /* מיושר לשמאל בגלל rtl */
    margin-left: auto;
}

/* הודעת טעינה מיוחדת */
.loading-message {
    background-color: #fffbeb;
    border-color: #fcd34d;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

/* קישורים גלובליים מתוך קובץ ה-FAQ (התוספת החדשה) */
.faq-link {
    color: #0066cc !important;         /* כחול קישורים קלאסי */
    text-decoration: underline !important; /* קו תחתי ברור */
    font-weight: bold;
    cursor: pointer;
    transition: color 0.15s ease;
}

.faq-link:hover {
    color: #004499 !important;         /* כחול כהה יותר בריחוף */
}

/* קישורים פנימיים */
.internal-link, .related-q {
    color: #d94a2b; /* אדום/כתום מודגש */
    font-weight: 600;
    cursor: pointer;
    text-decoration: underline;
}

.internal-link:hover, .related-q:hover {
    opacity: 0.8;
}

/* רספונסיביות - מעבר מטור לשורה בטאבלט ומעלה */
@media (min-width: 768px) {
    .container {
        display: grid;
        grid-template-columns: 1fr 2fr; /* FAQ: 1 חלק, Chat: 2 חלקים */
        gap: 2rem;
    }
    #chat-history {
        max-height: calc(100vh - 10rem); /* גובה מקסימלי שונה לדסקטופ */
    }
}
