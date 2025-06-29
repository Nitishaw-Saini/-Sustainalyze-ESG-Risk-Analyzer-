# -Sustainalyze-ESG-Risk-Analyzer-
Sustainalyze is an intelligent ESG (Environmental, Social, Governance) Risk Analyzer designed to help investors, analysts, and sustainability officers evaluate non-financial risks in publicly listed companies. It combines machine learning, natural language processing, and interactive dashboards to deliver actionable ESG insights.
🚀 Features
📊 Predict ESG risk levels using Random Forest and XGBoost models
Extract controversy keywords and generate AI ESG summaries using DistilGPT-2
Process structured and unstructured ESG data (scores, descriptions, controversies)
Streamlit & Power BI dashboards for visualization and decision support
Real-time prediction and sentiment analysis integration

🛠 Tech Stack
Python, Pandas, Scikit-learn, XGBoost
TextBlob, spaCy, RAKE, HuggingFace Transformers
Streamlit (Frontend), Power BI (Analytics Dashboard)
SQLite / CSV (for structured data)

📦 Components
data_preprocessing.py – Data cleaning, feature engineering
model_training.py – Random Forest/XGBoost training + evaluation
nlp_module.py – Sentiment analysis, keyword extraction
ai_insights.py – GPT-based ESG insight generation
dashboard_app.py – Streamlit-powered UI for end-users

📘 Use Cases
ESG risk classification for public companies
Automated ESG summary generation
Sector-wise ESG comparison and trend analysis
Sustainable investment screening and decision-making
