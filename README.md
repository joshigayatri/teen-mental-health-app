# 🧠 Teen Mental Health & Depression Screener

An end-to-end machine learning project that predicts depression risk in teenagers based on behavioral indicators (social media use, sleep, stress, anxiety, academic performance) — built, trained, and deployed as an interactive web app.

## 🔗 Live Demo
- **Streamlit App:** [your streamlit link here]
- **Docker Image:** [hub.docker.com/r/gayatrijoshi/teen-mental-health-app](https://hub.docker.com/r/gayatrijoshi/teen-mental-health-app)

## 📊 Project Highlights
- Handled a **severely imbalanced dataset** (97.4% vs 2.6%) using **SMOTE** and imbalance-aware metrics (precision, recall, F1, ROC-AUC) instead of accuracy
- Compared **Logistic Regression, Random Forest, and Gradient Boosting** — Logistic Regression performed best with a **0.965 ROC-AUC**
- Built an interactive **Streamlit** app for real-time risk prediction
- **Containerized with Docker** for portable deployment

## 🐳 Run with Docker

\```bash
docker pull gayatrijoshi/teen-mental-health-app:latest
docker run -p 8501:8501 gayatrijoshi/teen-mental-health-app:latest
\```

Then open `http://localhost:8501`

### Or build from source
\```bash
git clone https://github.com/joshigayatri/teen-mental-health-app.git
cd teen-mental-health-app
docker build -t teen-mental-health-app .
docker run -p 8501:8501 teen-mental-health-app
\```

## 🛠️ Tech Stack
Python · pandas · scikit-learn · imbalanced-learn · Streamlit · Docker

## ⚠️ Disclaimer
This is an academic/educational project, not a diagnostic tool. It should not be used for real mental health screening. If you or someone you know is struggling, please reach out to a qualified professional.
