# ✦ AI Resume Builder — Powered by ML

A professional AI-powered Resume Builder built with Python/Flask and ML algorithms for resume scoring, skill gap analysis, and career path prediction.

---

## 🚀 Features

| Feature | ML Algorithm Used |
|---|---|
| Resume Scoring | TF-IDF + Weighted Scoring |
| Skill Gap Analysis | Jaccard Similarity + Cosine Weighting |
| Career Prediction | Rule-based Random Forest Logic |
| AI Chat Assistant | NLP Keyword Matching |
| PDF Export | ReportLab |

---

## 🛠 Project Structure

```
ai-resume-builder/
├── app.py                    # Flask main application
├── models/
│   ├── resume_scorer.py      # ML resume scoring (TF-IDF style)
│   ├── skill_matcher.py      # Skill gap analysis (cosine similarity)
│   └── career_predictor.py   # Career path prediction (Random Forest)
├── utils/
│   ├── ai_assistant.py       # AI chatbot logic
│   └── resume_generator.py   # PDF generation (ReportLab)
├── templates/
│   └── index.html            # Full frontend UI
├── requirements.txt
├── Procfile                  # Render/Heroku deployment
├── render.yaml               # Render config
└── .gitignore
```

---

## ⚙️ Run Locally (VS Code)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-builder.git
cd ai-resume-builder
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
Create a `.env` file (optional):
```
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
```

### 5. Run the app
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📤 Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — AI Resume Builder"
git remote add origin https://github.com/YOUR_USERNAME/ai-resume-builder.git
git branch -M main
git push -u origin main
```

---

## 🌐 Deploy on Render

1. Go to [render.com](https://render.com) and sign up / log in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure the service
5. Set environment variable:
   - `SECRET_KEY` → any random string
6. Click **"Deploy Web Service"**

Your app will be live at: `https://ai-resume-builder.onrender.com`

> **Note:** Render free tier spins down after inactivity. First load may take ~30 seconds.

---

## 🤖 ML Algorithms Explained

### 1. Resume Scorer (`resume_scorer.py`)
- Scores resumes across 5 dimensions: completeness, impact, skills, education, formatting
- Uses **power word detection** (action verbs like "Led", "Built", "Optimized")
- Penalizes weak/passive language ("responsible for", "helped with")
- Rewards **quantified achievements** (numbers, percentages)
- Final score = weighted average of 5 dimensions

### 2. Skill Matcher (`skill_matcher.py`)
- Compares user skills against a curated job skills database
- Uses **Jaccard similarity**: |A∩B| / |A∪B|
- Weights required skills more heavily than preferred
- Returns matched skills, missing skills, and learning resources

### 3. Career Predictor (`career_predictor.py`)
- Profiles 8 career paths with keyword signatures
- Computes **cosine-style similarity** between user skill set and each career
- Adds bonuses for years of experience and relevant education
- Returns top 3 career matches with salary ranges and growth outlook

---

## 🛡 Tech Stack

- **Backend:** Python 3.10+, Flask 3.x
- **ML:** Pure Python (TF-IDF, Jaccard, weighted scoring — no sklearn needed)
- **PDF:** ReportLab
- **Frontend:** Vanilla HTML/CSS/JS (dark theme, fully responsive)
- **Deployment:** Render.com (gunicorn WSGI server)

---

## 📝 License

MIT License — free to use, modify, and distribute.
