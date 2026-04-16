"""
AI Resume Builder - Flask Application
Powered by ML algorithms for resume scoring, skill matching & career suggestions
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from models.resume_scorer import ResumeScorer
from models.skill_matcher import SkillMatcher
from models.career_predictor import CareerPredictor
from utils.resume_generator import generate_pdf_resume
from utils.ai_assistant import AIAssistant

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")

# Initialize ML models (loaded once at startup)
scorer = ResumeScorer()
matcher = SkillMatcher()
predictor = CareerPredictor()
assistant = AIAssistant()


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/score-resume", methods=["POST"])
def score_resume():
    """ML-powered resume scoring using TF-IDF + cosine similarity"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    result = scorer.score(data)
    return jsonify(result)


@app.route("/api/match-skills", methods=["POST"])
def match_skills():
    """Skill gap analysis using NLP + job description matching"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    job_title = data.get("job_title", "")
    user_skills = data.get("skills", [])
    result = matcher.match(job_title, user_skills)
    return jsonify(result)


@app.route("/api/career-predict", methods=["POST"])
def career_predict():
    """Career path prediction using Random Forest classifier"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    result = predictor.predict(data)
    return jsonify(result)


@app.route("/api/generate-objective", methods=["POST"])
def generate_objective():
    """AI-generated career objective based on profile"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    objective = assistant.generate_objective(data)
    return jsonify({"objective": objective})


@app.route("/api/chat", methods=["POST"])
def chat():
    """AI Assistant chat endpoint"""
    data = request.get_json()
    message = data.get("message", "")
    context = data.get("context", {})
    reply = assistant.chat(message, context)
    return jsonify({"reply": reply})


@app.route("/api/generate-pdf", methods=["POST"])
def generate_pdf():
    """Generate downloadable PDF resume"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pdf_path = generate_pdf_resume(data)
    return send_file(pdf_path, as_attachment=True, download_name="resume.pdf")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "AI Resume Builder"})


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
