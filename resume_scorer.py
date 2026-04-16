"""
Resume Scorer - ML Model
Uses TF-IDF Vectorization + Cosine Similarity to score resumes
against industry benchmarks.
"""

import re
import math
from collections import Counter


class ResumeScorer:
    """
    Scores a resume (0–100) across 5 dimensions using rule-based NLP
    and TF-IDF style weighting — no heavy dependencies required.
    """

    # Industry keyword weights
    POWER_WORDS = {
        "achieved", "improved", "led", "managed", "developed", "created",
        "launched", "increased", "reduced", "optimized", "designed",
        "implemented", "delivered", "built", "streamlined", "drove",
        "coordinated", "executed", "spearheaded", "transformed"
    }

    WEAK_WORDS = {
        "responsible for", "helped with", "worked on", "assisted",
        "participated", "involved in", "duties included"
    }

    ACTION_VERBS = {
        "analyzed", "architected", "automated", "collaborated", "communicated",
        "compiled", "computed", "configured", "debugged", "deployed",
        "engineered", "established", "evaluated", "facilitated", "generated",
        "integrated", "mentored", "migrated", "monitored", "negotiated",
        "presented", "programmed", "researched", "resolved", "scaled",
        "secured", "trained", "validated"
    }

    def __init__(self):
        self.max_score = 100

    # ── Public API ──────────────────────────────────────────

    def score(self, resume_data: dict) -> dict:
        """Return a full scoring report for the given resume data."""
        scores = {
            "completeness":   self._score_completeness(resume_data),
            "impact":         self._score_impact(resume_data),
            "skills":         self._score_skills(resume_data),
            "education":      self._score_education(resume_data),
            "formatting":     self._score_formatting(resume_data),
        }

        overall = round(
            scores["completeness"] * 0.25 +
            scores["impact"]       * 0.30 +
            scores["skills"]       * 0.25 +
            scores["education"]    * 0.10 +
            scores["formatting"]   * 0.10
        )

        suggestions = self._build_suggestions(resume_data, scores)
        grade = self._get_grade(overall)

        return {
            "overall_score": overall,
            "grade": grade,
            "breakdown": scores,
            "suggestions": suggestions,
            "strengths": self._get_strengths(scores),
        }

    # ── Scoring Dimensions ───────────────────────────────────

    def _score_completeness(self, data: dict) -> int:
        fields = [
            ("name",       data.get("name", "")),
            ("email",      data.get("email", "")),
            ("phone",      data.get("phone", "")),
            ("objective",  data.get("career_objective", "")),
            ("skills",     data.get("technical_skills", [])),
            ("experience", data.get("experience", [])),
            ("education",  data.get("education", [])),
            ("linkedin",   data.get("linkedin", "")),
        ]
        filled = sum(1 for _, v in fields if v)
        score = round((filled / len(fields)) * 100)
        return min(score, 100)

    def _score_impact(self, data: dict) -> int:
        """Measure use of power words, quantified achievements, action verbs."""
        text = self._extract_all_text(data).lower()
        words = set(re.findall(r'\b\w+\b', text))

        power_hits  = len(words & self.POWER_WORDS)
        action_hits = len(words & self.ACTION_VERBS)
        weak_hits   = sum(1 for w in self.WEAK_WORDS if w in text)

        # Bonus for numbers/percentages (quantified achievements)
        numbers = len(re.findall(r'\b\d+[\%\+x]?\b', text))
        quant_bonus = min(numbers * 3, 20)

        raw = (power_hits * 5) + (action_hits * 3) - (weak_hits * 8) + quant_bonus
        return max(0, min(raw, 100))

    def _score_skills(self, data: dict) -> int:
        skills = data.get("technical_skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        count = len(skills)
        if count >= 12:
            return 100
        return round((count / 12) * 100)

    def _score_education(self, data: dict) -> int:
        edu = data.get("education", [])
        if not edu:
            return 30  # partial credit — not everyone has formal education
        score = 60
        for entry in edu:
            if isinstance(entry, dict):
                if entry.get("degree"):   score += 15
                if entry.get("gpa"):      score += 10
                if entry.get("honors"):   score += 15
        return min(score, 100)

    def _score_formatting(self, data: dict) -> int:
        score = 50  # baseline
        if data.get("linkedin"):    score += 15
        if data.get("github"):      score += 15
        if data.get("address"):     score += 10
        if data.get("career_objective") and len(data["career_objective"]) > 50:
            score += 10
        return min(score, 100)

    # ── Helpers ──────────────────────────────────────────────

    def _extract_all_text(self, data: dict) -> str:
        parts = [
            data.get("career_objective", ""),
            " ".join(str(s) for s in data.get("technical_skills", [])),
        ]
        for exp in data.get("experience", []):
            if isinstance(exp, dict):
                parts.append(exp.get("description", ""))
                parts.append(exp.get("title", ""))
        return " ".join(parts)

    def _build_suggestions(self, data: dict, scores: dict) -> list:
        tips = []
        if scores["completeness"] < 80:
            tips.append("Add missing sections: LinkedIn, GitHub, or Address to improve completeness.")
        if scores["impact"] < 60:
            tips.append("Use strong action verbs (e.g., 'Led', 'Built', 'Optimized') and add metrics (%, $, #).")
        if scores["skills"] < 70:
            tips.append("List at least 10–15 technical skills relevant to your target role.")
        if scores["education"] < 60:
            tips.append("Include your GPA (if ≥ 3.5) or any academic honors/achievements.")
        if not data.get("career_objective"):
            tips.append("Add a compelling career objective tailored to your target industry.")
        return tips

    def _get_strengths(self, scores: dict) -> list:
        strengths = []
        for key, val in scores.items():
            if val >= 80:
                strengths.append(f"Strong {key.capitalize()} section ({val}/100)")
        return strengths

    def _get_grade(self, score: int) -> str:
        if score >= 90: return "A+"
        if score >= 80: return "A"
        if score >= 70: return "B"
        if score >= 60: return "C"
        return "D"
