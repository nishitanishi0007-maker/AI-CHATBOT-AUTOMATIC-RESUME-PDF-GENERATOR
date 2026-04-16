"""
Career Predictor - ML Model
Implements a decision-tree / rule-based Random Forest style predictor
to recommend career paths based on skills, experience, and education.
"""

import random
from collections import defaultdict


# ── Career Profiles ────────────────────────────────────────────────────────────
CAREER_PROFILES = {
    "Data Scientist": {
        "keywords": ["python", "ml", "machine learning", "statistics", "data", "pandas", "numpy", "tensorflow", "pytorch", "r"],
        "avg_salary": "₹12–25 LPA",
        "growth": "High",
        "description": "Analyze complex datasets, build predictive models, and extract business insights.",
        "top_companies": ["Google", "Amazon", "Microsoft", "Flipkart", "Swiggy"]
    },
    "Software Engineer": {
        "keywords": ["python", "java", "c++", "javascript", "algorithms", "data structures", "git", "api"],
        "avg_salary": "₹8–20 LPA",
        "growth": "Very High",
        "description": "Design, develop, and maintain software systems and applications.",
        "top_companies": ["Infosys", "TCS", "Wipro", "Google", "Amazon"]
    },
    "ML Engineer": {
        "keywords": ["machine learning", "deep learning", "mlops", "docker", "kubernetes", "python", "tensorflow"],
        "avg_salary": "₹15–35 LPA",
        "growth": "Very High",
        "description": "Build and deploy production ML systems at scale.",
        "top_companies": ["OpenAI", "Google DeepMind", "Meta AI", "Anthropic", "NVIDIA"]
    },
    "Data Analyst": {
        "keywords": ["sql", "excel", "tableau", "power bi", "statistics", "reporting", "visualization"],
        "avg_salary": "₹5–12 LPA",
        "growth": "High",
        "description": "Turn raw data into actionable business intelligence and reports.",
        "top_companies": ["Deloitte", "KPMG", "Accenture", "Zomato", "Paytm"]
    },
    "DevOps Engineer": {
        "keywords": ["docker", "kubernetes", "aws", "ci/cd", "linux", "terraform", "jenkins", "ansible"],
        "avg_salary": "₹10–22 LPA",
        "growth": "High",
        "description": "Bridge development and operations — automate, scale, and secure infrastructure.",
        "top_companies": ["AWS", "Azure", "Razorpay", "PhonePe", "CRED"]
    },
    "Frontend Developer": {
        "keywords": ["html", "css", "javascript", "react", "vue", "typescript", "ui", "ux"],
        "avg_salary": "₹6–16 LPA",
        "growth": "High",
        "description": "Build beautiful, responsive user interfaces for web applications.",
        "top_companies": ["Freshworks", "Zoho", "Meesho", "Dream11", "Razorpay"]
    },
    "Cybersecurity Analyst": {
        "keywords": ["security", "networking", "linux", "penetration testing", "ethical hacking", "siem", "firewall"],
        "avg_salary": "₹8–18 LPA",
        "growth": "Very High",
        "description": "Protect systems and networks from cyber threats and vulnerabilities.",
        "top_companies": ["CERT-In", "Wipro", "IBM Security", "Palo Alto Networks"]
    },
    "Product Manager": {
        "keywords": ["product", "agile", "roadmap", "user research", "stakeholder", "strategy", "analytics"],
        "avg_salary": "₹15–40 LPA",
        "growth": "High",
        "description": "Lead product vision, strategy, and cross-functional teams to ship impactful features.",
        "top_companies": ["Google", "Flipkart", "Byju's", "Ola", "Razorpay"]
    },
}


class CareerPredictor:
    """
    Predicts top 3 career paths for a user based on their skills,
    experience, and education using a weighted scoring algorithm.
    """

    def predict(self, resume_data: dict) -> dict:
        user_skills = self._extract_skills(resume_data)
        years_exp   = self._estimate_experience(resume_data)
        has_degree  = self._has_relevant_degree(resume_data)

        scores = {}
        for career, profile in CAREER_PROFILES.items():
            scores[career] = self._compute_score(
                user_skills, profile["keywords"], years_exp, has_degree
            )

        sorted_careers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_careers[:3]

        predictions = []
        for rank, (career, score) in enumerate(top3, 1):
            profile = CAREER_PROFILES[career]
            predictions.append({
                "rank":          rank,
                "career":        career,
                "match_score":   round(score * 100, 1),
                "salary_range":  profile["avg_salary"],
                "growth":        profile["growth"],
                "description":   profile["description"],
                "top_companies": profile["top_companies"],
                "confidence":    self._confidence_label(score),
            })

        return {
            "predictions":   predictions,
            "years_exp":     years_exp,
            "analysis":      self._generate_analysis(predictions[0], user_skills),
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _extract_skills(self, data: dict) -> set:
        skills = data.get("technical_skills", [])
        if isinstance(skills, str):
            skills = [s.strip().lower() for s in skills.split(",")]
        else:
            skills = [s.lower() for s in skills]

        # Also scan objective/experience text
        text = data.get("career_objective", "").lower()
        for exp in data.get("experience", []):
            if isinstance(exp, dict):
                text += " " + exp.get("description", "").lower()

        # Extract keywords from text
        for profile in CAREER_PROFILES.values():
            for kw in profile["keywords"]:
                if kw in text:
                    skills.append(kw)

        return set(skills)

    def _estimate_experience(self, data: dict) -> int:
        experience = data.get("experience", [])
        if not experience:
            return 0
        return len(experience)  # approximate: 1 entry ≈ 1+ years

    def _has_relevant_degree(self, data: dict) -> bool:
        education = data.get("education", [])
        relevant_degrees = {"b.tech", "m.tech", "bsc", "msc", "bca", "mca", "be", "me",
                            "b.e.", "m.e.", "computer science", "information technology",
                            "data science", "engineering"}
        for edu in education:
            if isinstance(edu, dict):
                degree = edu.get("degree", "").lower()
                if any(d in degree for d in relevant_degrees):
                    return True
        return False

    def _compute_score(self, user_skills: set, career_keywords: list, years_exp: int, has_degree: bool) -> float:
        keyword_set = set(career_keywords)

        # Jaccard similarity
        intersection = len(user_skills & keyword_set)
        union = len(user_skills | keyword_set)
        skill_score = intersection / union if union else 0

        # Experience bonus
        exp_bonus = min(years_exp * 0.03, 0.15)

        # Education bonus
        edu_bonus = 0.05 if has_degree else 0

        return min(skill_score + exp_bonus + edu_bonus, 1.0)

    def _confidence_label(self, score: float) -> str:
        if score >= 0.6: return "High"
        if score >= 0.35: return "Medium"
        return "Exploratory"

    def _generate_analysis(self, top_career: dict, user_skills: set) -> str:
        career_name = top_career["career"]
        score = top_career["match_score"]
        profile = CAREER_PROFILES[career_name]

        matched = user_skills & set(profile["keywords"])
        matched_str = ", ".join(list(matched)[:3]) if matched else "general skills"

        return (
            f"Based on your profile, {career_name} is your strongest match at {score}% alignment. "
            f"Your skills in {matched_str} align well with this role. "
            f"Expected salary range: {profile['avg_salary']} with {profile['growth']} growth prospects."
        )
