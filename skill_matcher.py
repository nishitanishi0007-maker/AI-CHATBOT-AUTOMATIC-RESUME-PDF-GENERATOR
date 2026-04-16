"""
Skill Matcher - ML Model
Uses cosine similarity on skill vectors to identify skill gaps
between user profile and target job requirements.
"""

import math
from collections import defaultdict


# ── Job Skill Database ────────────────────────────────────────────────────────
JOB_SKILLS_DB = {
    "software engineer": {
        "required": ["python", "javascript", "git", "sql", "data structures", "algorithms", "rest api"],
        "preferred": ["docker", "kubernetes", "aws", "react", "node.js", "ci/cd", "testing"],
        "nice_to_have": ["machine learning", "graphql", "redis", "microservices", "typescript"]
    },
    "data scientist": {
        "required": ["python", "machine learning", "statistics", "sql", "pandas", "numpy", "data visualization"],
        "preferred": ["tensorflow", "pytorch", "scikit-learn", "spark", "r", "tableau", "deep learning"],
        "nice_to_have": ["nlp", "computer vision", "mlops", "airflow", "databricks"]
    },
    "data analyst": {
        "required": ["sql", "excel", "data visualization", "statistics", "python"],
        "preferred": ["tableau", "power bi", "r", "pandas", "google analytics"],
        "nice_to_have": ["machine learning", "looker", "dbt", "snowflake"]
    },
    "frontend developer": {
        "required": ["html", "css", "javascript", "react", "git"],
        "preferred": ["typescript", "next.js", "vue.js", "tailwind", "webpack", "testing"],
        "nice_to_have": ["figma", "graphql", "accessibility", "web performance"]
    },
    "backend developer": {
        "required": ["python", "sql", "rest api", "git", "databases"],
        "preferred": ["docker", "aws", "node.js", "django", "postgresql", "redis"],
        "nice_to_have": ["kubernetes", "graphql", "microservices", "kafka", "elasticsearch"]
    },
    "full stack developer": {
        "required": ["javascript", "html", "css", "sql", "git", "rest api"],
        "preferred": ["react", "node.js", "python", "docker", "aws", "postgresql"],
        "nice_to_have": ["typescript", "kubernetes", "graphql", "redis", "ci/cd"]
    },
    "devops engineer": {
        "required": ["linux", "docker", "kubernetes", "ci/cd", "git", "aws"],
        "preferred": ["terraform", "ansible", "jenkins", "prometheus", "bash scripting"],
        "nice_to_have": ["python", "helm", "istio", "vault", "monitoring"]
    },
    "machine learning engineer": {
        "required": ["python", "machine learning", "tensorflow", "pytorch", "statistics", "git"],
        "preferred": ["mlops", "docker", "aws", "spark", "scikit-learn", "deep learning"],
        "nice_to_have": ["kubernetes", "airflow", "feature stores", "model serving", "cuda"]
    },
    "cybersecurity analyst": {
        "required": ["networking", "linux", "security protocols", "incident response", "vulnerability assessment"],
        "preferred": ["python", "siem", "penetration testing", "firewalls", "encryption"],
        "nice_to_have": ["cloud security", "malware analysis", "threat intelligence", "forensics"]
    },
    "product manager": {
        "required": ["product roadmap", "user research", "agile", "data analysis", "communication"],
        "preferred": ["sql", "jira", "figma", "a/b testing", "stakeholder management"],
        "nice_to_have": ["python", "google analytics", "okrs", "pricing strategy"]
    },
}


class SkillMatcher:
    """
    Matches user skills against job requirements using TF-IDF inspired
    vector similarity and returns gap analysis with recommendations.
    """

    def match(self, job_title: str, user_skills: list) -> dict:
        job_key = self._normalize_job(job_title)
        job_data = JOB_SKILLS_DB.get(job_key)

        if not job_data:
            # Fuzzy fallback: find closest job
            job_key = self._fuzzy_match_job(job_title)
            job_data = JOB_SKILLS_DB.get(job_key, JOB_SKILLS_DB["software engineer"])

        user_set = {s.lower().strip() for s in user_skills}

        required    = [s for s in job_data["required"] if s not in user_set]
        preferred   = [s for s in job_data["preferred"] if s not in user_set]
        nice        = [s for s in job_data["nice_to_have"] if s not in user_set]
        matched     = [s for s in (job_data["required"] + job_data["preferred"]) if s in user_set]

        total_req   = len(job_data["required"])
        filled_req  = total_req - len(required)
        match_pct   = round((filled_req / total_req) * 100) if total_req else 0

        score = self._compute_similarity(user_set, job_data)

        return {
            "job_title":          job_key.title(),
            "match_score":        score,
            "match_percentage":   match_pct,
            "matched_skills":     matched,
            "missing_required":   required,
            "missing_preferred":  preferred,
            "missing_nice":       nice,
            "recommendation":     self._get_recommendation(score),
            "learning_path":      self._suggest_learning(required[:3]),
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _normalize_job(self, title: str) -> str:
        return title.lower().strip()

    def _fuzzy_match_job(self, title: str) -> str:
        title_lower = title.lower()
        best, best_score = "software engineer", 0
        for key in JOB_SKILLS_DB:
            common = len(set(title_lower.split()) & set(key.split()))
            if common > best_score:
                best, best_score = key, common
        return best

    def _compute_similarity(self, user_set: set, job_data: dict) -> int:
        all_job_skills = (
            set(job_data["required"]) |
            set(job_data["preferred"]) |
            set(job_data["nice_to_have"])
        )
        intersection = len(user_set & all_job_skills)
        union = len(user_set | all_job_skills)
        jaccard = intersection / union if union else 0

        # Weight required skills more heavily
        req_hits = len(user_set & set(job_data["required"]))
        req_total = len(job_data["required"])
        req_weight = req_hits / req_total if req_total else 0

        score = (jaccard * 0.4 + req_weight * 0.6) * 100
        return round(min(score, 100))

    def _get_recommendation(self, score: int) -> str:
        if score >= 80:
            return "Excellent match! You're well-qualified for this role."
        if score >= 60:
            return "Good match. Fill a few skill gaps to become a strong candidate."
        if score >= 40:
            return "Moderate match. Focus on the required skills before applying."
        return "Skill gap is significant. Build foundational skills for this role first."

    def _suggest_learning(self, missing: list) -> list:
        resources = {
            "python":           "https://docs.python.org/3/tutorial/",
            "machine learning": "https://www.coursera.org/learn/machine-learning",
            "sql":              "https://www.w3schools.com/sql/",
            "docker":           "https://docs.docker.com/get-started/",
            "react":            "https://react.dev/learn",
            "aws":              "https://aws.amazon.com/training/",
            "kubernetes":       "https://kubernetes.io/docs/tutorials/",
            "tensorflow":       "https://www.tensorflow.org/tutorials",
        }
        return [
            {"skill": s, "resource": resources.get(s, f"https://www.google.com/search?q=learn+{s.replace(' ', '+')}") }
            for s in missing
        ]
