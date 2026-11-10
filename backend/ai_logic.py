import os

import ollama
from schemas import AnalysisResult, Checklist

LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
# A v1/v2 answer is ~150 tokens; a v3 checklist for a real posting can pass 512 (5 of 13 were cut off at 512).
# Without a cap, one real-resume eval call ran past 4,600 tokens and never ended; a truncated answer fails JSON
# validation and becomes FALLBACK_RESULT instead of hanging the request.
MAX_ANSWER_TOKENS = 1024

# v1: the original open-ended prompt. v2: adds a scoring rubric, written to fix the over-scoring the eval found.
# v3: no score from the model at all; it judges each core requirement and the score is computed in code, written
# to fix the over-scoring the real-resume eval found (+43 for v2).
PROMPT_VERSIONS = ("v1", "v2", "v3")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")  # v2 won the eval: lower bias, higher correlation and accuracy
if PROMPT_VERSION not in PROMPT_VERSIONS:
    raise ValueError(f"PROMPT_VERSION must be one of {PROMPT_VERSIONS}, got {PROMPT_VERSION!r}")

# The one place the hiring policy lives: the rubric prompt and recommendation_for() are both built from it,
# and the eval labels follow the same bands. (lowest score, recommendation, what the band means)
BANDS = (
    (85, "Hire", "meets every core requirement"),
    (50, "Interview", "meets most core requirements; the gaps could be learned on the job"),
    (35, "Review", "meets some core requirements but has significant gaps"),
    (0, "No Hire", "meets few core requirements or works in a different field"),
)
_BAND_LINES = "".join(
    f'    - {low}-{high}: {meaning} -> "{label}"\n'
    for (low, label, meaning), high in zip(BANDS, (100, *(b[0] - 1 for b in BANDS[:-1])), strict=True)
)

# The example is deliberately not an eval case.
RUBRIC = f"""
    SCORING RUBRIC. Score only against the requirements listed in the job description:
{_BAND_LINES}    The recommendation must match the score band. Generic overlap (for example "uses Python" or
    "ran A/B tests") does not count as meeting a specialised requirement.
    Example: a chef applying for a data engineering role scores about 5 -> "No Hire".
"""

# Returned (as this exact object) when the LLM call or its output fails, so callers can tell via `is`.
FALLBACK_RESULT = AnalysisResult(
    match_percentage=0, missing_keywords=[], recommendation="Review", analysis_summary="Analysis failed due to an error"
)


def recommendation_for(score: int) -> str:
    """The rubric's bands. The eval found the model's own recommendation contradicted its score ~30% of the time."""
    return next(label for low, label, _ in BANDS if score >= low)


def with_score_recommendation(result: AnalysisResult) -> AnalysisResult:
    if result is FALLBACK_RESULT:  # a failed call stays "Review", not a 0-score "No Hire"
        return result
    return result.model_copy(update={"recommendation": recommendation_for(result.match_percentage)})


def llama_query(prompt: str):
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"num_predict": MAX_ANSWER_TOKENS},
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Ollama Error:{e}")
        return None


CHECKLIST_PROMPT = """
    You are an expert Technical Recruiter. Check the Resume against the Job Description.

    RESUME:
    {resume}

    JOB DESCRIPTION:
    {jd}

    List the job's core requirements: what it says is required, at most 8, including the years or level of
    experience if it states one. Ignore nice-to-haves, location, language, salary and benefits. For each one,
    judge the resume strictly:
    - "met": the resume shows it directly, at the level asked.
    - "partial": related experience, or the right skill below the level asked.
    - "missing": not shown. Generic overlap (for example "uses Python" for "production ML with LLMs") is missing.

    Return the result strictly as a JSON object:
    {{
        "requirements": [{{"requirement": (string), "status": "met" | "partial" | "missing",
                          "evidence": (string: at most 10 words from the resume, or "none")}}],
        "analysis_summary": (string: brief explanation)
    }}
    """


def score_checklist(checklist: Checklist) -> AnalysisResult:
    """met = 1, partial = 0.5, missing = 0, as a percentage, so "meets every core requirement" is 100 (Hire)."""
    points = {"met": 1.0, "partial": 0.5, "missing": 0.0}
    reqs = checklist.requirements
    score = round(100 * sum(points[r.status] for r in reqs) / len(reqs))
    return AnalysisResult(
        match_percentage=score,
        missing_keywords=tuple(r.requirement for r in reqs if r.status == "missing"),
        recommendation=recommendation_for(score),
        analysis_summary=checklist.analysis_summary,
    )


def analyze_match(
    resume_description: str, job_description: str, prompt_version: str = PROMPT_VERSION
) -> AnalysisResult:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"prompt_version must be one of {PROMPT_VERSIONS}, got {prompt_version!r}")
    if prompt_version == "v3":
        response_text = llama_query(CHECKLIST_PROMPT.format(resume=resume_description, jd=job_description))
        try:
            return score_checklist(Checklist.model_validate_json(response_text or ""))
        except Exception as e:  # no answer, bad JSON or a schema violation (including a truncated answer)
            print(f"Parsing Error:{e}")
            return FALLBACK_RESULT
    rubric = RUBRIC if prompt_version == "v2" else ""
    prompt = f"""
     You are an expert Technical Recruiter. Compare the following Resume and Job Description.

    RESUME:
    {resume_description} 

    JOB DESCRIPTION:
    {job_description}
   {rubric}

    Analyze the match and return the result strictly as a JSON object
    REQUIRED JSON STRUCTURE:
    {{
        "match_percentage":(integer between 0-100),
        "missing_keywords":[list of strings],
        "recommendation":"Hire" | "No Hire" | "Interview" | "Review",
        "analysis_summary":(string:brief explanation of the score)
    }}
    """

    response_text = llama_query(prompt)
    try:
        return AnalysisResult.model_validate_json(response_text or "")
    except Exception as e:  # no answer, bad JSON or a schema violation
        print(f"Parsing Error:{e}")
        return FALLBACK_RESULT
