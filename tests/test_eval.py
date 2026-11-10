import math

import pytest
from run_eval import compute_metrics


def test_compute_metrics_skips_failed_runs():
    rows = [
        {
            "expected_score": 90,
            "expected_recommendation": "Hire",
            "context_chars": 1000,
            "scores": [80, 100],
            "recommendations": ["Hire", "Interview"],
            "failures": 1,
        },
        {
            "expected_score": 10,
            "expected_recommendation": "No Hire",
            "context_chars": 2000,
            "scores": [20, 20],
            "recommendations": ["No Hire", "No Hire"],
            "failures": 0,
        },
        {
            "expected_score": 50,
            "expected_recommendation": "Review",
            "context_chars": 3000,
            "scores": [],
            "recommendations": [],
            "failures": 2,
        },
    ]
    m = compute_metrics(rows)
    assert m["mae"] == 5
    assert m["bias"] == 5  # per-case errors are 0 and +10
    assert m["correlation"] == pytest.approx(1.0)
    assert m["rec_accuracy"] == 0.75
    assert m["derived_rec_accuracy"] == 0.75  # 80 -> Interview misses Hire; the rest match
    assert m["score_stdev"] == 5
    assert m["failures"] == 3
    assert m["context_chars"] == 1500


def test_compute_metrics_survives_every_call_failing():
    rows = [
        {
            "expected_score": 50,
            "expected_recommendation": "Review",
            "context_chars": 100,
            "scores": [],
            "recommendations": [],
            "failures": 3,
        }
    ] * 2
    m = compute_metrics(rows)
    assert m["failures"] == 6
    assert math.isnan(m["mae"])


def test_report_uses_current_labels():
    # --rescore joins labels at report time, so editing cases.csv changes the metrics without new LLM calls
    from run_eval import report

    row = {"context_chars": 100, "scores": [90], "recommendations": ["Hire"], "failures": 0}
    saved = {
        "runs": 1,
        "configs": ["rag:v2"],
        "setup": "LLM `the-run-model`",
        "latency": {"rag:v2": 1.0},
        "rows": {"rag:v2": [{**row, "resume": "a", "jd": "x"}, {**row, "resume": "b", "jd": "x", "scores": [10]}]},
    }
    cases = [
        {"resume": "a", "jd": "x", "expected_score": "90", "expected_recommendation": "Hire"},
        {"resume": "b", "jd": "x", "expected_score": "40", "expected_recommendation": "Review"},
    ]

    text = report(saved, cases)
    assert "| -15.0 |" in text  # bias: (0 + (10 - 40)) / 2
    assert "the-run-model" in text  # the run's saved settings, not today's environment

    new_case = {"resume": "c", "jd": "x", "expected_score": "50", "expected_recommendation": "Interview"}
    with pytest.raises(ValueError, match="need a full run"):
        report(saved, [*cases, new_case])


def test_load_texts_reads_txt_and_rejects_empty_resumes(tmp_path):
    from run_eval import load_texts

    (tmp_path / "a.txt").write_text("Python, FastAPI", encoding="utf-8")
    (tmp_path / "blank.txt").write_text("  \n", encoding="utf-8")
    (tmp_path / "x.txt").write_text("Backend role", encoding="utf-8")

    texts = load_texts([{"resume": "a", "jd": "x"}], tmp_path, tmp_path)
    assert texts == {"a": "Python, FastAPI", "jd:x": "Backend role"}
    with pytest.raises(ValueError, match="blank"):  # the app would reject it, so the eval must not score it
        load_texts([{"resume": "blank", "jd": "x"}], tmp_path, tmp_path)


def test_redact_keeps_skills_and_removes_personal_details():
    from redact import redact

    text = "\n".join(
        [
            "Priya Raman",
            "priya.raman@example.com | +91 98765 43210 | linkedin.com/in/priyaraman",
            "Date of Birth: 01/02/1999",
            "Gender: Female",
            "Senior Software Engineer, PayLoop (2021 - 2023)",
            "- Built FastAPI services; Python, PostgreSQL, Docker",
            "Declaration: I, Priya Raman, confirm the above.",
        ]
    )
    out = redact(text)

    assert "Priya" not in out and "Raman" not in out
    assert "example.com" not in out and "98765" not in out and "linkedin" not in out
    assert "1999" not in out and "Female" not in out
    assert "FastAPI" in out and "(2021 - 2023)" in out  # skills and date ranges survive


def test_redact_leaves_job_title_first_lines_alone():
    from redact import redact

    assert redact("ACCOUNTANT\nSummary\nAccountant with 5 years of experience").startswith("ACCOUNTANT")
