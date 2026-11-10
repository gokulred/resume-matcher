import json
import os
import tempfile

import ai_logic
import main
import parser
import pytest
from fastapi.testclient import TestClient
from schemas import AnalysisResult
from vector_db import chunk_text

RESULT = AnalysisResult(
    match_percentage=80, missing_keywords=["AWS"], recommendation="Interview", analysis_summary="Good fit"
)


def test_chunk_text_splits_single_spaced_text():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(30))
    chunks = chunk_text(text, max_chars=400)
    assert len(chunks) > 1
    assert "\n".join(chunks).splitlines() == text.splitlines()


def test_chunk_text_drops_blank_lines():
    assert chunk_text("a\n\n\nb") == ["a\nb"]


def test_analyze_match_parses_valid_json(monkeypatch):
    monkeypatch.setattr(ai_logic, "llama_query", lambda prompt: RESULT.model_dump_json())
    assert ai_logic.analyze_match("resume", "jd") == RESULT


def test_llama_query_caps_answer_length(monkeypatch):
    calls = []
    monkeypatch.setattr(ai_logic.ollama, "chat", lambda **kw: calls.append(kw) or {"message": {"content": "{}"}})
    ai_logic.llama_query("prompt")
    assert calls[0]["options"]["num_predict"] == ai_logic.MAX_ANSWER_TOKENS  # an uncapped call once never ended


def test_analyze_match_falls_back_on_bad_json(monkeypatch):
    monkeypatch.setattr(ai_logic, "llama_query", lambda prompt: '{"match_percentage": 150}')
    # run_eval counts failures by identity, so the fallback must be this exact object
    assert ai_logic.analyze_match("resume", "jd") is ai_logic.FALLBACK_RESULT


def test_match_pdf_returns_analysis(monkeypatch):
    monkeypatch.setattr(main, "parse_resume", lambda path: "Python developer")
    monkeypatch.setattr(main, "analyze_match", lambda ctx, jd: RESULT)
    monkeypatch.setattr(main, "CONTEXT_MODE", "full")

    response = TestClient(main.app).post(
        "/match-pdf", files={"file": ("cv.pdf", b"%PDF", "application/pdf")}, data={"job_description": "Python"}
    )

    assert response.status_code == 200
    assert response.json()["analysis"]["match_percentage"] == 80
    assert response.json()["parsed_text_preview"] == "Python developer"


def test_match_pdf_ignores_client_filename_and_cleans_up(monkeypatch):
    seen = []
    monkeypatch.setattr(main, "parse_resume", lambda path: seen.append(path))

    response = TestClient(main.app).post(
        "/match-pdf", files={"file": ("../../evil.pdf", b"x", "application/pdf")}, data={"job_description": "jd"}
    )

    assert response.status_code == 400
    assert os.path.dirname(seen[0]) == tempfile.gettempdir()
    assert "evil" not in seen[0]
    assert not os.path.exists(seen[0])


def test_unexpected_error_hides_internal_details(monkeypatch):
    def broken(ctx, jd):
        raise RuntimeError("connect to http://ollama:11434 failed at C:\\secret\\path")

    monkeypatch.setattr(main, "parse_resume", lambda path: "Python developer")
    monkeypatch.setattr(main, "analyze_match", broken)
    monkeypatch.setattr(main, "CONTEXT_MODE", "full")

    response = TestClient(main.app).post(
        "/match-pdf", files={"file": ("cv.pdf", b"%PDF", "application/pdf")}, data={"job_description": "Python"}
    )

    assert response.status_code == 500
    assert "secret" not in response.text and "11434" not in response.text


def test_get_context_rejects_unknown_mode():
    with pytest.raises(ValueError):
        main.get_context("resume", "jd", "ful")


def test_rag_sends_half_the_chunks_with_one_embedding_call(monkeypatch):
    import math

    from vector_db import VectorDB, chunk_text

    calls = []

    def fake_embeddings(self, texts):
        calls.append(len(texts))
        return [[float(i), 1.0] for i in range(len(texts))]

    monkeypatch.setattr(VectorDB, "get_embeddings", fake_embeddings)
    resume = "\n".join(f"line {i} " + "x" * 80 for i in range(15))
    chunks = chunk_text(resume)

    context = VectorDB().retrieve(resume, "jd", 0.5)

    assert sum(c in context for c in chunks) == math.ceil(len(chunks) * 0.5)
    assert calls == [len(chunks) + 1]  # every chunk and the job description in one request


def test_parser_falls_back_to_pypdf_when_llamaparse_fails(monkeypatch):
    class Page:
        def extract_text(self):
            return "Python developer"

    def broken(path):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(parser, "PDF_PARSER", "llamaparse")
    monkeypatch.setattr(parser, "_parse_with_llamaparse", broken)
    monkeypatch.setattr(parser, "PdfReader", lambda path: type("R", (), {"pages": [Page()]})())

    assert parser.parse_resume("cv.pdf") == "Python developer"


def test_rejects_oversized_upload_before_parsing(monkeypatch):
    monkeypatch.setattr(main, "parse_resume", lambda path: pytest.fail("should not parse"))

    response = TestClient(main.app).post(
        "/match-pdf",
        files={"file": ("cv.pdf", b"x" * (main.MAX_UPLOAD_BYTES + 1), "application/pdf")},
        data={"job_description": "jd"},
    )

    assert response.status_code == 413


def test_rubric_only_in_v2_prompt(monkeypatch):
    prompts = []
    monkeypatch.setattr(ai_logic, "llama_query", lambda prompt: prompts.append(prompt))

    ai_logic.analyze_match("resume", "jd", "v1")
    ai_logic.analyze_match("resume", "jd", "v2")

    assert "SCORING RUBRIC" not in prompts[0]
    assert "SCORING RUBRIC" in prompts[1]


def test_analyze_match_rejects_unknown_prompt_version():
    with pytest.raises(ValueError):
        ai_logic.analyze_match("resume", "jd", "v9")


def test_v3_score_is_computed_from_the_checklist(monkeypatch):
    answer = {
        "requirements": [
            {"requirement": "Python", "status": "met", "evidence": "5 years Python"},
            {"requirement": "Kubernetes", "status": "partial", "evidence": "Docker only"},
            {"requirement": "LLM experience", "status": "missing", "evidence": "none"},
            {"requirement": "5+ years", "status": "missing", "evidence": "none"},
        ],
        "analysis_summary": "Strong Python, little else",
    }
    monkeypatch.setattr(ai_logic, "llama_query", lambda prompt: json.dumps(answer))

    result = ai_logic.analyze_match("resume", "jd", "v3")

    assert result.match_percentage == 38  # (1 + 0.5) / 4
    assert result.recommendation == "Review"
    assert result.missing_keywords == ("LLM experience", "5+ years")


def test_v3_truncated_answer_falls_back(monkeypatch):
    # what num_predict leaves when it cuts an answer off mid-list
    monkeypatch.setattr(ai_logic, "llama_query", lambda prompt: '{"requirements": [{"requirement": "Py')
    assert ai_logic.analyze_match("resume", "jd", "v3") is ai_logic.FALLBACK_RESULT


def test_vector_db_survives_many_requests(monkeypatch):
    # a single shared collection used to crash after ~20 requests
    import random

    from vector_db import VectorDB

    rng = random.Random(0)
    monkeypatch.setattr(
        VectorDB, "get_embeddings", lambda self, texts: [[rng.random() for _ in range(8)] for _ in texts]
    )
    db = VectorDB()
    resume = "\n".join(f"line {i} " + "x" * 80 for i in range(20))

    for _ in range(50):
        assert db.retrieve(resume, "jd", 0.5)

    assert db.client.list_collections() == []


def test_api_derives_recommendation_from_score(monkeypatch):
    contradicting = AnalysisResult(
        match_percentage=90, missing_keywords=[], recommendation="Review", analysis_summary="s"
    )
    monkeypatch.setattr(main, "parse_resume", lambda path: "Python developer")
    monkeypatch.setattr(main, "analyze_match", lambda ctx, jd: contradicting)
    monkeypatch.setattr(main, "CONTEXT_MODE", "full")

    response = TestClient(main.app).post(
        "/match-pdf", files={"file": ("cv.pdf", b"%PDF", "application/pdf")}, data={"job_description": "Python"}
    )

    assert response.json()["analysis"]["recommendation"] == "Hire"


def test_failed_call_stays_review():
    assert ai_logic.with_score_recommendation(ai_logic.FALLBACK_RESULT).recommendation == "Review"


def test_embedding_failure_surfaces_and_leaves_nothing_behind(monkeypatch):
    from vector_db import VectorDB

    def ollama_down(self, texts):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(VectorDB, "get_embeddings", ollama_down)
    db = VectorDB()

    with pytest.raises(RuntimeError, match="ollama down"):  # the real error, not "collection does not exist"
        db.retrieve("some resume line", "jd", 0.5)
    assert db.client.list_collections() == []


def test_retrieved_chunks_keep_resume_order(monkeypatch):
    from vector_db import VectorDB, chunk_text

    resume = "\n".join(f"line {i} " + "x" * 80 for i in range(15))
    chunks = chunk_text(resume)
    # the query sits on the last chunk, so Chroma ranks the last chunks first
    monkeypatch.setattr(
        VectorDB, "get_embeddings", lambda self, texts: [[float(i), 0.0] for i in range(len(chunks))] + [[99.0, 0.0]]
    )

    context = VectorDB().retrieve(resume, "jd", 0.5)

    picked = chunks[len(chunks) // 2 :]
    assert context == "\n".join(picked)  # resume order, not similarity order
