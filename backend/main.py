import functools
import os
import shutil
import tempfile

import uvicorn
from ai_logic import analyze_match, with_score_recommendation
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from parser import parse_resume
from schemas import MatchResponse
from vector_db import VectorDB

# "rag": send the LLM only the resume chunks closest to the JD. "full": send the whole resume.
CONTEXT_MODES = ("rag", "full")
CONTEXT_MODE = os.getenv("CONTEXT_MODE", "rag")
if CONTEXT_MODE not in CONTEXT_MODES:
    raise ValueError(f"CONTEXT_MODE must be one of {CONTEXT_MODES}, got {CONTEXT_MODE!r}")
# Share of a resume's chunks sent to the LLM in rag mode. A fraction (not a fixed top-k) so a
# 2-page resume isn't cut harder than a 1-page one.
RETRIEVE_FRACTION = 0.5
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # resumes are well under this; caps disk use per request

app = FastAPI(title="AI Resume Matcher")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    # Must run before FastAPI parses the multipart body, which spools the whole upload to disk.
    # Known gap: chunked requests without Content-Length aren't capped; put a reverse proxy limit in front if exposed publicly.
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"detail": "PDF is larger than 5 MB."})
    return await call_next(request)


@functools.cache
def get_db() -> VectorDB:
    return VectorDB()


def get_context(resume_text: str, job_description: str, mode: str) -> str:
    if mode not in CONTEXT_MODES:
        raise ValueError(f"mode must be one of {CONTEXT_MODES}, got {mode!r}")
    if mode == "full":
        return resume_text

    context = get_db().retrieve(resume_text, job_description, RETRIEVE_FRACTION)
    return context or resume_text


@app.post("/match-pdf", response_model=MatchResponse)
def match_resume(file: UploadFile = File(...), job_description: str = Form(...)):
    # never build paths from the client's filename
    fd, temp_filename = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        resume_text = parse_resume(temp_filename)

        if not resume_text or not resume_text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")

        relevant_context = get_context(resume_text, job_description, CONTEXT_MODE)
        analysis_result = with_score_recommendation(analyze_match(relevant_context, job_description))

        return MatchResponse(parsed_text_preview=relevant_context, analysis=analysis_result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e!r}")  # the details stay in the server log: they can include paths and internal URLs
        raise HTTPException(status_code=500, detail="Analysis failed. Check that Ollama is running.") from e

    finally:
        os.remove(temp_filename)


@app.get("/")
def check():
    return {"status": "Running Success"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
