import uvicorn
import shutil
import os
import uuid 
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from parser import parse_resume 
from ai_logic import analyze_match
from vector_db import VectorDB
from schemas import MatchResponse
app = FastAPI(title="AI Resume Matcher")

db = VectorDB()

@app.post("/match-pdf",response_model=MatchResponse) 
async def match_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    temp_filename = f"temp_{file.filename}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
       
        resume_text = parse_resume(temp_filename)
        if not resume_text:
             raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")

        resume_id = str(uuid.uuid4())  
        db.add_resume(resume_text, resume_id)
        relevant_context = db.query_resume(job_description, n_results=5)
        analysis_result = analyze_match(relevant_context, job_description)

        return MatchResponse(
            status="sucess",
            parsed_text_preview = relevant_context[:500] + "...",
            analysis = analysis_result
        )
    
    except Exception as e:
        print(f"Error: {e}") 
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/")
def check():
    return {"status": "Running Success"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)