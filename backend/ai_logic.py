
import ollama 
import json
from schemas import AnalysisResult

def llama_query(prompt: str):
    
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user','content':prompt},
            ],format='json')
        return response['message']['content']
    except Exception as e:
        print(f"Ollama Error:{e}")
        return None
        



def analyze_match(resume_description:str , job_description:str)-> AnalysisResult:
    prompt = f"""
     You are an expert Technical Recruiter. Compare the following Resume and Job Description.

    RESUME:
    {resume_description} 

    JOB DESCRIPTION:
    {job_description}
   
  
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

    fallback_result = AnalysisResult(
        match_percentage = 0,
        missing_keywords = [],
        recommendation = "Review",
        analysis_summary = "Analysis failed due to errror"
    )

    if not response_text:
        return fallback_result


    try:
        data = json.loads(response_text)
        return AnalysisResult(**data)
    
    except (json.JSONDecodeError,Exception) as e:
        print(f"Parsing Error:{e}")
        return fallback_result

