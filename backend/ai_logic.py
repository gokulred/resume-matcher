
import ollama 
import json

def illama_query(prompt: str):
    
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user','content':prompt},
            ],format='json')
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"
        



def analyze_match(resume_description:str , job_description:str):
    prompt = f"""
     You are an expert Technical Recruiter. Compare the following Resume and Job Description.

    RESUME:
    {resume_description} 

    JOB DESCRIPTION:
    {job_description}
   
    IMPORTANT OUTPUT FORMAT:
    Analyze the match and return the result strictly as a JSON object
    REQUIRED JSON STRUCTURE:
    {{
        "match_percentage":(integer between 0 and 100),
        "missing_keywords":[(list of strings)],
        "recommendation":(string:"Hire","No Hire", "Interview"),
        "analysis_summary":(string:brief explanation of the score)
    }}
    """
    
    response_text = illama_query(prompt)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:

        return {
            "match_percentage": 0,
            "missing_keywords": [],
            "recommendation": "Error",
            "analysis_summary": "Raw output: " + response_text
        }

