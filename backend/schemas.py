from pydantic import BaseModel,Field
from typing import List,Literal

class AnalysisResult(BaseModel):
    match_percentage:int = Field(ge=0,le=100,description="A score between 0 and 100 representing the match quality")
    missing_keywords: List[str] = Field(default_factory=list,description="List of critical skills or keywords missing from resume ")
    recommendation: Literal["Hire","No Hire","Interview","Review"] = Field(description="The final recommendation based on the analysis")
    analysis_summary:str = Field(description="A brief professional summary explaining the score")
class MatchResponse(BaseModel):
    parsed_text_preview: str
    analysis:AnalysisResult
