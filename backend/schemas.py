from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalysisResult(BaseModel):
    # frozen (and a tuple below) because ai_logic.FALLBACK_RESULT is one shared instance
    model_config = ConfigDict(frozen=True)

    match_percentage: int = Field(ge=0, le=100, description="A score between 0 and 100 representing the match quality")
    missing_keywords: tuple[str, ...] = Field(
        default_factory=tuple, description="List of critical skills or keywords missing from resume "
    )
    recommendation: Literal["Hire", "No Hire", "Interview", "Review"] = Field(
        description="The final recommendation based on the analysis"
    )
    analysis_summary: str = Field(description="A brief professional summary explaining the score")


class Requirement(BaseModel):
    requirement: str
    status: Literal["met", "partial", "missing"]
    evidence: str = ""


class Checklist(BaseModel):
    """What the v3 prompt asks for: a verdict per core requirement. The score is computed from it in code."""

    requirements: list[Requirement] = Field(min_length=1)
    analysis_summary: str


class MatchResponse(BaseModel):
    parsed_text_preview: str
    analysis: AnalysisResult
