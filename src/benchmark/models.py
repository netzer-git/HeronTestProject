from typing import List, Literal
from pydantic import BaseModel

class EvaluationCase(BaseModel):
    case_id: int
    level: Literal["easy", "medium", "hard"]
    keywords: List[Literal["creative_writing", "math", "coding-cyber"]]
    score: float # 0.0 to 1.0

class MockResult(BaseModel):
    cases: List[EvaluationCase]
    model_name: str
    run_num: int

class AgentReport(BaseModel):
    model_name: str
    average_score: float
    safety_score: float
    anomalies_detected: bool
    summary: str
