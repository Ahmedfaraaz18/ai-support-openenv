from typing import Dict, List
from pydantic import BaseModel, Field, validator


class Observation(BaseModel):
    ticket_id: int
    message: str
    user_history: List[str]
    current_status: str
    urgency_hint: str


class Action(BaseModel):
    assign_category: str = Field(..., pattern="^(billing|technical|account|other)$")
    set_priority: str = Field(..., pattern="^(low|medium|high)$")
    response: str

    @validator("response")
    def non_empty_response(cls, v: str):
        if not v.strip():
            raise ValueError("Response must not be empty")
        return v


class Reward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    breakdown: Dict[str, float]


class State(BaseModel):
    step_count: int
    ticket_resolved: bool
    total_reward: float
