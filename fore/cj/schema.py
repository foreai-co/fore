"""Schema for the Critical Journeys API."""
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class State(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    RESOURCES_EXCEEDED = "resources_exceeded"


class TestStep(BaseModel):
    step: str
    expectation: Optional[str] = None


class TestCaseRequest(BaseModel):
    website: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    test_steps: List[TestStep]
    credentials: Optional[Dict[str, Union[str, int, float]]] = None
    programming_language: str = "typescript"


class TestGenerationStep(BaseModel):
    planner_message: str
    generated_code: Optional[str] = None
    screenshot: Optional[str] = None


class TestCase(TestCaseRequest):
    id: Optional[str] = Field(alias="_id", default=None)
    state: State = State.PENDING

    final_script: str = ""
    test_generation_steps: List[TestGenerationStep] = []
