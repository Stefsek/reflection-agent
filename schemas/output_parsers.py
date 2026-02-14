"""Pydantic schemas for structured outputs from generation and reflection nodes.

This module defines the output schemas used by the reflection agent to ensure
consistent structured responses from the LLM.
"""

from pydantic import BaseModel, Field
from typing import List


class GenerationOutput(BaseModel):
    """Schema for generation node output.

    Attributes:
        generated_prompt: The generated prompt text based on the user's request.
        reasoning: Explanation of design decisions and why the prompt was
            crafted this way.
    """

    generated_prompt: str = Field(
        description="The generated prompt based on the user's request"
    )
    reasoning: str = Field(
        description="Explanation of why this prompt was crafted this way"
    )


class ReflectionOutput(BaseModel):
    """Schema for reflection node output.

    Attributes:
        critique: Critical analysis identifying weaknesses and areas for
            improvement in the generated prompt.
        suggestions: List of specific, actionable suggestions for improving
            the prompt quality.
    """

    critique: str = Field(
        description="Critical analysis of the generated prompt, identifying weaknesses"
    )
    suggestions: List[str] = Field(
        description="Specific suggestions for improving the prompt"
    )
