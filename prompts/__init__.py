"""Prompts package for the reflection agent.

This package contains system messages, prompt templates, and formatting
utilities used throughout the reflection agent workflow.
"""

from .system_messages import (
    GENERATION_SYSTEM_MESSAGE,
    REFLECTION_SYSTEM_MESSAGE
)
from .prompt_formatter import PromptFormatter

__all__ = [
    "GENERATION_SYSTEM_MESSAGE",
    "REFLECTION_SYSTEM_MESSAGE",
    "PromptFormatter"
]
