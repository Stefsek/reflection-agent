"""Core module for the Reflection Agent implementation.

This module provides the main ReflectionAgent class that implements a
LangGraph-based reflection pattern for iterative prompt generation and
refinement through self-critique cycles.

"""

from .reflection_agent import ReflectionAgent

__all__ = [
    "ReflectionAgent"
]