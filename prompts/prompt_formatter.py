"""Formatter utilities for structuring agent outputs into display text.

This module provides formatting functions that convert structured model outputs
into human-readable text using predefined templates.
"""

from .prompt_templates import GENERATION_TEMPLATE, REFLECTION_FEEDBACK_TEMPLATE


class PromptFormatter:
    """Formatter for agent node outputs.

    Provides static methods to format generation and reflection outputs
    into structured text suitable for display or message passing.
    """

    @staticmethod
    def format_generation_output(generated_prompt: str, reasoning: str) -> str:
        """Formats generation output into structured text.

        Args:
            generated_prompt: The generated prompt text.
            reasoning: Explanation of the design decisions behind the prompt.

        Returns:
            Formatted string with labeled sections for prompt and reasoning.
        """
        return GENERATION_TEMPLATE.format(
            generated_prompt=generated_prompt,
            reasoning=reasoning
        )

    @staticmethod
    def format_reflection_output(critique: str, suggestions: list[str]) -> str:
        """Formats reflection output into structured feedback text.

        Args:
            critique: Critical analysis of the generated prompt.
            suggestions: List of improvement suggestions.

        Returns:
            Formatted string with labeled sections for critique and numbered
            suggestions.
        """
        # Format suggestions as numbered list
        suggestions_text = "\n".join(
            f"{i}. {suggestion}" for i, suggestion in enumerate(suggestions, 1)
        )

        return REFLECTION_FEEDBACK_TEMPLATE.format(
            critique=critique,
            suggestions=suggestions_text
        )