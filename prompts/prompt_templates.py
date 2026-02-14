"""Prompt templates for formatting agent outputs.

This module defines LangChain PromptTemplate instances used to structure
generation and reflection outputs into consistent, readable text formats.
"""

from langchain_core.prompts import PromptTemplate


GENERATION_TEMPLATE = PromptTemplate(
    input_variables=["generated_prompt", "reasoning"],
    template=
    """
    **Generated Prompt:**
    {generated_prompt}

    **Reasoning:**
    {reasoning}
    """
)

REFLECTION_FEEDBACK_TEMPLATE = PromptTemplate(
    input_variables=["critique", "suggestions"],
    template=
    """
    Based on the evaluation of your previous prompt, here is the feedback:

    **Critique:**
    {critique}

    **Suggestions for Improvement:**
    {suggestions}

    Please revise the prompt by addressing the critique and incorporating these suggestions to create a higher quality result."""
)
