"""Utility functions for token usage tracking and aggregation.

This module provides helper functions for working with LangChain usage metadata
callbacks to track and aggregate token usage across multiple model invocations.
"""

from typing import Any, Dict


class TokenUsageUtils:
    """Utility class for token usage calculations.

    Provides static methods for aggregating token usage metrics from LangChain
    usage metadata callbacks.
    """

    @staticmethod
    def sum_tokens(usage_metadata: Dict[str, Any], token_type: str) -> int:
        """Sums token counts of a specific type across all models.

        Aggregates token usage from multiple model invocations or providers
        in the usage metadata dictionary.

        Args:
            usage_metadata: Dictionary mapping model identifiers to their usage
                statistics. Each value should be a dict containing token counts.
            token_type: The type of tokens to sum (e.g., "input_tokens",
                "output_tokens", "total_tokens").

        Returns:
            Total count of tokens for the specified type across all models.
            Returns 0 for any model that doesn't have the specified token_type.
        """
        return sum(
            model_usage.get(token_type, 0) for model_usage in usage_metadata.values()
        )
