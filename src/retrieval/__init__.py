"""Few-shot example retrieval for T2G (English text → ASL gloss).

Retrieves similar ``(text, gloss)`` training examples so that GRPO/eval
prompts can be augmented with few-shot demonstrations drawn from the
TRAIN split.
"""

from .example_retriever import ExampleRetriever, RetrievedExample, normalize_text

__all__ = ["ExampleRetriever", "RetrievedExample", "normalize_text"]
