"""Generation utilities for LLM-based movie recommendation ranking.

This module contains functions extracted from notebooks 02_run_finetuning.ipynb
and 03_evaluation_study.ipynb to provide reusable utilities for:
- Detecting assistant separator tokens in chat templates
- Formatting conversations with chat templates
- Extracting predicted items from LLM responses

These functions support the two-stage recommendation pipeline where LLMs rank
candidate movies based on conversation context and user preferences.
"""

from __future__ import annotations

import re
from typing import Any

from stability.utils import canonicalize


def detect_assistant_separator(
    tokenizer: Any,  # noqa: ANN401
    supports_system_role: bool = True,
) -> str:
    r"""Return the separator string for the assistant role based on the chat template.

    This function detects the token sequence that separates the user/system portion
    of a conversation from the assistant's response. It supports common chat template
    formats including LLaMA 3, Qwen, Phi-3, Gemma, and Mistral models.

    Args:
        tokenizer: A tokenizer with chat template support (e.g., from transformers).
            Must have an `apply_chat_template` method that accepts a list of message
            dicts and returns a formatted string.
        supports_system_role: Whether the model supports a system role in its chat
            template. Set to False for models like Gemma that don't map "system"
            role properly. Defaults to True.

    Returns:
        The assistant separator string that can be used to split prompts from
        expected completions. For example:
        - LLaMA 3: "<|start_header_id|>assistant<|end_header_id|>\n\n"
        - Qwen: "<|im_start|>assistant\n"
        - Phi-3: "<|assistant|>\n"
        - Gemma: "model\n"
        - Mistral: "[/INST]"

    Raises:
        ValueError: If the assistant separator could not be automatically detected
            from the chat template. The error message includes the formatted test
            output for manual inspection.

    Notes:
        The function uses a test conversation with system, user, and assistant
        messages to detect the separator pattern. The detection order prioritizes
        more specific patterns (e.g., full LLaMA 3 header) over generic ones.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained(
        ...     "meta-llama/Meta-Llama-3-8B-Instruct"
        ... )
        >>> sep = detect_assistant_separator(tokenizer)
        >>> sep
        '<|start_header_id|>assistant<|end_header_id|>\n\n'

    """
    test_messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "test"},
    ]
    if supports_system_role:
        test_messages.insert(0, {"role": "system", "content": "test"})
    formatted = tokenizer.apply_chat_template(test_messages, tokenize=False)

    # Common patterns for different models
    patterns = {
        "llama3": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "qwen": "<|im_start|>assistant\n",
        "phi3": "<|assistant|>\n",
        "gemma": "model\n",
        "mistral": "[/INST]",  # Mistral uses [INST] and [/INST]
    }

    # Find whether the pattern exists in the formatted text
    for pattern in patterns.values():
        if pattern in formatted:
            return pattern

    # Fallback, look for the assistant marker
    if "<|start_header_id|>assistant<|end_header_id|>" in formatted:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if "<|im_start|>assistant" in formatted:
        return "<|im_start|>assistant\n"
    if "<|assistant|>" in formatted:
        return "<|assistant|>\n"
    if "[/INST]" in formatted:
        return "[/INST]"

    # If nothing found, raise with formatted text for manual inspection
    msg = (
        "Could not automatically detect the assistant separator.\n"
        f"Formatted test messages:\n{formatted}\n"
        "Please inspect the above output to determine the correct separator."
    )
    raise ValueError(msg)


def format_with_chat_template(
    batch: dict[str, list[dict[str, str]]],
    tokenizer: Any,  # noqa: ANN401
) -> dict[str, list[str]]:
    """Return the full chat prompt as a single string for each message in the batch.

    This function applies the tokenizer's chat template to format conversations
    into the expected input format for the model. It is designed to work with
    HuggingFace datasets' map function in batched mode.

    Args:
        batch: A batch from the dataset, containing a "messages" key with a list
            of message lists. Each message list contains dicts with "role" and
            "content" keys (e.g., [{"role": "user", "content": "Hello"}]).
        tokenizer: A tokenizer with chat template support (e.g., from transformers).
            Must have an `apply_chat_template` method.

    Returns:
        A dictionary with a "text" key containing a list of formatted chat prompts,
        one for each message list in the batch.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> from functools import partial
        >>> tokenizer = AutoTokenizer.from_pretrained(
        ...     "meta-llama/Meta-Llama-3-8B-Instruct"
        ... )
        >>> batch = {
        ...     "messages": [
        ...         [
        ...             {"role": "user", "content": "Hello"},
        ...             {"role": "assistant", "content": "Hi!"},
        ...         ]
        ...     ]
        ... }
        >>> result = format_with_chat_template(batch, tokenizer)
        >>> "text" in result
        True

    Notes:
        This function is typically used with `dataset.map()`:

        >>> dataset = dataset.map(
        ...     partial(format_with_chat_template, tokenizer=tokenizer),
        ...     batched=True,
        ... )

    """
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        for messages in batch["messages"]
    ]
    return {"text": texts}


def extract_pred_items(response: str, max_items: int = 10) -> list[str]:
    r"""Extract predicted items (movie recommendations) from raw LLM response text.

    Parses the response text to extract a list of movie recommendations.
    Handles common formatting patterns including numbered lists, bullet points,
    and plain text lines. Deduplicates results using canonical string matching.

    Args:
        response: The raw text response from the language model, typically
            containing a numbered or bulleted list of movie recommendations.
        max_items: Maximum number of items to extract. Defaults to 10.

    Returns:
        A list of extracted item strings (movie titles), deduplicated and
        limited to `max_items`. Returns an empty list if no valid items
        are found.

    Examples:
        >>> response = '''
        ... 1. The Shawshank Redemption
        ... 2. The Godfather
        ... 3. Pulp Fiction
        ... '''
        >>> extract_pred_items(response)
        ['The Shawshank Redemption', 'The Godfather', 'Pulp Fiction']

        >>> response = "- Movie A\\n- Movie B\\n- Movie A"  # Duplicate
        >>> extract_pred_items(response)
        ['Movie A', 'Movie B']

    Notes:
        - Lines shorter than 2 characters or longer than 120 are filtered out
        - Common list prefixes (numbers, dashes, bullets) are stripped
        - Deduplication uses canonicalize() for case-insensitive matching
        - Original casing is preserved in the output

    """
    response = response.strip()
    lines = [
        re.sub(r"^\d+[.)]\s*", "", ln).strip(" -*•\t")
        for ln in response.splitlines()
        if ln.strip()
    ]

    items: list[str] = []
    for ln in lines:
        # Filter by reasonable length (movie title constraints)
        if ln and 2 <= len(ln) <= 120:
            items.append(ln)
        if len(items) >= max_items:
            break

    # Deduplicate using canonical form
    seen: set[str | None] = set()
    dedup: list[str] = []
    for it in items:
        it_canon = canonicalize(it)
        if it_canon and it_canon not in seen:
            seen.add(it_canon)
            dedup.append(it)

    return dedup[:max_items]
