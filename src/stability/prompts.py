"""Prompt building utilities for OpenAI completions API."""

import re


def build_openai_prompt(
    messages: list[dict[str, str]],
    use_conversation: bool = True,
    use_candidates: bool = True,
) -> str:
    """Build prompt string from ChatML messages for OpenAI completions API.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        use_conversation: Whether to keep the <CONVERSATION> block in the prompt.
        use_candidates: Whether to keep the <CANDIDATES> block in the prompt.

    Returns:
        Formatted prompt string combining system and user messages.

    """
    # Combine system and user messages
    prompt_parts = []
    for msg in messages:
        prompt_parts.append(msg["content"])
    prompt = "\n\n".join(prompt_parts)

    # Remove conversation block if requested
    if not use_conversation:
        prompt = re.sub(
            r"\n?<CONVERSATION>.*?</CONVERSATION>\n?",
            "\n",
            prompt,
            flags=re.DOTALL,
        )
        # Also remove conversation-related rules
        old_rules = (
            " - Use the conversation tagged with <CONVERSATION> between the SEEKER (user) and RECOMMENDER (system) as context.\n"
            " - Rank movies by relevance based on the conversation and user preferences.\n"
        )
        new_rules = " - Rank movies by relevance based on user preferences.\n"
        prompt = prompt.replace(old_rules, new_rules)
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    # Remove candidates block if requested
    if not use_candidates:
        prompt = re.sub(
            r"\n?<CANDIDATES>.*?</CANDIDATES>\n?",
            "\n",
            prompt,
            flags=re.DOTALL,
        )
        # Also remove candidate-related rules
        old_rules = (
            " - Use the exact movie titles from the provided list of options tagged with <CANDIDATES>.\n"
            " - Do NOT recommend movies not in the provided list.\n"
        )
        prompt = prompt.replace(old_rules, "")
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    return prompt
