"""Tests for prompts module."""

from stability.prompts import build_openai_prompt


def test_build_openai_prompt_with_conversation(sample_prompt_messages):
    """Test prompt building with conversation included."""
    prompt = build_openai_prompt(sample_prompt_messages, use_conversation=True)

    assert isinstance(prompt, str)
    assert "<CONVERSATION>" in prompt
    assert "SEEKER:" in prompt
    assert "RECOMMENDER:" in prompt


def test_build_openai_prompt_without_conversation(sample_prompt_messages):
    """Test prompt building with conversation removed."""
    prompt = build_openai_prompt(sample_prompt_messages, use_conversation=False)

    assert isinstance(prompt, str)
    assert "<CONVERSATION>" not in prompt
    assert "</CONVERSATION>" not in prompt
    assert "SEEKER:" not in prompt


def test_build_openai_prompt_combines_messages(sample_prompt_messages):
    """Test that messages are properly combined."""
    prompt = build_openai_prompt(sample_prompt_messages, use_conversation=True)

    # Check that both message contents are in the prompt
    assert "You are a movie recommendation system" in prompt
    assert "<CANDIDATES>" in prompt


def test_build_openai_prompt_cleans_newlines(sample_prompt_messages):
    """Test that extra newlines are cleaned up."""
    prompt = build_openai_prompt(sample_prompt_messages, use_conversation=False)

    # Should not have excessive newlines
    assert "\n\n\n\n" not in prompt
    assert prompt.count("\n\n") <= prompt.count("\n") // 2


def test_build_openai_prompt_empty_messages():
    """Test with empty messages list."""
    prompt = build_openai_prompt([], use_conversation=True)

    assert isinstance(prompt, str)
    assert prompt == ""


def test_build_openai_prompt_single_message():
    """Test with single message."""
    messages = [{"role": "user", "content": "Hello world"}]
    prompt = build_openai_prompt(messages, use_conversation=True)

    assert prompt == "Hello world"


def test_build_openai_prompt_rules_replacement():
    """Test that conversation-based rules are replaced correctly."""
    prompt = build_openai_prompt(
        [
            {
                "role": "system",
                "content": (
                    "RULES:\n"
                    " - Use the conversation tagged with <CONVERSATION> between the SEEKER (user) and RECOMMENDER (system) as context.\n"
                    " - Rank movies by relevance based on the conversation and user preferences.\n"
                ),
            },
        ],
        use_conversation=False,
    )

    # The exact replacement happens when both lines are present
    assert "between the SEEKER (user) and RECOMMENDER (system) as context" not in prompt
    assert "based on user preferences" in prompt


def test_build_openai_prompt_preserves_candidates():
    """Test that candidate list is preserved."""
    prompt = build_openai_prompt(
        [
            {
                "role": "user",
                "content": "<CANDIDATES>\n1. Movie A\n2. Movie B\n</CANDIDATES>",
            },
        ],
        use_conversation=True,
    )

    assert "<CANDIDATES>" in prompt
    assert "1. Movie A" in prompt
    assert "2. Movie B" in prompt


def test_build_openai_prompt_with_candidates():
    """Test that candidates block is preserved when use_candidates=True."""
    messages = [
        {
            "role": "system",
            "content": (
                "RULES:\n"
                " - Use the exact movie titles from the provided list of options tagged with <CANDIDATES>.\n"
                " - Do NOT recommend movies not in the provided list.\n"
            ),
        },
        {
            "role": "user",
            "content": "<CANDIDATES>\nMovie A\nMovie B\nMovie C\n</CANDIDATES>",
        },
    ]
    prompt = build_openai_prompt(messages, use_candidates=True)

    assert "<CANDIDATES>" in prompt
    assert "Movie A" in prompt
    assert "Do NOT recommend movies not in the provided list" in prompt


def test_build_openai_prompt_without_candidates():
    """Test that candidates block is stripped when use_candidates=False."""
    messages = [
        {
            "role": "system",
            "content": (
                "RULES:\n"
                " - Use the exact movie titles from the provided list of options tagged with <CANDIDATES>.\n"
                " - Do NOT recommend movies not in the provided list.\n"
            ),
        },
        {
            "role": "user",
            "content": "<CANDIDATES>\nMovie A\nMovie B\nMovie C\n</CANDIDATES>",
        },
    ]
    prompt = build_openai_prompt(messages, use_candidates=False)

    assert "<CANDIDATES>" not in prompt
    assert "</CANDIDATES>" not in prompt
    assert "Movie A" not in prompt
    assert "Do NOT recommend movies not in the provided list" not in prompt
