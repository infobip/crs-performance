"""Tests for generation module."""

import pytest

from stability.generation import (
    detect_assistant_separator,
    extract_pred_items,
    format_with_chat_template,
)


def _format_llama3(messages: list[dict[str, str]]) -> str:
    """Format messages for LLaMA 3 template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    return "".join(parts)


def _format_qwen(messages: list[dict[str, str]]) -> str:
    """Format messages for Qwen template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def _format_phi3(messages: list[dict[str, str]]) -> str:
    """Format messages for Phi-3 template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|>\n{content}<|end|>\n")
    return "".join(parts)


def _format_gemma(messages: list[dict[str, str]]) -> str:
    """Format messages for Gemma template."""
    parts = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        content = msg["content"]
        parts.append(f"{role}\n{content}\n")
    return "".join(parts)


def _format_mistral(messages: list[dict[str, str]]) -> str:
    """Format messages for Mistral template."""
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"[INST]{msg['content']}[/INST]")
        else:
            parts.append(msg["content"])
    return "".join(parts)


_TEMPLATE_FORMATTERS = {
    "llama3": _format_llama3,
    "qwen": _format_qwen,
    "phi3": _format_phi3,
    "gemma": _format_gemma,
    "mistral": _format_mistral,
}


class MockTokenizer:
    """Mock tokenizer for testing detect_assistant_separator."""

    def __init__(self, template_type: str = "llama3"):
        self.template_type = template_type

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,  # noqa: ARG002
        add_generation_prompt: bool = True,  # noqa: ARG002
    ) -> str:
        """Apply mock chat template based on template type."""
        formatter = _TEMPLATE_FORMATTERS.get(self.template_type)
        if formatter:
            return formatter(messages)
        if self.template_type == "unknown":
            return f"###USER: {messages[0]['content']} ###ASSISTANT: {messages[-1]['content']}"
        return str(messages)


class TestDetectAssistantSeparator:
    """Tests for detect_assistant_separator function."""

    def test_detect_llama3_separator(self):
        """Test detection of LLaMA 3 assistant separator."""
        tokenizer = MockTokenizer("llama3")
        sep = detect_assistant_separator(tokenizer)
        assert sep == "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def test_detect_qwen_separator(self):
        """Test detection of Qwen assistant separator."""
        tokenizer = MockTokenizer("qwen")
        sep = detect_assistant_separator(tokenizer)
        assert sep == "<|im_start|>assistant\n"

    def test_detect_phi3_separator(self):
        """Test detection of Phi-3 assistant separator."""
        tokenizer = MockTokenizer("phi3")
        sep = detect_assistant_separator(tokenizer)
        assert sep == "<|assistant|>\n"

    def test_detect_gemma_separator(self):
        """Test detection of Gemma assistant separator."""
        tokenizer = MockTokenizer("gemma")
        sep = detect_assistant_separator(tokenizer)
        assert sep == "model\n"

    def test_detect_mistral_separator(self):
        """Test detection of Mistral assistant separator."""
        tokenizer = MockTokenizer("mistral")
        sep = detect_assistant_separator(tokenizer)
        assert sep == "[/INST]"

    def test_detect_gemma_separator_without_system_role(self):
        """Test Gemma detection with supports_system_role=False."""
        tokenizer = MockTokenizer("gemma")
        sep = detect_assistant_separator(tokenizer, supports_system_role=False)
        assert sep == "model\n"

    def test_detect_unknown_raises_error(self):
        """Test that unknown template type raises ValueError."""
        tokenizer = MockTokenizer("unknown")
        with pytest.raises(ValueError, match="Could not automatically detect"):
            detect_assistant_separator(tokenizer)


class TestFormatWithChatTemplate:
    """Tests for format_with_chat_template function."""

    def test_format_single_message_batch(self):
        """Test formatting a batch with a single conversation."""
        tokenizer = MockTokenizer("llama3")
        batch = {
            "messages": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            ]
        }
        result = format_with_chat_template(batch, tokenizer)
        assert "text" in result
        assert len(result["text"]) == 1
        assert "Hello" in result["text"][0]
        assert "Hi there!" in result["text"][0]

    def test_format_multiple_message_batch(self):
        """Test formatting a batch with multiple conversations."""
        tokenizer = MockTokenizer("llama3")
        batch = {
            "messages": [
                [{"role": "user", "content": "First"}],
                [{"role": "user", "content": "Second"}],
                [{"role": "user", "content": "Third"}],
            ]
        }
        result = format_with_chat_template(batch, tokenizer)
        assert "text" in result
        assert len(result["text"]) == 3
        assert "First" in result["text"][0]
        assert "Second" in result["text"][1]
        assert "Third" in result["text"][2]

    def test_format_empty_batch(self):
        """Test formatting an empty batch."""
        tokenizer = MockTokenizer("llama3")
        batch = {"messages": []}
        result = format_with_chat_template(batch, tokenizer)
        assert result == {"text": []}

    def test_format_system_user_assistant_messages(self):
        """Test formatting a complete conversation with system, user, assistant."""
        tokenizer = MockTokenizer("llama3")
        batch = {
            "messages": [
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            ]
        }
        result = format_with_chat_template(batch, tokenizer)
        assert "text" in result
        text = result["text"][0]
        assert "You are helpful." in text
        assert "What is 2+2?" in text
        assert "4" in text

    def test_format_with_different_tokenizers(self):
        """Test that different tokenizers produce different formats."""
        batch = {"messages": [[{"role": "user", "content": "test"}]]}
        llama_result = format_with_chat_template(batch, MockTokenizer("llama3"))
        qwen_result = format_with_chat_template(batch, MockTokenizer("qwen"))

        assert llama_result["text"][0] != qwen_result["text"][0]
        assert "<|start_header_id|>" in llama_result["text"][0]
        assert "<|im_start|>" in qwen_result["text"][0]


class TestExtractPredItems:
    """Tests for extract_pred_items function."""

    def test_extract_numbered_list(self):
        """Test extraction from numbered list — numbered prefixes are stripped."""
        response = """1. The Shawshank Redemption
2. The Godfather
3. Pulp Fiction"""
        result = extract_pred_items(response)
        assert result == [
            "The Shawshank Redemption",
            "The Godfather",
            "Pulp Fiction",
        ]

    def test_extract_numbered_list_with_colon_in_title(self):
        """Test that '10. 2001: A Space Odyssey' strips prefix but keeps colon."""
        response = "10. 2001: A Space Odyssey"
        result = extract_pred_items(response)
        assert result == ["2001: A Space Odyssey"]

    def test_extract_parenthesized_numbering(self):
        """Test extraction from '1) Movie' style numbering."""
        response = """1) The Matrix
2) Inception
3) Interstellar"""
        result = extract_pred_items(response)
        assert result == ["The Matrix", "Inception", "Interstellar"]

    def test_extract_no_false_strip_on_number_in_title(self):
        """Test that 'Movie 2' is NOT stripped (no numbered prefix)."""
        response = "Movie 2"
        result = extract_pred_items(response)
        assert result == ["Movie 2"]

    def test_extract_mixed_numbered_and_bullet(self):
        """Test extraction from mixed numbered + bullet formats."""
        response = """1. The Matrix
- Inception
2. Interstellar
* Pulp Fiction"""
        result = extract_pred_items(response)
        assert result == ["The Matrix", "Inception", "Interstellar", "Pulp Fiction"]

    def test_extract_bullet_list(self):
        """Test extraction from bullet point list."""
        response = """- Movie A
- Movie B
- Movie C"""
        result = extract_pred_items(response)
        assert result == ["Movie A", "Movie B", "Movie C"]

    def test_extract_with_deduplication(self):
        """Test that duplicates are removed."""
        response = """- Movie A
- Movie B
- Movie A
- Movie C"""
        result = extract_pred_items(response)
        # Only unique items should be returned
        assert result == ["Movie A", "Movie B", "Movie C"]

    def test_extract_case_insensitive_dedup(self):
        """Test case-insensitive deduplication."""
        response = """- The Matrix
- THE MATRIX
- the matrix"""
        result = extract_pred_items(response)
        assert len(result) == 1
        assert result[0] == "The Matrix"  # First occurrence preserved

    def test_extract_max_items(self):
        """Test max_items parameter limits output."""
        response = """1. Movie 1
2. Movie 2
3. Movie 3
4. Movie 4
5. Movie 5"""
        result = extract_pred_items(response, max_items=3)
        assert len(result) == 3

    def test_extract_empty_response(self):
        """Test extraction from empty response."""
        result = extract_pred_items("")
        assert result == []

    def test_extract_whitespace_only(self):
        """Test extraction from whitespace-only response."""
        result = extract_pred_items("   \n\n   \t  ")
        assert result == []

    def test_extract_filters_short_lines(self):
        """Test that very short lines (< 2 chars) are filtered."""
        response = """A
BB
CCC"""
        result = extract_pred_items(response)
        # 'A' should be filtered out (too short)
        assert "A" not in result
        assert "BB" in result
        assert "CCC" in result

    def test_extract_filters_long_lines(self):
        """Test that very long lines (> 120 chars) are filtered."""
        short_line = "Normal Movie Title"
        long_line = "A" * 121  # Too long
        response = f"{short_line}\n{long_line}"
        result = extract_pred_items(response)
        assert short_line in result
        assert long_line not in result

    def test_extract_strips_bullet_markers(self):
        """Test that bullet markers are stripped."""
        response = """• Movie with bullet
- Movie with dash
* This should work too"""
        result = extract_pred_items(response)
        # Bullets and dashes should be stripped
        assert "Movie with bullet" in result
        assert "Movie with dash" in result

    def test_extract_handles_mixed_formatting(self):
        """Test extraction from mixed formatting."""
        response = """Here are my recommendations:
1. The Matrix (1999)
- Inception
* Interstellar

These are great movies!"""
        result = extract_pred_items(response)
        # Should extract movie lines and filter out non-movie lines
        assert len(result) > 0
        # First line "Here are my recommendations:" should be included as valid text

    def test_extract_preserves_original_casing(self):
        """Test that original casing is preserved."""
        response = "- The MATRIX"
        result = extract_pred_items(response)
        assert result[0] == "The MATRIX"

    def test_extract_handles_tabs_and_spaces(self):
        """Test handling of tabs and extra spaces."""
        response = """	 - Movie A
   -  Movie B	"""
        result = extract_pred_items(response)
        assert "Movie A" in result
        assert "Movie B" in result

    def test_extract_default_max_items(self):
        """Test default max_items is 10."""
        response = "\n".join([f"Movie {i}" for i in range(15)])
        result = extract_pred_items(response)
        assert len(result) == 10  # Default max_items

    def test_extract_with_special_characters(self):
        """Test extraction with special characters in titles."""
        response = """- Amélie
- Léon: The Professional
- Crouching Tiger, Hidden Dragon"""
        result = extract_pred_items(response)
        assert len(result) == 3
        assert "Amélie" in result
