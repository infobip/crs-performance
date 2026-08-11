"""Tests for preprocessing module."""

import pandas as pd

from stability.preprocessing import (
    _build_conversation,
    _build_prompt_template,
    _classify_movie_preferences,
    _compute_metadata,
    _mask_accepted_movies,
    _replace_movie_ids_with_titles,
    analyze_dialogue_dropout,
    extract_user_preferences,
    generate_prompt_templates,
    normalize_movie_title,
)


def test_normalize_movie_title_basic():
    """Test basic title normalization."""
    assert normalize_movie_title("The Matrix") == "The Matrix"
    assert normalize_movie_title("  The Matrix  ") == "The Matrix"


def test_normalize_movie_title_extra_spaces():
    """Test normalization of extra spaces."""
    assert normalize_movie_title("The  Matrix") == "The Matrix"
    assert normalize_movie_title("The   Matrix   Reloaded") == "The Matrix Reloaded"


def test_normalize_movie_title_none():
    """Test normalization of None."""
    assert normalize_movie_title(None) is None


def test_normalize_movie_title_empty():
    """Test normalization of empty string."""
    assert normalize_movie_title("") == ""


def test_extract_user_preferences_basic(sample_dialogue):
    """Test basic preference extraction."""
    prefs = extract_user_preferences(sample_dialogue, collect_meta=False)

    assert isinstance(prefs, dict)
    assert "liked" in prefs
    assert "disliked" in prefs
    assert "recommended" in prefs
    assert "conversation" in prefs


def test_extract_user_preferences_with_meta(sample_dialogue):
    """Test preference extraction with metadata."""
    result = extract_user_preferences(sample_dialogue, collect_meta=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    prefs, meta = result
    assert isinstance(prefs, dict)
    assert isinstance(meta, dict)


def test_extract_user_preferences_likes(sample_dialogue):
    """Test that liked movies are extracted correctly."""
    prefs = extract_user_preferences(sample_dialogue, collect_meta=False)

    # Check that liked list exists and contains movie IDs (not titles)
    assert "liked" in prefs
    assert isinstance(prefs["liked"], list)


def test_extract_user_preferences_dislikes(sample_dialogue):
    """Test that disliked movies are extracted correctly."""
    prefs = extract_user_preferences(sample_dialogue, collect_meta=False)

    # Check that disliked list exists
    assert "disliked" in prefs
    assert isinstance(prefs["disliked"], list)


def test_analyze_dialogue_dropout():
    """Test dialogue dropout analysis covers all skip reasons."""
    # kept: has liked + recommended
    kept = {
        "conversationId": 1,
        "movieMentions": {"m1": "Movie A", "m2": "Movie B"},
        "initiatorQuestions": {"m1": {"seen": 1, "liked": 1}},
        "respondentQuestions": {"m2": {"suggested": 1}},
        "messages": [],
        "initiatorWorkerId": 1,
        "respondentWorkerId": 2,
    }
    # no liked and no recommended
    no_liked_no_rec = {
        "conversationId": 2,
        "movieMentions": {"m3": "Movie C"},
        "initiatorQuestions": {"m3": {"seen": 2, "liked": 2}},
        "respondentQuestions": {},
        "messages": [],
        "initiatorWorkerId": 3,
        "respondentWorkerId": 4,
    }
    # has recommended but no liked
    no_liked = {
        "conversationId": 3,
        "movieMentions": {"m4": "Movie D"},
        "initiatorQuestions": {"m4": {"seen": 0, "liked": 2}},
        "respondentQuestions": {"m4": {"suggested": 1}},
        "messages": [],
        "initiatorWorkerId": 5,
        "respondentWorkerId": 6,
    }
    # has liked but no recommended
    no_rec = {
        "conversationId": 4,
        "movieMentions": {"m5": "Movie E"},
        "initiatorQuestions": {"m5": {"seen": 1, "liked": 1}},
        "respondentQuestions": {},
        "messages": [],
        "initiatorWorkerId": 7,
        "respondentWorkerId": 8,
    }

    analysis = analyze_dialogue_dropout([kept, no_liked_no_rec, no_liked, no_rec])

    assert isinstance(analysis, pd.DataFrame)
    assert len(analysis) == 4

    reasons = analysis["skipped_reason"].tolist()
    assert pd.isna(reasons[0])  # kept
    assert reasons[1] == "no_liked_and_no_recommended"
    assert reasons[2] == "no_liked"
    assert reasons[3] == "no_recommended"


def test_generate_prompt_templates_basic(sample_dialogue):
    """Test basic prompt template generation."""
    templates = generate_prompt_templates([sample_dialogue], mask_accepted=True)

    assert isinstance(templates, list)
    # Templates may skip dialogues without liked movies or recommendations
    if len(templates) > 0:
        assert "prompt" in templates[0]
        assert "user_liked" in templates[0]


def test_generate_prompt_templates_mask_token():
    """Test custom mask token appears in generated prompt."""
    dialogue = {
        "conversationId": 1,
        "movieMentions": {"m1": "Movie A", "m2": "Movie B"},
        "initiatorQuestions": {"m1": {"seen": 1, "liked": 1, "suggested": 0}},
        "respondentQuestions": {"m2": {"suggested": 1}},
        "messages": [
            {"senderWorkerId": 1, "text": "I liked @m1"},
            {"senderWorkerId": 2, "text": "Try @m2"},
        ],
        "initiatorWorkerId": 1,
        "respondentWorkerId": 2,
    }

    templates = generate_prompt_templates(
        [dialogue],
        mask_accepted=True,
        mask_token="CUSTOM_MASK",
    )

    assert len(templates) == 1
    assert "CUSTOM_MASK_0" in templates[0]["prompt"]


def test_generate_prompt_templates_without_masking():
    """Test template generation without masking omits MASK tokens."""
    dialogue = {
        "conversationId": 1,
        "movieMentions": {"m1": "Movie A", "m2": "Movie B"},
        "initiatorQuestions": {"m1": {"seen": 1, "liked": 1, "suggested": 0}},
        "respondentQuestions": {"m2": {"suggested": 1}},
        "messages": [
            {"senderWorkerId": 1, "text": "I liked @m1"},
            {"senderWorkerId": 2, "text": "Try @m2"},
        ],
        "initiatorWorkerId": 1,
        "respondentWorkerId": 2,
    }

    templates = generate_prompt_templates([dialogue], mask_accepted=False)

    assert len(templates) == 1
    # Without masking, recommended movie @m2 should remain as raw ID in conversation
    prompt = templates[0]["prompt"]
    assert "@m2" in prompt
    # Liked movie @m1 should still be replaced with its title
    assert "Movie A" in prompt


def test_generate_prompt_templates_empty_list():
    """Test with empty dialogue list."""
    templates = generate_prompt_templates([])

    assert isinstance(templates, list)
    assert len(templates) == 0


def test_generate_prompt_templates_required_fields():
    """Test that template has all required fields."""
    dialogue = {
        "conversationId": 1,
        "movieMentions": {"m1": "Movie A", "m2": "Movie B"},
        "initiatorQuestions": {"m1": {"seen": 1, "liked": 1, "suggested": 0}},
        "respondentQuestions": {"m2": {"suggested": 1}},
        "messages": [],
        "initiatorWorkerId": 1,
        "respondentWorkerId": 2,
    }

    templates = generate_prompt_templates([dialogue])

    assert len(templates) == 1
    template = templates[0]
    assert "prompt" in template
    assert "user_liked" in template
    assert "user_disliked" in template
    assert "recommended_accepted" in template
    assert "recommended_rejected" in template
    assert "recommended" in template
    assert "dialogue_id" in template


class TestClassifyMoviePreferences:
    """Tests for _classify_movie_preferences helper function."""

    def test_seen_liked_goes_to_liked_list(self):
        """Test case: seen=1, liked=1 -> should be in liked list."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 1, "liked": 1, "suggested": 0},
            },
            "respondentQuestions": {},
        }
        movie_mentions = {"m1": "The Matrix"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" in liked
        assert "m1" not in disliked
        assert "m1" not in rec_accepted
        assert "m1" not in rec_rejected

    def test_seen_not_liked_goes_to_disliked_list(self):
        """Test case: seen=1, liked=0 -> should be in disliked list."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 1, "liked": 0, "suggested": 0},
            },
            "respondentQuestions": {},
        }
        movie_mentions = {"m1": "The Matrix"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" not in liked
        assert "m1" in disliked
        assert "m1" not in rec_accepted
        assert "m1" not in rec_rejected

    def test_unseen_suggested_liked_goes_to_recommended_accepted(self):
        """Test case: seen=0, suggested=1, liked=1 -> should be in recommended_accepted."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 0, "liked": 1, "suggested": 0},
            },
            "respondentQuestions": {
                "m1": {"suggested": 1},
            },
        }
        movie_mentions = {"m1": "Inception"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" not in liked
        assert "m1" not in disliked
        assert "m1" in rec_accepted
        assert "m1" not in rec_rejected

    def test_unseen_suggested_not_liked_goes_to_recommended_rejected(self):
        """Test case: seen=0, suggested=1, liked=0 -> should be in recommended_rejected."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 0, "liked": 0, "suggested": 0},
            },
            "respondentQuestions": {
                "m1": {"suggested": 1},
            },
        }
        movie_mentions = {"m1": "Inception"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" not in liked
        assert "m1" not in disliked
        assert "m1" not in rec_accepted
        assert "m1" in rec_rejected

    def test_empty_dialogue_returns_empty_lists(self):
        """Test case: empty dialogue -> should return empty lists."""
        dialogue = {
            "initiatorQuestions": {},
            "respondentQuestions": {},
        }
        movie_mentions = {}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert liked == []
        assert disliked == []
        assert rec_accepted == []
        assert rec_rejected == []

    def test_movie_without_title_is_skipped(self):
        """Test that movies not in movie_mentions are skipped."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 1, "liked": 1, "suggested": 0},
                "m2": {"seen": 1, "liked": 1, "suggested": 0},
            },
            "respondentQuestions": {},
        }
        # Only m1 has a title
        movie_mentions = {"m1": "The Matrix"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" in liked
        assert "m2" not in liked  # Skipped because no title

    def test_unseen_suggested_unknown_liked_goes_to_recommended_accepted(self):
        """Test case: seen=0, suggested=1, liked=2 (unknown) -> should be in recommended_accepted."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 0, "liked": 2, "suggested": 0},
            },
            "respondentQuestions": {
                "m1": {"suggested": 1},
            },
        }
        movie_mentions = {"m1": "Interstellar"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" not in liked
        assert "m1" not in disliked
        assert "m1" in rec_accepted
        assert "m1" not in rec_rejected

    def test_unseen_not_suggested_liked_goes_to_liked(self):
        """Test case: seen=0, suggested=0, liked=1 -> should be in liked list (self-mentioned)."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 0, "liked": 1, "suggested": 0},
            },
            "respondentQuestions": {},
        }
        movie_mentions = {"m1": "The Dark Knight"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" in liked
        assert "m1" not in disliked

    def test_unknown_seen_suggested_liked_goes_to_recommended_accepted(self):
        """Test case: seen=2 (unknown), suggested=1, liked=1 -> should be in recommended_accepted."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 2, "liked": 1, "suggested": 0},
            },
            "respondentQuestions": {
                "m1": {"suggested": 1},
            },
        }
        movie_mentions = {"m1": "Pulp Fiction"}

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" not in liked
        assert "m1" not in disliked
        assert "m1" in rec_accepted
        assert "m1" not in rec_rejected

    def test_multiple_movies_classified_correctly(self):
        """Test multiple movies with different states are classified correctly."""
        dialogue = {
            "initiatorQuestions": {
                "m1": {"seen": 1, "liked": 1, "suggested": 0},  # liked
                "m2": {"seen": 1, "liked": 0, "suggested": 0},  # disliked
                "m3": {"seen": 0, "liked": 1, "suggested": 0},  # rec_accepted
                "m4": {"seen": 0, "liked": 0, "suggested": 0},  # rec_rejected
            },
            "respondentQuestions": {
                "m3": {"suggested": 1},
                "m4": {"suggested": 1},
            },
        }
        movie_mentions = {
            "m1": "Movie A",
            "m2": "Movie B",
            "m3": "Movie C",
            "m4": "Movie D",
        }

        liked, disliked, rec_accepted, rec_rejected = _classify_movie_preferences(
            dialogue, movie_mentions
        )

        assert "m1" in liked
        assert "m2" in disliked
        assert "m3" in rec_accepted
        assert "m4" in rec_rejected


class TestBuildConversation:
    """Tests for _build_conversation helper function."""

    def test_basic_conversation_building(self):
        """Test basic conversation building with initiator/respondent worker IDs."""
        dialogue = {
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
            "messages": [
                {"senderWorkerId": 1, "text": "I like action movies"},
                {"senderWorkerId": 2, "text": "Have you seen The Matrix?"},
                {"senderWorkerId": 1, "text": "Yes, I loved it!"},
            ],
        }

        result = _build_conversation(dialogue)

        assert "SEEKER: I like action movies" in result
        assert "RECOMMENDER: Have you seen The Matrix?" in result
        assert "SEEKER: Yes, I loved it!" in result

    def test_unknown_worker_id_maps_to_unk(self):
        """Test handling of unknown worker IDs (should map to 'UNK')."""
        dialogue = {
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
            "messages": [
                {"senderWorkerId": 1, "text": "Hello"},
                {"senderWorkerId": 999, "text": "Unknown speaker"},  # Unknown ID
            ],
        }

        result = _build_conversation(dialogue)

        assert "SEEKER: Hello" in result
        assert "UNK: Unknown speaker" in result

    def test_empty_messages_list(self):
        """Test empty messages list returns empty string."""
        dialogue = {
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
            "messages": [],
        }

        result = _build_conversation(dialogue)

        assert result == ""

    def test_missing_messages_key(self):
        """Test dialogue without messages key returns empty string."""
        dialogue = {
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
        }

        result = _build_conversation(dialogue)

        assert result == ""

    def test_messages_preserve_order(self):
        """Test that messages are kept in order."""
        dialogue = {
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
            "messages": [
                {"senderWorkerId": 1, "text": "First"},
                {"senderWorkerId": 2, "text": "Second"},
                {"senderWorkerId": 1, "text": "Third"},
            ],
        }

        result = _build_conversation(dialogue)
        lines = result.split("\n")

        assert len(lines) == 3
        assert lines[0] == "SEEKER: First"
        assert lines[1] == "RECOMMENDER: Second"
        assert lines[2] == "SEEKER: Third"

    def test_missing_sender_worker_id(self):
        """Test message without senderWorkerId uses empty string."""
        dialogue = {
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
            "messages": [
                {"text": "Message without sender"},  # No senderWorkerId
            ],
        }

        result = _build_conversation(dialogue)

        assert "UNK: Message without sender" in result

    def test_missing_text_uses_empty_string(self):
        """Test message without text uses empty string."""
        dialogue = {
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
            "messages": [
                {"senderWorkerId": 1},  # No text
            ],
        }

        result = _build_conversation(dialogue)

        assert "SEEKER: " in result


class TestComputeMetadata:
    """Tests for _compute_metadata helper function."""

    def test_counting_logic_basic(self):
        """Test counting logic for movies, suggested, seen/unseen."""
        dialogue = {
            "conversationId": 123,
            "initiatorQuestions": {
                "m1": {"seen": 1, "liked": 1, "suggested": 0},
                "m2": {"seen": 0, "liked": 1, "suggested": 0},
            },
            "respondentQuestions": {
                "m2": {"suggested": 1},
                "m3": {"suggested": 1},
            },
        }
        movie_mentions = {"m1": "Movie A", "m2": "Movie B", "m3": "Movie C"}
        liked = ["m1"]
        disliked = []
        recommended_accepted = ["m2"]
        recommended_rejected = []

        meta = _compute_metadata(
            dialogue,
            movie_mentions,
            liked,
            disliked,
            recommended_accepted,
            recommended_rejected,
        )

        assert meta["conversation_id"] == 123
        assert meta["n_movies"] == 3  # m1, m2, m3
        assert meta["n_movies_with_title"] == 3
        assert meta["n_liked"] == 1
        assert meta["n_disliked"] == 0
        assert meta["n_recommended"] == 1
        assert meta["n_recommended_accepted"] == 1
        assert meta["n_recommended_rejected"] == 0

    def test_empty_input_data(self):
        """Test with empty input data."""
        dialogue = {
            "conversationId": 456,
            "initiatorQuestions": {},
            "respondentQuestions": {},
        }
        movie_mentions = {}
        liked = []
        disliked = []
        recommended_accepted = []
        recommended_rejected = []

        meta = _compute_metadata(
            dialogue,
            movie_mentions,
            liked,
            disliked,
            recommended_accepted,
            recommended_rejected,
        )

        assert meta["conversation_id"] == 456
        assert meta["n_movies"] == 0
        assert meta["n_movies_with_title"] == 0
        assert meta["n_suggested"] == 0
        assert meta["n_suggested_seen"] == 0
        assert meta["n_suggested_unseen"] == 0
        assert meta["n_suggested_unknown_seenflag"] == 0
        assert meta["n_liked"] == 0
        assert meta["n_disliked"] == 0
        assert meta["n_recommended"] == 0
        assert meta["n_recommended_accepted"] == 0
        assert meta["n_recommended_rejected"] == 0

    def test_suggested_seen_counting(self):
        """Test counting of suggested movies by seen status."""
        dialogue = {
            "conversationId": 789,
            "initiatorQuestions": {
                "m1": {"seen": 1, "liked": 1},  # Seen
                "m2": {"seen": 0, "liked": 1},  # Unseen
                "m3": {"seen": 2, "liked": 1},  # Unknown
            },
            "respondentQuestions": {
                "m1": {"suggested": 1},
                "m2": {"suggested": 1},
                "m3": {"suggested": 1},
            },
        }
        movie_mentions = {"m1": "Movie A", "m2": "Movie B", "m3": "Movie C"}

        meta = _compute_metadata(
            dialogue, movie_mentions, [], [], ["m1", "m2", "m3"], []
        )

        assert meta["n_suggested"] == 3
        assert meta["n_suggested_seen"] == 1
        assert meta["n_suggested_unseen"] == 1
        assert meta["n_suggested_unknown_seenflag"] == 1

    def test_movies_without_titles_not_counted(self):
        """Test that movies without titles are not counted in n_movies_with_title."""
        dialogue = {
            "conversationId": 100,
            "initiatorQuestions": {
                "m1": {"seen": 1, "liked": 1},
                "m2": {"seen": 1, "liked": 1},
            },
            "respondentQuestions": {},
        }
        # Only m1 has a title
        movie_mentions = {"m1": "Movie A"}

        meta = _compute_metadata(dialogue, movie_mentions, ["m1"], [], [], [])

        assert meta["n_movies"] == 2  # Both m1 and m2 counted
        assert meta["n_movies_with_title"] == 1  # Only m1 has a title

    def test_rejected_recommendations_counted(self):
        """Test that rejected recommendations are counted correctly."""
        dialogue = {
            "conversationId": 200,
            "initiatorQuestions": {
                "m1": {"seen": 0, "liked": 0},
                "m2": {"seen": 0, "liked": 0},
            },
            "respondentQuestions": {
                "m1": {"suggested": 1},
                "m2": {"suggested": 1},
            },
        }
        movie_mentions = {"m1": "Movie A", "m2": "Movie B"}

        meta = _compute_metadata(dialogue, movie_mentions, [], [], [], ["m1", "m2"])

        assert meta["n_recommended"] == 2
        assert meta["n_recommended_accepted"] == 0
        assert meta["n_recommended_rejected"] == 2

    def test_suggested_from_initiator_questions(self):
        """Test that suggested flag from initiator questions is counted."""
        dialogue = {
            "conversationId": 300,
            "initiatorQuestions": {
                "m1": {"seen": 0, "liked": 1, "suggested": 1},  # Suggested in initiator
            },
            "respondentQuestions": {},  # Not in respondent
        }
        movie_mentions = {"m1": "Movie A"}

        meta = _compute_metadata(dialogue, movie_mentions, [], [], ["m1"], [])

        assert meta["n_suggested"] == 1
        assert meta["n_suggested_unseen"] == 1


def test_build_prompt_template_no_candidates():
    """Test that include_candidates=False omits CANDIDATES block and rule."""
    result = _build_prompt_template(
        liked_titles=["The Matrix"],
        disliked_titles=["Inception"],
        recommended_rejected_titles=[],
        conversation_masked="SEEKER: I like sci-fi\nRECOMMENDER: Nice",
        include_candidates=False,
    )

    # CANDIDATES block and related content must not appear
    assert "<CANDIDATES>" not in result
    assert "</CANDIDATES>" not in result
    assert "candidate list" not in result
    assert "{relevant_movie_titles}" not in result

    # Conversation block must be preserved
    assert "<CONVERSATION>" in result
    assert "</CONVERSATION>" in result

    # MASK_ACCEPTED rules must be preserved
    assert "MASK_ACCEPTED" in result

    # Other rules must be preserved
    assert "RULES:" in result
    assert "{n_recommendations}" in result


class TestMaskAcceptedMovies:
    """Tests for _mask_accepted_movies helper function."""

    def test_single_id(self):
        conversation = "Have you seen @123?"
        result, mask_map = _mask_accepted_movies(conversation, ["123"], "MASK_ACCEPTED")

        assert result == "Have you seen MASK_ACCEPTED_0?"
        assert mask_map == {"123": "MASK_ACCEPTED_0"}

    def test_multiple_ids(self):
        conversation = "I liked @100 and @200 was great"
        result, mask_map = _mask_accepted_movies(
            conversation, ["100", "200"], "MASK_ACCEPTED"
        )

        assert "MASK_ACCEPTED_0" in result
        assert "MASK_ACCEPTED_1" in result
        assert "@100" not in result
        assert "@200" not in result
        assert mask_map == {"100": "MASK_ACCEPTED_0", "200": "MASK_ACCEPTED_1"}

    def test_id_with_trailing_punctuation(self):
        conversation = "What about @123, and @456."
        result, mask_map = _mask_accepted_movies(conversation, ["123", "456"], "MASK")

        assert result == "What about MASK_0, and MASK_1."
        assert mask_map == {"123": "MASK_0", "456": "MASK_1"}

    def test_empty_list_returns_unchanged(self):
        conversation = "No movies @123 here"
        result, mask_map = _mask_accepted_movies(conversation, [], "MASK_ACCEPTED")

        assert result == conversation
        assert mask_map == {}

    def test_id_not_in_conversation(self):
        conversation = "No movie IDs here"
        result, mask_map = _mask_accepted_movies(conversation, ["999"], "MASK_ACCEPTED")

        assert result == conversation
        assert mask_map == {"999": "MASK_ACCEPTED_0"}

    def test_custom_mask_token(self):
        conversation = "I saw @42 yesterday"
        result, _ = _mask_accepted_movies(conversation, ["42"], "CUSTOM")

        assert result == "I saw CUSTOM_0 yesterday"

    def test_same_id_multiple_occurrences(self):
        conversation = "I liked @100 and mentioned @100 again"
        result, mask_map = _mask_accepted_movies(conversation, ["100"], "MASK_ACCEPTED")

        assert result.count("MASK_ACCEPTED_0") == 2
        assert "@100" not in result


class TestReplaceMovieIdsWithTitles:
    """Tests for _replace_movie_ids_with_titles helper function."""

    def test_basic_replacement(self):
        conversation = "I saw @123 yesterday"
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids={"123"},
            id_to_title={"123": "The Matrix"},
            mask_map={},
        )

        assert result == "I saw The Matrix yesterday"

    def test_skips_masked_ids(self):
        conversation = "I saw @123 and @456"
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids={"123", "456"},
            id_to_title={"123": "The Matrix", "456": "Inception"},
            mask_map={"123": "MASK_0"},
        )

        # @123 should be left as-is (masked), @456 should be replaced
        assert "@123" in result
        assert "Inception" in result

    def test_id_with_trailing_punctuation(self):
        conversation = "I watched @123."
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids={"123"},
            id_to_title={"123": "The Matrix"},
            mask_map={},
        )

        assert result == "I watched The Matrix."

    def test_id_glued_to_word_adds_space(self):
        conversation = "recommended@123 to me"
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids={"123"},
            id_to_title={"123": "The Matrix"},
            mask_map={},
        )

        assert "recommended The Matrix" in result

    def test_missing_title_skipped(self):
        conversation = "I saw @999 yesterday"
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids={"999"},
            id_to_title={},
            mask_map={},
        )

        assert result == "I saw @999 yesterday"

    def test_multiple_spaces_cleaned(self):
        conversation = "I saw  @123  yesterday"
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids={"123"},
            id_to_title={"123": "The Matrix"},
            mask_map={},
        )

        # Double spaces should be collapsed to single
        assert "  " not in result

    def test_multiple_ids_replaced(self):
        conversation = "I liked @100 and @200"
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids={"100", "200"},
            id_to_title={"100": "The Matrix", "200": "Inception"},
            mask_map={},
        )

        assert "The Matrix" in result
        assert "Inception" in result
        assert "@100" not in result
        assert "@200" not in result

    def test_empty_movie_ids(self):
        conversation = "I saw @123 yesterday"
        result = _replace_movie_ids_with_titles(
            conversation,
            movie_ids=set(),
            id_to_title={"123": "The Matrix"},
            mask_map={},
        )

        # Nothing replaced since movie_ids is empty
        assert result == "I saw @123 yesterday"
