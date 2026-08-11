"""Data preprocessing utilities for ReDial dataset."""

import logging
import re
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def normalize_movie_title(title: str | None) -> str | None:
    """Normalize movie title by removing extra whitespace.

    Args:
        title: Movie title string or None

    Returns:
        Normalized title with single spaces, or None if input is None

    Example:
        >>> normalize_movie_title("The   Matrix")
        'The Matrix'
        >>> normalize_movie_title(None)
        None

    """
    if not isinstance(title, str):
        return title
    return " ".join(title.split())


def _extract_movie_mentions(dialogue: dict[str, Any]) -> dict[str, str | None]:
    """Extract and normalize movie mentions from dialogue.

    Args:
        dialogue: Dialogue dictionary containing movieMentions

    Returns:
        Dictionary mapping movie IDs to normalized titles

    """
    try:
        movie_mentions = dialogue.get("movieMentions", {})
        return {
            movie_id: normalize_movie_title(title)
            for movie_id, title in movie_mentions.items()
        }
    except AttributeError:
        logger.warning("Invalid movieMentions format in dialogue")
        return {}


def _build_conversation(dialogue: dict[str, Any]) -> str:
    """Build conversation string from dialogue messages.

    Args:
        dialogue: Dialogue dictionary containing messages

    Returns:
        Formatted conversation string with SEEKER/RECOMMENDER labels

    """
    worker_dispatcher = {
        dialogue.get("initiatorWorkerId"): "SEEKER",
        dialogue.get("respondentWorkerId"): "RECOMMENDER",
    }

    conversation_parts = []
    for turn in dialogue.get("messages", []):
        worker_id = turn.get("senderWorkerId", "")
        text = turn.get("text", "")
        role = worker_dispatcher.get(worker_id, "UNK")
        conversation_parts.append(f"{role}: {text}")

    return "\n".join(conversation_parts)


def _normalize_questions(questions: dict | list | None) -> dict[str, Any]:
    """Normalize questions to dictionary format.

    Args:
        questions: Questions object (dict, list, or None)

    Returns:
        Dictionary of questions, empty if invalid format

    """
    if isinstance(questions, dict):
        return questions
    # list or None case
    return {}


def _classify_movie_preferences(
    dialogue: dict[str, Any],
    movie_mentions: dict[str, str | None],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Classify movies into liked/disliked/recommended categories.

    Args:
        dialogue: Dialogue dictionary
        movie_mentions: Dictionary of movie ID to title mappings

    Returns:
        Tuple of (liked, disliked, recommended_accepted, recommended_rejected)

    Notes:
        Classification logic:
        - seen=1, liked=1 -> liked
        - seen=1, liked=0 -> disliked
        - seen=0, suggested=1, liked!=0 -> recommended_accepted
        - seen=0, suggested=1, liked=0 -> recommended_rejected

    """
    liked: list[str] = []
    disliked: list[str] = []
    recommended_accepted: list[str] = []
    recommended_rejected: list[str] = []

    # Normalize questions
    initiator_questions = _normalize_questions(dialogue.get("initiatorQuestions"))
    respondent_questions = _normalize_questions(dialogue.get("respondentQuestions"))

    # Get all movie IDs
    all_movie_ids = set(initiator_questions.keys()) | set(respondent_questions.keys())

    for movie_id in all_movie_ids:
        title = movie_mentions.get(movie_id, "")
        if not title:
            continue

        # Get labels from both perspectives
        init_labels = initiator_questions.get(movie_id, {})
        resp_labels = respondent_questions.get(movie_id, {})

        # Check if suggested (primary from respondent, fallback to initiator)
        is_suggested = (
            resp_labels.get("suggested", 0) == 1 or init_labels.get("suggested", 0) == 1
        )

        # Interpret labels: 1=yes, 0=no, 2=unknown/not mentioned
        seen_flag = init_labels.get("seen", 2)
        like_flag = init_labels.get("liked", 2)

        # Classification logic
        if seen_flag == 1:  # Seeker has seen the movie
            if like_flag == 1:
                liked.append(movie_id)
            elif like_flag == 0:
                disliked.append(movie_id)
        elif seen_flag == 0:  # Seeker hasn't seen the movie
            if is_suggested:
                if like_flag in (1, 2):  # Liked or unknown
                    recommended_accepted.append(movie_id)
                elif like_flag == 0:
                    recommended_rejected.append(movie_id)
            elif like_flag == 1:  # Unseen but self-mentioned as liked
                liked.append(movie_id)
            elif like_flag == 0:
                disliked.append(movie_id)
        elif is_suggested:  # Unknown seen status but suggested
            if like_flag in (1, 2):
                recommended_accepted.append(movie_id)
            elif like_flag == 0:
                recommended_rejected.append(movie_id)

    return liked, disliked, recommended_accepted, recommended_rejected


def _compute_metadata(
    dialogue: dict[str, Any],
    movie_mentions: dict[str, str | None],
    liked: list[str],
    disliked: list[str],
    recommended_accepted: list[str],
    recommended_rejected: list[str],
) -> dict[str, Any]:
    """Compute metadata for dialogue analysis.

    Args:
        dialogue: Dialogue dictionary
        movie_mentions: Movie ID to title mappings
        liked: List of liked movie IDs
        disliked: List of disliked movie IDs
        recommended_accepted: List of accepted recommendation IDs
        recommended_rejected: List of rejected recommendation IDs

    Returns:
        Dictionary containing metadata statistics

    """
    initiator_questions = _normalize_questions(dialogue.get("initiatorQuestions"))
    respondent_questions = _normalize_questions(dialogue.get("respondentQuestions"))
    all_movie_ids = set(initiator_questions.keys()) | set(respondent_questions.keys())

    n_movies_with_title = sum(1 for mid in all_movie_ids if movie_mentions.get(mid))

    # Count suggested movies
    n_suggested = 0
    n_suggested_seen = 0
    n_suggested_unseen = 0
    n_suggested_unknown_seenflag = 0

    for movie_id in all_movie_ids:
        if not movie_mentions.get(movie_id):
            continue

        init_labels = initiator_questions.get(movie_id, {})
        resp_labels = respondent_questions.get(movie_id, {})
        is_suggested = (
            resp_labels.get("suggested", 0) == 1 or init_labels.get("suggested", 0) == 1
        )

        if is_suggested:
            n_suggested += 1
            seen_flag = init_labels.get("seen", 2)
            if seen_flag == 1:
                n_suggested_seen += 1
            elif seen_flag == 0:
                n_suggested_unseen += 1
            else:
                n_suggested_unknown_seenflag += 1

    return {
        "conversation_id": dialogue.get("conversationId"),
        "n_movies": len(all_movie_ids),
        "n_movies_with_title": n_movies_with_title,
        "n_suggested": n_suggested,
        "n_suggested_seen": n_suggested_seen,
        "n_suggested_unseen": n_suggested_unseen,
        "n_suggested_unknown_seenflag": n_suggested_unknown_seenflag,
        "n_liked": len(liked),
        "n_disliked": len(disliked),
        "n_recommended": len(recommended_accepted) + len(recommended_rejected),
        "n_recommended_accepted": len(recommended_accepted),
        "n_recommended_rejected": len(recommended_rejected),
    }


def extract_user_preferences(
    dialogue: dict[str, Any],
    *,
    collect_meta: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Extract user preferences from a dialogue.

    Args:
        dialogue: Dialogue dictionary from ReDial dataset
        collect_meta: Whether to collect metadata statistics

    Returns:
        If collect_meta is False: preferences dictionary
        If collect_meta is True: tuple of (preferences, metadata)

    """
    # Extract movie mentions
    movie_mentions = _extract_movie_mentions(dialogue)

    # Build conversation
    conversation = _build_conversation(dialogue)

    # Classify preferences
    liked, disliked, recommended_accepted, recommended_rejected = (
        _classify_movie_preferences(dialogue, movie_mentions)
    )

    # Build preferences dictionary
    preferences = {
        "liked": liked,
        "disliked": disliked,
        "recommended_accepted": recommended_accepted,
        "recommended_rejected": recommended_rejected,
        "recommended": recommended_accepted + recommended_rejected,
        "conversation": conversation.strip(),
        "id_to_title": movie_mentions,
    }

    if collect_meta:
        metadata = _compute_metadata(
            dialogue,
            movie_mentions,
            liked,
            disliked,
            recommended_accepted,
            recommended_rejected,
        )
        return preferences, metadata

    return preferences


def analyze_dialogue_dropout(data: list[dict[str, Any]]) -> pd.DataFrame:
    """Analyze dialogue dropout reasons.

    Args:
        data: List of dialogue dictionaries

    Returns:
        DataFrame with dropout analysis including skip reasons

    Notes:
        Dialogues are skipped if:
        - No liked movies AND no recommendations
        - No liked movies
        - No recommendations

    """
    rows = []
    for dialogue in data:
        prefs, meta = extract_user_preferences(dialogue, collect_meta=True)
        # Type narrowing: when collect_meta=True, extract_user_preferences returns tuple[dict, dict]
        assert isinstance(prefs, dict)
        assert isinstance(meta, dict)

        # Determine skip reason
        reason = None
        if not prefs["liked"] and not prefs["recommended"]:
            reason = "no_liked_and_no_recommended"
        elif not prefs["liked"]:
            reason = "no_liked"
        elif not prefs["recommended"]:
            reason = "no_recommended"

        meta["skipped_reason"] = reason
        rows.append(meta)

    return pd.DataFrame(rows)


def print_dropout_analysis(analysis: pd.DataFrame, dataset_name: str) -> None:
    """Print formatted dropout analysis.

    Args:
        analysis: DataFrame from analyze_dialogue_dropout()
        dataset_name: Name of dataset for display

    """
    total = len(analysis)
    skip_counts = analysis["skipped_reason"].value_counts(dropna=True)

    print(f"Dropout analysis for {dataset_name}:")
    print(f"  Total dialogues: {total}")
    print(f"  Kept dialogues: {total - sum(v for k, v in skip_counts.items() if k)}")
    print("  Skip reasons:")

    for reason, cnt in skip_counts.items():
        pct = 100 * cnt / total
        status = reason or "kept"
        print(f"    {status}: {cnt} ({pct:.1f}%)")


def _compile_id_pattern(movie_id: str) -> re.Pattern:
    """Compile regex pattern for movie ID matching.

    Args:
        movie_id: Movie ID string

    Returns:
        Compiled regex pattern

    """
    return re.compile(rf"@{re.escape(movie_id)}(?P<punc>[.,!?;:]?)")


def _mask_accepted_movies(
    conversation: str,
    recommended_accepted: list[str],
    mask_token: str,
) -> tuple[str, dict[str, str]]:
    """Mask accepted recommendations in conversation.

    Args:
        conversation: Original conversation text
        recommended_accepted: List of accepted recommendation IDs
        mask_token: Base token for masking (e.g., "MASK_ACCEPTED")

    Returns:
        Tuple of (masked_conversation, mask_map)

    """
    conversation_masked = conversation
    mask_map: dict[str, str] = {}

    for i, movie_id in enumerate(recommended_accepted):
        placeholder = f"{mask_token}_{i}"
        mask_map[movie_id] = placeholder
        pattern = _compile_id_pattern(movie_id)
        conversation_masked = pattern.sub(
            lambda m, ph=placeholder: f"{ph}{m.group('punc')}",
            conversation_masked,
        )

    return conversation_masked, mask_map


def _replace_movie_ids_with_titles(
    conversation: str,
    movie_ids: set[str],
    id_to_title: dict[str, str],
    mask_map: dict[str, str],
) -> str:
    """Replace remaining movie IDs with titles.

    Args:
        conversation: Conversation text (possibly with some masked IDs)
        movie_ids: Set of movie IDs to replace
        id_to_title: Mapping of movie ID to title
        mask_map: IDs already masked (skip these)

    Returns:
        Conversation with IDs replaced by titles

    """
    conversation_modified = conversation

    for movie_id in movie_ids:
        if movie_id in mask_map:  # Already masked
            continue

        title = id_to_title.get(movie_id)
        if not title:
            continue

        pattern = _compile_id_pattern(movie_id)

        def _repl(match: re.Match, title_text: str = title) -> str:
            punc = match.group("punc") or ""
            start = match.start()
            # Add space if ID is glued to word (e.g., "word@12345")
            need_space = start > 0 and conversation_modified[start - 1].isalnum()
            prefix = " " if need_space else ""
            return f"{prefix}{title_text}{punc}"

        conversation_modified = pattern.sub(_repl, conversation_modified)

    # Clean up multiple spaces
    return re.sub(r"[ ]{2,}", " ", conversation_modified)


def _build_prompt_template(
    liked_titles: list[str],
    disliked_titles: list[str],
    recommended_rejected_titles: list[str],
    conversation_masked: str,
    *,
    include_candidates: bool = True,
) -> str:
    """Build prompt template string.

    Args:
        liked_titles: List of liked movie titles
        disliked_titles: List of disliked movie titles
        recommended_rejected_titles: List of rejected recommendation titles
        conversation_masked: Masked conversation text
        include_candidates: Whether to include candidate list and related rules

    Returns:
        Formatted prompt template string

    """
    prompt_parts = []

    # User preferences
    prompt_parts.append(
        "The user has watched and likes the following movies: "
        f"{', '.join(liked_titles)}.",
    )

    if disliked_titles:
        prompt_parts.append(
            "The user has watched and didn't like the following movies: "
            f"{', '.join(disliked_titles)}.",
        )

    if recommended_rejected_titles:
        prompt_parts.append(
            "The user rejected these previous suggestions: "
            f"{', '.join(recommended_rejected_titles)}.",
        )

    # Request
    prompt_parts.append(
        "\nRecommend EXACTLY {n_recommendations} movies this user would watch.\n",
    )

    # Rules
    rules = ["RULES:"]
    if include_candidates:
        rules.append(
            " - Choose ONLY from the provided candidate list tagged with <CANDIDATES>.",
        )
    rules.extend(
        [
            " - Respond with EXACTLY {n_recommendations} movie titles, one per line, with NO additional text, numbering, punctuation, or commentary of any kind.",
            " - Start listing movies right away and do NOT include any text other than movie titles.",
            " - Use the conversation tagged with <CONVERSATION> between the SEEKER, and RECOMMENDER as context.",
            " - In the conversation, each `MASK_ACCEPTED_x` placeholder represents a specific movie that was recommended and accepted by the user.",
            " - The order of recommendations MUST strictly follow this logic:",
            "   1. First, list movies corresponding to each `MASK_ACCEPTED_x` placeholder in ascending numerical order (MASK_ACCEPTED_0, then MASK_ACCEPTED_1, etc.)",
            "   2. Then, fill remaining slots (up to {n_recommendations} total) with the most relevant candidates from the list, ranked by relevance",
            " - Output format example:",
            "   Movie Title 1 (Year)",
            "   Movie Title 2 (Year)",
            "   Movie Title 3 (Year)",
            "   ...",
            "   Movie Title {n_recommendations} (Year)\n",
        ],
    )
    prompt_parts.extend(rules)

    # Placeholders
    if include_candidates:
        prompt_parts.append("<CANDIDATES>\n{relevant_movie_titles}\n</CANDIDATES>\n")
    prompt_parts.append(f"<CONVERSATION>\n{conversation_masked}\n</CONVERSATION>")

    return "\n".join(prompt_parts)


def generate_prompt_templates(
    data: list[dict],
    *,
    mask_accepted: bool = True,
    mask_token: str = "MASK_ACCEPTED",
    include_candidates: bool = True,
) -> list[dict]:
    """Generate prompt templates from dialogue data.

    Args:
        data: List of dialogue dictionaries
        mask_accepted: Whether to mask accepted recommendations
        mask_token: Base token for masking
        include_candidates: Whether to include candidate list and related rules

    Returns:
        List of prompt template dictionaries with keys:
        - prompt: Template string with placeholders
        - user_liked: List of liked movie titles
        - user_disliked: List of disliked movie titles
        - recommended_accepted: List of accepted recommendation titles
        - recommended_rejected: List of rejected recommendation titles
        - recommended: All recommended titles
        - dialogue_id: Conversation ID

    Notes:
        Skips dialogues with no liked movies or no recommendations.
        Masks accepted recommendations as MASK_ACCEPTED_0, MASK_ACCEPTED_1, etc.

    """
    examples: list[dict] = []

    for dialogue in tqdm(data, desc="Generating templates"):
        # Extract preferences (without collect_meta, always returns dict)
        prefs = extract_user_preferences(dialogue)
        assert isinstance(
            prefs, dict
        )  # Type narrowing: collect_meta=False returns dict

        # Get fields with defaults
        liked = prefs.get("liked") or []
        disliked = prefs.get("disliked") or []
        recommended_accepted = prefs.get("recommended_accepted") or []
        recommended_rejected = prefs.get("recommended_rejected") or []
        recommended = prefs.get("recommended") or []
        conversation = prefs.get("conversation") or ""
        id_to_title = prefs.get("id_to_title") or {}

        # Skip if no liked movies or no recommendations
        if not liked or not recommended:
            continue

        # Mask accepted recommendations
        if mask_accepted:
            conversation_masked, mask_map = _mask_accepted_movies(
                conversation,
                recommended_accepted,
                mask_token,
            )
        else:
            conversation_masked = conversation
            mask_map = {}

        # Replace remaining IDs with titles
        remaining_ids = set(liked) | set(disliked) | set(recommended_rejected)
        conversation_masked = _replace_movie_ids_with_titles(
            conversation_masked,
            remaining_ids,
            id_to_title,
            mask_map,
        )

        # Build title lists
        liked_titles = [id_to_title.get(i, i) for i in liked]
        disliked_titles = [id_to_title.get(i, i) for i in disliked]
        recommended_accepted_titles = [
            id_to_title.get(i, i) for i in recommended_accepted
        ]
        recommended_rejected_titles = [
            id_to_title.get(i, i) for i in recommended_rejected
        ]
        recommended_titles = [id_to_title.get(i, i) for i in recommended]

        # Build prompt
        prompt = _build_prompt_template(
            liked_titles,
            disliked_titles,
            recommended_rejected_titles,
            conversation_masked,
            include_candidates=include_candidates,
        )

        examples.append(
            {
                "prompt": prompt,
                "user_liked": liked_titles,
                "user_disliked": disliked_titles,
                "recommended_accepted": recommended_accepted_titles,
                "recommended_rejected": recommended_rejected_titles,
                "recommended": recommended_titles,
                "dialogue_id": dialogue.get("conversationId", None),
            },
        )

    logger.info(
        "Generated %d prompt templates from %d dialogues",
        len(examples),
        len(data),
    )
    return examples
