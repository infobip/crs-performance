"""Build chat-formatted prompt examples for fine-tuning and evaluation.

Reads prompt templates produced by 00_build_prompt_templates.py, selects
candidate pools via embedding similarity, and writes JSONL files with
chat-formatted examples (system / user / assistant messages).

Usage:
    uv run python scripts/02_build_prompt_examples.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from huggingface_hub import set_client_factory
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.stability.candidates import build_candidates_block
from src.stability.embeddings import build_embeddings

# Constants
DATA_PATH = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SEED = 42
N_RECOMMENDATIONS = 10
N_CANDIDATES = [0, 250, 500, 1000, -1]  # -1 = all movies

SYSTEM_PROMPT = (
    "You are a movie recommendation engine. "
    "Provide exactly the requested number of movie titles, one per line, "
    "in order of preference. "
    "Use the exact movie titles from the provided list of options. "
    "The first recommendations should prioritize movies that best match "
    "the user's stated preferences."
)

SYSTEM_PROMPT_ZERO_SHOT = (
    "You are a movie recommendation engine. "
    "Provide exactly the requested number of movie titles, one per line, "
    "in order of preference. "
    "The first recommendations should prioritize movies that best match "
    "the user's stated preferences."
)


# Data classes
@dataclass(frozen=True)
class ExampleConfig:
    """Configuration for a single prompt-example output file."""

    label: str
    n_candidates: int
    split: str
    inject_ground_truth: bool


# Helpers
def _safe_format(s: str, **kwargs: str) -> str:
    """Return ``s.format(**kwargs)`` safely.

    Treats unmatched ``{`` as literal ``{{`` and unmatched ``}`` as
    literal ``}}``, preventing :class:`KeyError` / :class:`ValueError`
    on templates that contain literal braces (e.g. output-format examples).
    """
    open2 = "@@OPEN2@@"
    close2 = "@@CLOSE2@@"
    unmatched_close = "@@UNMATCHED_CLOSE@@"

    # 1) Protect existing doubled-brace escapes
    s2 = s.replace("{{", open2).replace("}}", close2)

    # 2) Protect valid format fields like {name}, {0}, {value:.2f}
    field_re = re.compile(r"\{(?:[a-zA-Z_]\w*|\d+)(?:[^{}]*)?\}")
    fields: dict[str, str] = {}

    def _store(m: re.Match) -> str:
        token = f"@@FIELD{len(fields)}@@"
        fields[token] = m.group(0)
        return token

    s3 = field_re.sub(_store, s2)

    # 3) Replace remaining '}' (unmatched closers)
    s4 = s3.replace("}", unmatched_close)

    # 4) Escape remaining '{' (unmatched openers)
    s5 = s4.replace("{", "{{")

    # 5) Restore protected fields
    for token, original in fields.items():
        s5 = s5.replace(token, original)

    # 6) Restore doubled-brace escapes
    s5 = s5.replace(open2, "{{").replace(close2, "}}")

    # 7) Format
    formatted = s5.format(**kwargs)

    # 8) Replace unmatched-close marker with literal '}}'
    return formatted.replace(unmatched_close, "}}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    with Path.open(path) as f:
        return [json.loads(line) for line in f]


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a JSONL file."""
    with Path.open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _deduplicate(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# Core logic
def load_embeddings(
    enriched_texts: list[str],
    model: SentenceTransformer,
    embeddings_path: Path,
) -> np.ndarray:
    """Load or compute L2-normalized embeddings."""
    if embeddings_path.exists():
        embeds = np.load(embeddings_path)
        assert embeds.shape[0] == len(enriched_texts), "Embedding count mismatch"
        print(f"Loaded {embeds.shape[0]} extended embeddings")
    else:
        embeds = build_embeddings(
            enriched_texts,
            model=model,
            batch_size=100,
            normalize=True,
            verbose=True,
        )
        np.save(embeddings_path, embeds)
        print(f"Computed and saved {embeds.shape[0]} extended embeddings")

    # Ensure L2-normalized
    if not np.allclose(np.linalg.norm(embeds, axis=1), 1.0):
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeds = (embeds / norms).astype(np.float32)
        print("L2-normalized embeddings")
    else:
        print("Embeddings already L2-normalized")

    return embeds


def prepare_prompt_examples(
    prompt_templates: list[dict[str, Any]],
    system_prompt: str,
    embeds: np.ndarray,
    all_movie_titles: list[str],
    title_to_idx: dict[str, int],
    n_candidates: int,
    n_recommendations: int = 10,
    inject_ground_truth: bool = False,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Build chat-formatted prompt examples from templates.

    Assistant output is always padded to exactly *n_recommendations* lines.
    Ground-truth titles come first, then fillers from the ranked candidate
    pool (or from *all_movie_titles* for zero-shot).

    Args:
        prompt_templates: Output of ``generate_prompt_templates()``.
        system_prompt: System message for the chat.
        embeds: Movie embeddings array ``(num_movies, dim)``.
        all_movie_titles: Movie titles aligned with *embeds*.
        title_to_idx: Title -> embedding index mapping.
        n_candidates: Pool size. ``0`` = zero-shot, ``-1`` = all movies.
        n_recommendations: Number of recommendations to request in prompt.
        inject_ground_truth: Ensure GT titles appear in candidate pool.
        rng: Random generator for reproducible shuffling.

    Returns:
        List of dicts with keys ``messages``, ``ground_truth``,
        ``ground_truth_count``.

    """
    rows: list[dict[str, Any]] = []
    skipped = 0

    for template in tqdm(prompt_templates):
        gt_raw = [
            m
            for m in (template.get("recommended_accepted") or [])
            if isinstance(m, str) and m.strip()
        ]
        if not gt_raw:
            skipped += 1
            continue

        gt = _deduplicate(gt_raw)
        user_liked: list[str] = template.get("user_liked") or []
        user_disliked: list[str] = template.get("user_disliked") or []
        prompt_text: str = template.get("prompt", "")

        # Select candidates
        candidates: list[str]
        ranked: list[str]

        if n_candidates == 0:
            candidates, ranked = [], []
        elif n_candidates == -1:
            assert rng is not None, "rng required for n_candidates=-1"
            excluded = set(user_liked) | set(user_disliked)
            pool = [t for t in all_movie_titles if t not in excluded]
            if inject_ground_truth:
                missing = [t for t in gt if t not in set(pool) and t in title_to_idx]
                pool.extend(missing)
            ranked = list(pool)
            candidates = [pool[i] for i in rng.permutation(len(pool))]
        else:
            gt_arg = gt if inject_ground_truth else None
            result = build_candidates_block(
                user_liked=user_liked,
                user_disliked=user_disliked,
                all_titles=all_movie_titles,
                embeds=embeds,
                title_to_idx=title_to_idx,
                pool_size=n_candidates,
                ground_truth=gt_arg,
                rng=rng,
                return_ranked=True,
            )
            # return_ranked=True guarantees a tuple
            assert isinstance(result, tuple)
            candidates, ranked = result

        # Fill prompt template
        fmt_kwargs: dict[str, str] = {
            "n_recommendations": str(n_recommendations),
        }
        if n_candidates != 0:
            fmt_kwargs["relevant_movie_titles"] = "\n".join(candidates)
        user_prompt = _safe_format(prompt_text, **fmt_kwargs)

        # Build assistant output (always exactly n_recommendations lines)
        gt_capped = gt[:n_recommendations]
        n_filler = max(0, n_recommendations - len(gt_capped))
        gt_set = set(gt_capped)

        filler_source = ranked if ranked else all_movie_titles
        excluded_from_fillers = gt_set | set(user_liked) | set(user_disliked)
        fillers = [t for t in filler_source if t not in excluded_from_fillers][
            :n_filler
        ]
        output_lines = (gt_capped + fillers)[:n_recommendations]

        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "\n".join(output_lines)},
                ],
                "ground_truth_count": len(gt_capped),
                "ground_truth": gt_capped,
            },
        )

    print(f"Built {len(rows)} examples (skipped {skipped} with no ground truth)")
    return rows


def validate_examples(
    rows: list[dict[str, Any]],
    n_recommendations: int,
) -> dict[str, int]:
    """Validate prompt examples for ordering, count, and duplicates."""
    issues = {"gt_order_mismatch": 0, "wrong_count": 0, "duplicates": 0}
    for row in rows:
        gt: list[str] = row["ground_truth"]
        lines = [
            line.strip()
            for line in row["messages"][2]["content"].split("\n")
            if line.strip()
        ]
        if any(i >= len(lines) or lines[i] != gt[i] for i in range(len(gt))):
            issues["gt_order_mismatch"] += 1
        if len(lines) != n_recommendations:
            issues["wrong_count"] += 1
        if len(lines) != len(set(lines)):
            issues["duplicates"] += 1
    return issues


def main() -> None:
    """Build and save prompt examples for all configurations."""
    if os.environ.get("HF_DISABLE_SSL_VERIFY") == "1":
        set_client_factory(
            lambda: httpx.Client(
                verify=False,
                follow_redirects=True,
                timeout=httpx.Timeout(10.0, write=60.0),
            ),
        )

    processed = DATA_PATH / "processed"

    # Load movie titles
    movies_df = pd.read_csv(
        processed / "movies_with_mentions_processed.csv",
    ).drop_duplicates("title_norm")
    all_movie_titles: list[str] = movies_df["title_norm"].tolist()
    print(f"Loaded {len(all_movie_titles)} unique movie titles")

    # Load prompt templates
    train_templates = _read_jsonl(processed / "train_prompt_templates.jsonl")
    test_templates = _read_jsonl(processed / "test_prompt_templates.jsonl")
    test_templates_zs = _read_jsonl(processed / "test_prompt_templates_zero_shot.jsonl")
    print(
        f"Loaded templates: {len(train_templates)} train, "
        f"{len(test_templates)} test, {len(test_templates_zs)} zero-shot test",
    )

    # Load / compute embeddings
    metadata_df = pd.read_csv(processed / "movies_metadata_tmdb.csv")
    movies_merged = movies_df.merge(
        metadata_df[["title_norm", "enriched_text"]],
        on="title_norm",
        how="left",
    )
    movies_merged["enriched_text"] = movies_merged["enriched_text"].fillna(
        movies_merged["title_norm"],
    )
    enriched_texts: list[str] = movies_merged["enriched_text"].tolist()
    with_meta = sum(
        t != n
        for t, n in zip(enriched_texts, movies_merged["title_norm"], strict=False)
    )
    print(f"Enriched texts: {len(enriched_texts)} ({with_meta} with metadata)")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings_path = (
        processed / f"movies_embeds_extended_{EMBEDDING_MODEL.split('/')[1]}.npy"
    )
    embeds = load_embeddings(enriched_texts, model, embeddings_path)

    # Build configs
    title_to_idx = {t: i for i, t in enumerate(all_movie_titles)}
    rng = np.random.default_rng(SEED)

    configs: list[tuple[ExampleConfig, list[dict[str, Any]]]] = [
        # Fine-tuning train set: GT injected into candidates
        (
            ExampleConfig("c250", 250, "train", inject_ground_truth=True),
            train_templates,
        ),
        # Evaluation test sets: no GT injection
        *[
            (
                ExampleConfig(
                    label="c0" if nc == 0 else ("cALL" if nc == -1 else f"c{nc}"),
                    n_candidates=nc,
                    split="test",
                    inject_ground_truth=False,
                ),
                test_templates_zs if nc == 0 else test_templates,
            )
            for nc in N_CANDIDATES
        ],
    ]

    # Generate, validate, and save
    for cfg, templates in configs:
        sys_prompt = SYSTEM_PROMPT_ZERO_SHOT if cfg.n_candidates == 0 else SYSTEM_PROMPT

        rows = prepare_prompt_examples(
            prompt_templates=templates,
            system_prompt=sys_prompt,
            embeds=embeds,
            all_movie_titles=all_movie_titles,
            title_to_idx=title_to_idx,
            n_candidates=cfg.n_candidates,
            n_recommendations=N_RECOMMENDATIONS,
            inject_ground_truth=cfg.inject_ground_truth,
            rng=rng,
        )

        issues = validate_examples(rows, n_recommendations=N_RECOMMENDATIONS)
        if any(v > 0 for v in issues.values()):
            msg = f"Validation failed for {cfg.split}_{cfg.label}: {issues}"
            raise ValueError(msg)

        output_path = (
            processed
            / f"{cfg.split}_prompt_examples_{cfg.label}_r{N_RECOMMENDATIONS}.jsonl"
        )
        _write_jsonl(rows, output_path)
        print(f"  -> Saved {len(rows)} examples to {output_path.name}")


if __name__ == "__main__":
    main()
