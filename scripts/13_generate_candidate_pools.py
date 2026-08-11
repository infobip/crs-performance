"""Generate candidate pool JSONL files for selected trained CF and Sequential models.

One-time data preparation step. Processes all test templates and generates
ChatML-formatted JSONL files (same format as CBF pools from script 02),
so they plug directly into 04_evaluation.ipynb.

Run after training final models (notebooks 08 and 11).

Default usage (EASE and SASRec at 250 candidates):
    uv run python scripts/13_generate_candidate_pools.py

Custom candidate sizes:
    uv run python scripts/13_generate_candidate_pools.py --top-k 100 250 500

Specific model only:
    uv run python scripts/13_generate_candidate_pools.py --models EASE SASRec

With candidate provenance sidecars:
    uv run python scripts/13_generate_candidate_pools.py --with-sidecars
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

# Import shared functions from 12_inference.py
_spec = importlib.util.spec_from_file_location(
    "inference", Path(__file__).parent / "12_inference.py"
)
if _spec is None or _spec.loader is None:
    msg = "Could not import helper functions from scripts/12_inference.py"
    raise ImportError(msg)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_user_to_seq = _mod.build_user_to_seq
get_item_scores = _mod.get_item_scores
get_item_scores_sequential = _mod.get_item_scores_sequential
load_recbole_model = _mod.load_recbole_model
scores_to_ranked_items = _mod.scores_to_ranked_items
DATA_PATH = _mod.DATA_PATH

# Shared checkpoint metadata helpers
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stability.checkpoint import load_checkpoint_metadata  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a movie recommendation engine. "
    "Provide exactly the requested number of movie titles, one per line, "
    "in order of preference. "
    "Use the exact movie titles from the provided list of options. "
    "The first recommendations should prioritize movies that best match "
    "the user's stated preferences."
)

N_RECS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_format(s: str, **kwargs: str) -> str:
    """Format string safely, treating unmatched braces as literals."""
    open2, close2 = "@@OPEN2@@", "@@CLOSE2@@"
    unmatched_close = "@@UNMATCHED_CLOSE@@"

    s2 = s.replace("{{", open2).replace("}}", close2)

    field_re = re.compile(r"\{(?:[a-zA-Z_]\w*|\d+)(?:[^{}]*)?\}")
    fields: dict[str, str] = {}

    def _store(m: re.Match) -> str:
        token = f"@@FIELD{len(fields)}@@"
        fields[token] = m.group(0)
        return token

    s3 = field_re.sub(_store, s2)
    s4 = s3.replace("}", unmatched_close)
    s5 = s4.replace("{", "{{")

    for token, original in fields.items():
        s5 = s5.replace(token, original)

    s5 = s5.replace(open2, "{{").replace(close2, "}}")
    formatted = s5.format(**kwargs)
    return formatted.replace(unmatched_close, "}}")


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _get_device() -> str:
    # RecBole does not support MPS — only CUDA or CPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    git_path = shutil.which("git")
    if git_path is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [git_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def _check_checkpoint_metadata_status(checkpoint_path: Path) -> dict[str, Any]:
    """Check whether a checkpoint has a valid metadata sidecar.

    Returns a dict with 'status' ('present' or 'missing') and the
    loaded metadata when present.
    """
    try:
        metadata = load_checkpoint_metadata(checkpoint_path)
    except FileNotFoundError:
        return {"status": "missing", "metadata": None}
    else:
        return {"status": "present", "metadata": metadata}


def build_pool_metadata(
    model_name: str,
    model_type: str,
    top_k: int,
    checkpoint_path: Path,
    data_paths: dict[str, str],
) -> dict[str, Any]:
    """Build metadata dict for a generated candidate pool."""
    ckpt_status = _check_checkpoint_metadata_status(checkpoint_path)
    return {
        "retriever_type": model_type,
        "retriever_model": model_name,
        "n_candidates": top_k,
        "source_checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata_status": ckpt_status["status"],
        "checkpoint_metadata": ckpt_status["metadata"],
        "data_paths": data_paths,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
    }


def save_pool_metadata(metadata: dict[str, Any], jsonl_path: Path) -> Path:
    """Write metadata sidecar next to the pool JSONL file."""
    meta_path = jsonl_path.with_suffix(".metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return meta_path


def save_candidate_sidecar(
    provenance: list[dict],
    jsonl_path: Path,
) -> Path:
    """Write candidate provenance sidecar next to the pool JSONL file.

    Each line corresponds to the same-index row in the pool JSONL and
    contains ``candidate_item_ids``, ``candidate_titles``, and
    ``candidate_scores``.
    """
    sidecar_path = jsonl_path.with_suffix(".candidates.jsonl")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w") as f:
        for row in provenance:
            f.write(json.dumps(row) + "\n")
    return sidecar_path


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------
def generate_pool_for_model(
    model_name: str,
    model_type: str,
    templates: list[dict[str, Any]],
    dialogue_to_user: dict[str, int],
    id_to_title: dict[int, str],
    title_to_id: dict[str, int],
    top_k: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate candidate pool JSONL rows and provenance for a single model.

    Returns ``(rows, provenance)`` where ``provenance`` contains the
    candidate item IDs, titles, and scores for each prompt row.
    """
    model, dataset, test_dataset, _config = load_recbole_model(
        model_name, model_type, device
    )
    user_to_seq = build_user_to_seq(test_dataset) if model_type == "sequential" else {}

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    skipped = 0

    for tmpl in tqdm(templates, desc=f"  {model_name}", leave=False):
        gt_raw = tmpl.get("recommended_accepted", [])
        if not gt_raw:
            skipped += 1
            continue

        did = str(tmpl.get("dialogue_id", ""))
        user_id = dialogue_to_user.get(did)
        if user_id is None:
            skipped += 1
            continue

        gt = _deduplicate([m for m in gt_raw if isinstance(m, str) and m.strip()])
        user_liked = tmpl.get("user_liked", [])
        user_disliked = tmpl.get("user_disliked", [])
        prompt_text = tmpl.get("prompt", "")

        # Get model scores
        if model_type == "sequential":
            scores = get_item_scores_sequential(model, dataset, user_id, user_to_seq)
        else:
            scores = get_item_scores(model, dataset, user_id)

        if scores is None:
            skipped += 1
            continue

        # Build exclusion set
        exclude_ids: set[int] = set()
        for t in user_liked + user_disliked:
            mid = title_to_id.get(t)
            if mid is not None:
                exclude_ids.add(mid)

        # Rank items
        ranked = scores_to_ranked_items(
            scores, dataset, id_to_title, exclude_ids, top_k
        )

        if not ranked:
            skipped += 1
            continue

        ranked_titles = [title for _, title, _ in ranked]

        # Build prompt
        fmt_kwargs = {
            "n_recommendations": str(N_RECS),
            "relevant_movie_titles": "\n".join(ranked_titles),
        }
        user_prompt = _safe_format(prompt_text, **fmt_kwargs)

        # Build assistant output (gt first, then fillers)
        gt_capped = gt[:N_RECS]
        excluded_from_fillers = set(gt_capped) | set(user_liked) | set(user_disliked)
        fillers = [t for t in ranked_titles if t not in excluded_from_fillers]
        output_lines = (gt_capped + fillers)[:N_RECS]

        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "\n".join(output_lines)},
                ],
                "ground_truth_count": len(gt_capped),
                "ground_truth": gt_capped,
            }
        )

        provenance.append(
            {
                "candidate_item_ids": [int(iid) for iid, _, _ in ranked],
                "candidate_titles": [title for _, title, _ in ranked],
                "candidate_scores": [float(score) for _, _, score in ranked],
            }
        )

    print(f"  {model_name}: {len(rows)} examples, {skipped} skipped")
    return rows, provenance


def main() -> None:
    """Generate RecBole named candidate-pool prompt files."""
    parser = argparse.ArgumentParser(
        description="Generate candidate pool JSONL files for selected trained models.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[250],
        help="Candidate pool sizes (default: 250).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["EASE", "SASRec"],
        help="Specific models to process (default: EASE SASRec).",
    )
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument(
        "--n-recs",
        type=int,
        default=N_RECS,
        help="Number of recommendations in output (default: 10).",
    )
    parser.add_argument(
        "--with-sidecars",
        action="store_true",
        help="Write candidate provenance sidecars (.candidates.jsonl).",
    )
    args = parser.parse_args()

    device = args.device or _get_device()
    recbole_path = DATA_PATH / "recbole"
    output_dir = DATA_PATH / "processed"

    # Discover models
    cf_dir = recbole_path / "final_models"
    seq_dir = recbole_path / "final_models_seq"

    cf_models = sorted(p.stem for p in cf_dir.glob("*.pth")) if cf_dir.exists() else []
    seq_models = (
        sorted(p.stem for p in seq_dir.glob("*.pth")) if seq_dir.exists() else []
    )

    if args.models:
        cf_models = [m for m in cf_models if m in args.models]
        seq_models = [m for m in seq_models if m in args.models]

    if not cf_models and not seq_models:
        print("Error: No trained models found.")
        sys.exit(1)

    print(f"CF models: {cf_models}")
    print(f"Sequential models: {seq_models}")
    print(f"Candidate sizes: {args.top_k}")
    print()

    # Load templates
    templates_path = DATA_PATH / "processed" / "test_prompt_templates.jsonl"
    if not templates_path.exists():
        print(f"Error: {templates_path} not found. Run script 00 first.")
        sys.exit(1)
    templates = _read_jsonl(templates_path)
    print(f"Loaded {len(templates)} test templates")

    # Process CF models
    if cf_models:
        print("\n" + "=" * 60)
        print("Generating CF model candidate pools")
        print("=" * 60)

        subdir = "redial"
        dialogue_to_user, _, id_to_title = _load_mappings("cf", recbole_path)
        title_to_id = {v: k for k, v in id_to_title.items()}

        for model_name in cf_models:
            checkpoint_path = recbole_path / "final_models" / f"{model_name}.pth"
            for top_k in args.top_k:
                rows, provenance = generate_pool_for_model(
                    model_name,
                    "cf",
                    templates,
                    dialogue_to_user,
                    id_to_title,
                    title_to_id,
                    top_k,
                    device,
                )
                out_path = (
                    output_dir
                    / f"test_prompt_examples_cf_{model_name}_c{top_k}_r{args.n_recs}.jsonl"
                )
                _write_jsonl(rows, out_path)

                meta = build_pool_metadata(
                    model_name=model_name,
                    model_type="cf",
                    top_k=top_k,
                    checkpoint_path=checkpoint_path,
                    data_paths={
                        "templates": str(templates_path),
                        "mappings": str(
                            recbole_path / subdir / "evaluation_targets.json"
                        ),
                        "id_to_title": str(recbole_path / subdir / "id_to_title.json"),
                    },
                )
                meta_path = save_pool_metadata(meta, out_path)
                print(f"  Saved pool to {out_path}")
                print(f"  Saved metadata to {meta_path}")

                if args.with_sidecars:
                    sidecar_path = save_candidate_sidecar(provenance, out_path)
                    print(f"  Saved sidecar to {sidecar_path}")

    # Process Sequential models
    if seq_models:
        print("\n" + "=" * 60)
        print("Generating Sequential model candidate pools")
        print("=" * 60)

        subdir = "redial_seq"
        dialogue_to_user, _, id_to_title = _load_mappings("sequential", recbole_path)
        title_to_id = {v: k for k, v in id_to_title.items()}

        for model_name in seq_models:
            checkpoint_path = recbole_path / "final_models_seq" / f"{model_name}.pth"
            for top_k in args.top_k:
                rows, provenance = generate_pool_for_model(
                    model_name,
                    "sequential",
                    templates,
                    dialogue_to_user,
                    id_to_title,
                    title_to_id,
                    top_k,
                    device,
                )
                out_path = (
                    output_dir
                    / f"test_prompt_examples_seq_{model_name}_c{top_k}_r{args.n_recs}.jsonl"
                )
                _write_jsonl(rows, out_path)

                meta = build_pool_metadata(
                    model_name=model_name,
                    model_type="sequential",
                    top_k=top_k,
                    checkpoint_path=checkpoint_path,
                    data_paths={
                        "templates": str(templates_path),
                        "mappings": str(
                            recbole_path / subdir / "evaluation_targets.json"
                        ),
                        "id_to_title": str(recbole_path / subdir / "id_to_title.json"),
                    },
                )
                meta_path = save_pool_metadata(meta, out_path)
                print(f"  Saved pool to {out_path}")
                print(f"  Saved metadata to {meta_path}")

                if args.with_sidecars:
                    sidecar_path = save_candidate_sidecar(provenance, out_path)
                    print(f"  Saved sidecar to {sidecar_path}")

    print("\nDone!")


def _load_mappings(
    model_type: str, recbole_path: Path
) -> tuple[dict[str, int], dict[int, Any], dict[int, str]]:
    """Load dialogue_to_user, test_targets, and id_to_title."""
    subdir = "redial" if model_type == "cf" else "redial_seq"
    targets_path = recbole_path / subdir / "evaluation_targets.json"
    id_to_title_path = recbole_path / subdir / "id_to_title.json"

    with targets_path.open() as f:
        targets_data = json.load(f)
    dialogue_to_user = {
        str(k): int(v) for k, v in targets_data.get("test_dialogue_to_user", {}).items()
    }
    test_targets = {int(k): v for k, v in targets_data["test_targets"].items()}

    with id_to_title_path.open() as f:
        id_to_title = {int(k): v for k, v in json.load(f).items()}

    return dialogue_to_user, test_targets, id_to_title


if __name__ == "__main__":
    main()
