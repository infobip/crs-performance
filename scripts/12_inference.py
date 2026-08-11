r"""Inference tool for trained CF and Sequential recommendation models.

Loads a trained RecBole model and generates top-K recommendations.

CF models require a dialogue_id (user must exist in dataset).
Sequential models accept arbitrary item_id sequences or a dialogue_id.

Usage:
    # CF inference
    uv run python scripts/12_inference.py --model EASE --type cf --dialogue-id 20001

    # Sequential inference with item IDs
    uv run python scripts/12_inference.py --model SASRec --type sequential \
      --item-ids 75796 76042 76067

    # Sequential inference with dialogue ID
    uv run python scripts/12_inference.py --model SASRec --type sequential \
      --dialogue-id 20001

    # List available models
    uv run python scripts/12_inference.py --list-models

    # Smoke test all models
    uv run python scripts/12_inference.py --smoke-test

Reference files:
    data/processed/test_prompt_templates.jsonl  -- browse dialogues by dialogue_id
    data/recbole/redial/id_to_title.json        -- browse movies by item_id
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if not getattr(torch, "_recbole_patched", False):
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)

    torch.load = _patched_load
    torch._recbole_patched = True

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, init_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("data")
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_device() -> str:
    # RecBole does not support MPS — only CUDA or CPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _deduplicate(items: list) -> list:
    seen: set = set()
    result: list = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_recbole_model(
    model_name: str,
    model_type: str,
    device: str,
) -> tuple:
    """Load a trained RecBole model with its dataset.

    Returns (model, dataset, test_dataset, config).
    For CF models, test_dataset is None.
    For sequential models, test_dataset contains the frozen test sequences.
    """
    recbole_path = DATA_PATH / "recbole"

    if model_type == "cf":
        dataset_name = "redial"
        hp_path = recbole_path / "hpo_results" / "best_hyperparameters.json"
        weights_path = recbole_path / "final_models" / f"{model_name}.pth"
    else:
        dataset_name = "redial_seq"
        hp_path = recbole_path / "hpo_results_seq" / "best_hyperparameters.json"
        weights_path = recbole_path / "final_models_seq" / f"{model_name}.pth"

    dataset_path = recbole_path / dataset_name

    for p, desc in [
        (hp_path, "Best hyperparameters"),
        (weights_path, "Model weights"),
    ]:
        if not p.exists():
            print(f"Error: {desc} not found at {p}")
            sys.exit(1)

    inter_files = list(dataset_path.glob("*.inter"))
    if not inter_files:
        print(f"Error: No .inter files found in {dataset_path}")
        print("Run notebook 06 (CF) or 09 (Sequential) first.")
        sys.exit(1)

    with hp_path.open() as f:
        all_hp = json.load(f)
    if model_name not in all_hp:
        print(f"Error: {model_name} not found in {hp_path}")
        print(f"Available models: {list(all_hp.keys())}")
        sys.exit(1)
    best_params = all_hp[model_name].get("params", {})

    config_dict = {
        "data_path": str(recbole_path),
        "dataset": dataset_name,
        "benchmark_filename": ["train", "valid", "test"],
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]}
        if model_type == "cf"
        else {"inter": ["user_id", "item_id", "item_id_list"]},
        "MAX_ITEM_LIST_LENGTH": 50,
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "split": {"LS": "valid_and_test"},
            "mode": "full",
        },
        "metrics": ["NDCG"],
        "topk": [10],
        "valid_metric": "NDCG@10",
        "train_neg_sample_args": None,
        "device": device,
        "seed": SEED,
        "reproducibility": True,
        "show_progress": False,
        "state": "WARNING",
        "model": model_name,
        **best_params,
    }

    if model_type == "sequential":
        config_dict.update(
            {
                "LIST_SUFFIX": "_list",
                "ITEM_LIST_LENGTH_FIELD": "item_length",
                "alias_of_item_id": ["item_id_list"],
                "repeatable": True,
            }
        )

    if model_type == "cf" and model_name in ("BPR", "LightGCN", "NeuMF"):
        config_dict["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": 1,
            "dynamic": False,
        }

    config = Config(model=model_name, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data._dataset).to(device)

    # Pop/Random don't have learned weights — their state comes from training data.
    # Re-fit them from training data instead of loading empty state_dict.
    if model_name in ("Pop", "Random"):
        from recbole.trainer import Trainer

        trainer = Trainer(config, model)
        trainer.fit(train_data, show_progress=False)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(f"Loaded {model_name} ({model_type}) on {device}")
    print(f"  Params: {best_params}")
    print(f"  Items: {dataset.item_num}, Users: {dataset.user_num}")

    test_dataset = test_data.dataset if model_type == "sequential" else None
    return model, train_data._dataset, test_dataset, config


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def get_item_scores(model, dataset, user_id: int) -> np.ndarray | None:
    """Get item scores for a single user (CF models)."""
    user_token2id = dataset.field2token_id[dataset.uid_field]
    internal_uid = user_token2id.get(str(user_id))
    if internal_uid is None:
        return None

    user_tensor = torch.LongTensor([internal_uid]).to(model.device)

    with torch.no_grad():
        try:
            interaction = Interaction({dataset.uid_field: user_tensor})
            interaction = interaction.to(model.device)
            scores = model.full_sort_predict(interaction).cpu().numpy().flatten()
        except NotImplementedError:
            n_items = dataset.item_num
            item_tensor = torch.arange(n_items).to(model.device)
            interaction = Interaction(
                {
                    dataset.uid_field: user_tensor.expand(n_items),
                    dataset.iid_field: item_tensor,
                }
            )
            interaction = interaction.to(model.device)
            scores = model.predict(interaction).cpu().numpy()

    return scores


def get_item_scores_sequential(
    model, dataset, user_id: int, user_to_seq: dict
) -> np.ndarray | None:
    """Get item scores for a sequential model using a known user's sequence."""
    user_token2id = dataset.field2token_id[dataset.uid_field]
    internal_uid = user_token2id.get(str(user_id))
    if internal_uid is None:
        return None

    user_tensor = torch.LongTensor([internal_uid]).to(model.device)
    interaction_dict = {dataset.uid_field: user_tensor}

    seq_field = getattr(dataset, "item_list_length_field", None)
    if seq_field and internal_uid in user_to_seq:
        seq_data = user_to_seq[internal_uid]
        item_seq_field = dataset.iid_field + "_list"
        interaction_dict[item_seq_field] = seq_data["seq"].unsqueeze(0).to(model.device)
        interaction_dict[seq_field] = torch.LongTensor([int(seq_data["len"])]).to(
            model.device
        )

    with torch.no_grad():
        interaction = Interaction(interaction_dict)
        interaction = interaction.to(model.device)
        return model.full_sort_predict(interaction).cpu().numpy().flatten()


def get_item_scores_from_sequence(
    model, dataset, config, item_ids: list[int]
) -> np.ndarray | None:
    """Get item scores for a sequential model from an arbitrary item_id sequence.

    Does not require a known user — builds the sequence tensor directly.
    """
    item_token2id = dataset.field2token_id[dataset.iid_field]
    max_len = config["MAX_ITEM_LIST_LENGTH"]

    # Convert original item_ids to internal RecBole IDs
    internal_ids = []
    for item_id in item_ids:
        internal_id = item_token2id.get(str(item_id))
        if internal_id is not None:
            internal_ids.append(internal_id)

    if not internal_ids:
        return None

    # Truncate to max_len (keep most recent items)
    if len(internal_ids) > max_len:
        internal_ids = internal_ids[-max_len:]

    seq_len = len(internal_ids)

    # Build padded sequence tensor (pad with 0 on the left)
    item_seq = torch.zeros(max_len, dtype=torch.long)
    item_seq[-seq_len:] = torch.LongTensor(internal_ids)

    # Build interaction — sequential models only need item_seq and item_length
    item_seq_field = dataset.iid_field + "_list"
    len_field = getattr(dataset, "item_list_length_field", "item_length")

    interaction_dict = {
        item_seq_field: item_seq.unsqueeze(0).to(model.device),
        len_field: torch.LongTensor([seq_len]).to(model.device),
    }

    with torch.no_grad():
        interaction = Interaction(interaction_dict)
        interaction = interaction.to(model.device)
        return model.full_sort_predict(interaction).cpu().numpy().flatten()


def build_user_to_seq(dataset) -> dict:
    """Build mapping from internal user_id to their item sequence."""
    user_to_seq = {}
    inter_feat = dataset.inter_feat
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    item_seq_field = iid_field + "_list"
    len_field = getattr(dataset, "item_list_length_field", "item_length")

    if item_seq_field not in inter_feat:
        return user_to_seq

    for i in range(len(inter_feat)):
        uid = int(inter_feat[uid_field][i])
        seq = inter_feat[item_seq_field][i]
        seq_len = inter_feat[len_field][i] if len_field in inter_feat else len(seq)
        if uid not in user_to_seq or int(seq_len) > int(user_to_seq[uid]["len"]):
            user_to_seq[uid] = {"seq": seq, "len": seq_len}

    return user_to_seq


def scores_to_ranked_items(
    scores: np.ndarray,
    dataset,
    id_to_title: dict[int, str],
    exclude_ids: set[int],
    top_k: int,
    candidate_ids: set[int] | None = None,
) -> list[tuple[int, str, float]]:
    """Convert score array to ranked list of (item_id, title, score) tuples."""
    item_id2token = dataset.field2id_token[dataset.iid_field]

    scored_items = []
    for internal_id, score in enumerate(scores):
        if internal_id >= len(item_id2token):
            break
        token = item_id2token[internal_id]
        if token == "[PAD]":
            continue
        try:
            original_id = int(token)
        except ValueError:
            continue
        if original_id in exclude_ids:
            continue
        if candidate_ids is not None and original_id not in candidate_ids:
            continue
        title = id_to_title.get(original_id)
        if title:
            scored_items.append((original_id, title, float(score)))

    scored_items.sort(key=lambda x: x[2], reverse=True)
    return scored_items[:top_k]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_mappings(model_type: str) -> tuple[dict, dict, dict[int, str]]:
    """Load dialogue_to_user, test_targets, and id_to_title mappings.

    Returns (dialogue_to_user, test_targets, id_to_title).
    """
    recbole_path = DATA_PATH / "recbole"
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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def list_available_models() -> None:
    """Print available trained models."""
    recbole_path = DATA_PATH / "recbole"

    cf_dir = recbole_path / "final_models"
    seq_dir = recbole_path / "final_models_seq"

    print("CF models (data/recbole/final_models/):")
    if cf_dir.exists():
        models = sorted(p.stem for p in cf_dir.glob("*.pth"))
        print(f"  {', '.join(models)}" if models else "  (none)")
    else:
        print("  (directory not found)")

    print("\nSequential models (data/recbole/final_models_seq/):")
    if seq_dir.exists():
        models = sorted(p.stem for p in seq_dir.glob("*.pth"))
        print(f"  {', '.join(models)}" if models else "  (none)")
    else:
        print("  (directory not found)")


def run_smoke_test() -> None:
    """Load and test ALL CF and sequential models."""
    device = _get_device()
    recbole_path = DATA_PATH / "recbole"
    all_passed = True

    # Discover models
    cf_models = sorted(p.stem for p in (recbole_path / "final_models").glob("*.pth"))
    seq_models = sorted(
        p.stem for p in (recbole_path / "final_models_seq").glob("*.pth")
    )

    if not cf_models and not seq_models:
        print("Error: No trained models found.")
        sys.exit(1)

    # Test CF models
    if cf_models:
        print("=" * 60)
        print("Testing CF models")
        print("=" * 60)
        dialogue_to_user, test_targets, id_to_title = load_mappings("cf")
        test_dialogue_ids = list(dialogue_to_user.keys())[:3]

        for model_name in cf_models:
            print(f"\n--- {model_name} ---")
            try:
                model, dataset, _, config = load_recbole_model(model_name, "cf", device)
                passed = True
                for did in test_dialogue_ids:
                    user_id = dialogue_to_user[did]
                    scores = get_item_scores(model, dataset, user_id)
                    if scores is None:
                        print(f"  FAIL: dialogue_id={did} — no scores returned")
                        passed = False
                        continue
                    ranked = scores_to_ranked_items(
                        scores, dataset, id_to_title, set(), 10
                    )
                    gt = test_targets.get(user_id, [])
                    gt_titles = {id_to_title.get(gid, "") for gid in gt}
                    hits = sum(1 for _, t, _ in ranked if t in gt_titles)
                    print(
                        f"  dialogue_id={did}: {len(ranked)} recs, "
                        f"{hits}/{len(gt)} ground truth in top-10"
                    )
                # Test CBF candidate pool re-ranking
                did = test_dialogue_ids[0]
                user_id = dialogue_to_user[did]
                scores = get_item_scores(model, dataset, user_id)
                if scores is not None:
                    # Use a small subset of items as candidate pool
                    pool_ids = set(list(id_to_title.keys())[:50])
                    ranked = scores_to_ranked_items(
                        scores,
                        dataset,
                        id_to_title,
                        set(),
                        10,
                        candidate_ids=pool_ids,
                    )
                    all_in_pool = all(iid in pool_ids for iid, _, _ in ranked)
                    if ranked and all_in_pool:
                        print(
                            f"  CBF pool reranking (50 candidates): "
                            f"{len(ranked)} recs, all in pool"
                        )
                    else:
                        print("  FAIL: CBF pool reranking — items outside pool")
                        passed = False

                if passed:
                    print("  PASS")
                else:
                    all_passed = False
            except Exception as e:
                print(f"  FAIL: {e}")
                all_passed = False

    # Test Sequential models
    if seq_models:
        print("\n" + "=" * 60)
        print("Testing Sequential models")
        print("=" * 60)
        dialogue_to_user, test_targets, id_to_title = load_mappings("sequential")
        test_dialogue_ids = list(dialogue_to_user.keys())[:3]

        for model_name in seq_models:
            print(f"\n--- {model_name} ---")
            try:
                model, dataset, test_dataset, config = load_recbole_model(
                    model_name, "sequential", device
                )
                user_to_seq = build_user_to_seq(test_dataset)
                passed = True

                # Test with dialogue_id (known user)
                for did in test_dialogue_ids:
                    user_id = dialogue_to_user[did]
                    scores = get_item_scores_sequential(
                        model, dataset, user_id, user_to_seq
                    )
                    if scores is None:
                        print(f"  FAIL: dialogue_id={did} — no scores returned")
                        passed = False
                        continue
                    ranked = scores_to_ranked_items(
                        scores, dataset, id_to_title, set(), 10
                    )
                    gt = test_targets.get(user_id, [])
                    gt_titles = {id_to_title.get(gid, "") for gid in gt}
                    hits = sum(1 for _, t, _ in ranked if t in gt_titles)
                    print(
                        f"  dialogue_id={did}: {len(ranked)} recs, "
                        f"{hits}/{len(gt)} ground truth in top-10"
                    )

                # Test with arbitrary item_ids (no known user)
                sample_ids = list(id_to_title.keys())[:5]
                scores = get_item_scores_from_sequence(
                    model, dataset, config, sample_ids
                )
                if scores is not None:
                    ranked = scores_to_ranked_items(
                        scores, dataset, id_to_title, set(sample_ids), 5
                    )
                    print(
                        f"  arbitrary sequence ({len(sample_ids)} items): "
                        f"{len(ranked)} recs returned"
                    )
                else:
                    print("  FAIL: arbitrary sequence — no scores returned")
                    passed = False

                # Test CBF candidate pool re-ranking
                did = test_dialogue_ids[0]
                user_id = dialogue_to_user[did]
                scores = get_item_scores_sequential(
                    model, dataset, user_id, user_to_seq
                )
                if scores is not None:
                    pool_ids = set(list(id_to_title.keys())[:50])
                    ranked = scores_to_ranked_items(
                        scores,
                        dataset,
                        id_to_title,
                        set(),
                        10,
                        candidate_ids=pool_ids,
                    )
                    all_in_pool = all(iid in pool_ids for iid, _, _ in ranked)
                    if ranked and all_in_pool:
                        print(
                            f"  CBF pool reranking (50 candidates): "
                            f"{len(ranked)} recs, all in pool"
                        )
                    else:
                        print("  FAIL: CBF pool reranking — items outside pool")
                        passed = False

                if passed:
                    print("  PASS")
                else:
                    all_passed = False
            except Exception as e:
                print(f"  FAIL: {e}")
                all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All models PASSED")
    else:
        print("Some models FAILED")
        sys.exit(1)


def run_single_inference(args) -> None:
    """Run inference for a single dialogue or item sequence."""
    device = args.device or _get_device()
    model_type = args.type

    dialogue_to_user, test_targets, id_to_title = load_mappings(model_type)
    title_to_id = {v: k for k, v in id_to_title.items()}

    model, dataset, test_dataset, config = load_recbole_model(
        args.model, model_type, device
    )

    # Get scores based on input type
    exclude_ids: set[int] = set()

    if args.dialogue_id is not None:
        did = str(args.dialogue_id)
        user_id = dialogue_to_user.get(did)
        if user_id is None:
            print(f"Error: dialogue_id {did} not found in evaluation_targets.json")
            print(
                f"Available test dialogue IDs: {list(dialogue_to_user.keys())[:10]}..."
            )
            sys.exit(1)

        # Load template to show context
        templates_path = DATA_PATH / "processed" / "test_prompt_templates.jsonl"
        user_liked = []
        user_disliked = []
        if templates_path.exists():
            with templates_path.open() as f:
                for line in f:
                    tmpl = json.loads(line)
                    if str(tmpl.get("dialogue_id")) == did:
                        user_liked = tmpl.get("user_liked", [])
                        user_disliked = tmpl.get("user_disliked", [])
                        break

        # Build exclusion set from liked/disliked
        for t in user_liked + user_disliked:
            mid = title_to_id.get(t)
            if mid is not None:
                exclude_ids.add(mid)

        # Print input context
        print(f"\nModel: {args.model} ({model_type})")
        print(f"Input: dialogue_id={did}, user_id={user_id}")
        if user_liked:
            print(f"  Liked: {', '.join(user_liked)}")
        if user_disliked:
            print(f"  Disliked: {', '.join(user_disliked)}")

        # Get scores
        if model_type == "sequential":
            user_to_seq = build_user_to_seq(test_dataset)
            scores = get_item_scores_sequential(model, dataset, user_id, user_to_seq)
        else:
            scores = get_item_scores(model, dataset, user_id)

    elif args.item_ids is not None:
        if model_type != "sequential":
            print("Error: --item-ids is only supported for sequential models.")
            print("CF models require --dialogue-id.")
            sys.exit(1)

        item_ids = args.item_ids
        exclude_ids = set(item_ids)

        # Print input context
        print(f"\nModel: {args.model} ({model_type})")
        print(f"Input sequence ({len(item_ids)} items):")
        for i, iid in enumerate(item_ids, 1):
            title = id_to_title.get(iid, "(unknown)")
            print(f"  {i}. [{iid}] {title}")

        # Warn about unknown IDs
        unknown = [iid for iid in item_ids if iid not in id_to_title]
        if unknown:
            print(f"  Warning: {len(unknown)} item_ids not in catalog: {unknown}")

        scores = get_item_scores_from_sequence(model, dataset, config, item_ids)

    else:
        if model_type == "cf":
            print("Error: CF models require --dialogue-id.")
        else:
            print("Error: Sequential models require --item-ids or --dialogue-id.")
        sys.exit(1)

    if scores is None:
        print("Error: Could not generate scores for this input.")
        sys.exit(1)

    # Rank and display
    ranked = scores_to_ranked_items(
        scores, dataset, id_to_title, exclude_ids, args.top_k
    )

    print(f"\nTop {args.top_k} recommendations:")
    for i, (item_id, title, score) in enumerate(ranked, 1):
        print(f"  {i:3d}. [{item_id}] {title}  (score: {score:.4f})")

    # Show ground truth overlap if available
    if args.dialogue_id is not None:
        user_id = dialogue_to_user[str(args.dialogue_id)]
        gt = test_targets.get(user_id, [])
        if gt:
            gt_titles = {id_to_title.get(gid, "") for gid in gt}
            rec_titles = {t for _, t, _ in ranked}
            hits = gt_titles & rec_titles
            print(f"\nGround truth ({len(gt)} items): ", end="")
            print(", ".join(id_to_title.get(gid, f"ID:{gid}") for gid in gt))
            print(f"Hits in top-{args.top_k}: {len(hits)}/{len(gt)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference tool for trained CF and Sequential recommendation models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reference files:\n"
            "  data/processed/test_prompt_templates.jsonl  — browse dialogues by dialogue_id\n"
            "  data/recbole/redial/id_to_title.json        — browse movies by item_id\n"
            "\n"
            "Examples:\n"
            "  %(prog)s --model EASE --type cf --dialogue-id 20001\n"
            "  %(prog)s --model SASRec --type sequential --item-ids 75796 76042 76067\n"
            "  %(prog)s --model SASRec --type sequential --dialogue-id 20001\n"
            "  %(prog)s --list-models\n"
            "  %(prog)s --smoke-test\n"
        ),
    )
    parser.add_argument("--model", help="Model name (EASE, SASRec, etc.)")
    parser.add_argument("--type", choices=["cf", "sequential"], help="Model type")
    parser.add_argument(
        "--dialogue-id",
        type=int,
        default=None,
        help="Dialogue ID for inference (CF: required, Sequential: optional)",
    )
    parser.add_argument(
        "--item-ids",
        type=int,
        nargs="+",
        default=None,
        help="Item IDs for sequential inference (e.g., --item-ids 75796 76042 76067)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations"
    )
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available trained models and exit",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test all CF and sequential models and exit",
    )
    args = parser.parse_args()

    # Suppress RecBole / pandas noise globally
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Prevent RecBole Config from printing "command line args ... will not be used"
    _orig_argv = sys.argv
    sys.argv = [sys.argv[0]]

    try:
        if args.list_models:
            list_available_models()
        elif args.smoke_test:
            run_smoke_test()
        else:
            if not args.model or not args.type:
                parser.error("--model and --type are required for inference")
            run_single_inference(args)
    finally:
        sys.argv = _orig_argv


if __name__ == "__main__":
    main()
