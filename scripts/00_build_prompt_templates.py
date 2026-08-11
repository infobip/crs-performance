"""Generate prompt templates for training and evaluation from ReDial.

Usage:
    uv run python scripts/00_build_prompt_templates.py
"""

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.stability.preprocessing import (
    generate_prompt_templates,
    normalize_movie_title,
)

# Global variables
DATA_PATH = Path("data")


def main() -> None:
    """Preprocess the ReDial dataset and generate prompt templates."""
    # Read movies with the associated number of mentions
    redial_dataset_path = DATA_PATH / "input" / "redial_dataset"
    try:
        movies_with_mentions = pd.read_csv(
            redial_dataset_path / "movies_with_mentions.csv"
        )
    except FileNotFoundError:
        with zipfile.ZipFile(DATA_PATH / "input" / "redial_dataset.zip", "r") as z:
            z.extractall(redial_dataset_path)
        movies_with_mentions = pd.read_csv(
            redial_dataset_path / "movies_with_mentions.csv"
        )

    # Rename columns for consistency and easier processing
    movies_with_mentions = movies_with_mentions.rename(
        columns={
            "movieId": "id",
            "movieName": "title",
            "nbMentions": "num_mentions",
        },
    )

    # Normalize movie titles
    movies_with_mentions = movies_with_mentions.assign(
        title_norm=movies_with_mentions["title"].map(normalize_movie_title),
    )

    # Save processed movies with mentions
    movies_with_mentions.to_csv(
        DATA_PATH / "processed" / "movies_with_mentions_processed.csv",
        index=False,
    )

    # Read train data; each line is a dialogue object in jsonl format
    train_data = []
    with Path.open(redial_dataset_path / "train_data.jsonl") as f:
        for line in f:
            train_data.append(json.loads(line))
    print(f"Loaded {len(train_data)} train conversations")

    # Read test data; again, each line is a dialogue object in jsonl format
    test_data = []
    with Path.open(redial_dataset_path / "test_data.jsonl") as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test conversations")

    # Process train data
    train_prompts = generate_prompt_templates(train_data)
    print(f"Created {len(train_prompts)} train prompts")

    # Save processed train prompts to a jsonl file
    with Path.open(DATA_PATH / "processed" / "train_prompt_templates.jsonl", "w") as f:
        f.writelines(json.dumps(example) + "\n" for example in train_prompts)
        print(f"Saved {len(train_prompts)} train prompts")

    # Process test data
    test_prompts = generate_prompt_templates(test_data)
    print(f"Created {len(test_prompts)} test prompts")

    # Save processed test prompts to a jsonl file
    with Path.open(DATA_PATH / "processed" / "test_prompt_templates.jsonl", "w") as f:
        f.writelines(json.dumps(example) + "\n" for example in test_prompts)
        print(f"Saved {len(test_prompts)} test prompts")

    # Process test data for zero-shot evaluation (without user preferences)
    test_prompts_zero_shot = generate_prompt_templates(
        test_data, include_candidates=False
    )
    print(
        f"Created {len(test_prompts_zero_shot)} test prompts for zero-shot evaluation"
    )

    # Save processed zero-shot test prompts to a jsonl file
    with Path.open(
        DATA_PATH / "processed" / "test_prompt_templates_zero_shot.jsonl", "w"
    ) as f:
        f.writelines(json.dumps(example) + "\n" for example in test_prompts_zero_shot)
        print(
            f"Saved {len(test_prompts_zero_shot)} test prompts for zero-shot evaluation"
        )


if __name__ == "__main__":
    main()
