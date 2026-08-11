"""Utility functions for data I/O, parsing, and canonicalization.

This module provides:
- DataFrame I/O with metadata preservation (.attrs)
- String canonicalization for movie title matching
- LLM output parsing with fuzzy matching
- Bootstrap confidence interval computation
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_frame_with_attrs(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV and attributes to a separate .attrs file.

    Args:
        df: DataFrame to save
        path: Path to save the DataFrame and attributes

    Raises:
        OSError: If file cannot be written

    Notes:
        - Saves DataFrame to {path}.csv
        - Saves attributes to {path}.attrs
        - Creates parent directories if needed

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False)
        logger.debug("Saved DataFrame to %s", path)
    except OSError:
        logger.exception("Failed to save DataFrame to %s", path)
        raise

    attrs_path = path.with_suffix(".attrs")
    if df.attrs:
        try:
            with attrs_path.open("w") as f:
                for key, value in df.attrs.items():
                    f.write(f"{key}: {value}\n")
            logger.debug("Saved attributes to %s", attrs_path)
        except OSError:
            logger.exception("Failed to save attributes to %s", attrs_path)
            raise


def load_frame_with_attrs(path: Path) -> pd.DataFrame:
    """Load DataFrame from CSV and attributes from a .attrs file.

    Args:
        path: Path to the DataFrame file

    Returns:
        DataFrame with loaded attributes (if .attrs file exists)

    Raises:
        FileNotFoundError: If DataFrame file doesn't exist
        pd.errors.ParserError: If CSV cannot be parsed

    Notes:
        - Loads DataFrame from {path}.csv
        - Loads attributes from {path}.attrs if it exists
        - Automatically converts numeric attribute values

    """
    path = Path(path)

    try:
        df = pd.read_csv(path)
        logger.debug("Loaded DataFrame from %s", path)
    except FileNotFoundError:
        logger.exception("DataFrame file not found: %s", path)
        raise
    except pd.errors.ParserError:
        logger.exception("Failed to parse CSV file: %s", path)
        raise

    attrs_path = path.with_suffix(".attrs")
    if attrs_path.exists():
        try:
            with attrs_path.open() as f:
                attrs = {}
                for line in f:
                    if ":" not in line:
                        continue
                    key, value = line.strip().split(": ", 1)

                    # Try to convert to int or float if possible
                    if re.match(r"^-?\d+$", value):
                        value = int(value)
                    elif re.match(r"^-?\d+\.\d+$", value):
                        value = float(value)

                    attrs[key] = value
                df.attrs = attrs
            logger.debug("Loaded attributes from %s", attrs_path)
        except OSError:
            logger.warning("Failed to load attributes from %s", attrs_path)

    return df


@overload
def canonicalize(s: str) -> str: ...
@overload
def canonicalize(s: None) -> None: ...
def canonicalize(s: str | None) -> str | None:
    """Return a canonicalized version of the string for comparison.

    Args:
        s: Input string.

    Returns:
        Canonicalized string.

    """
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)

    # normalize dashes (em-dash and en-dash to hyphen)
    s = s.replace("\u2014", "-").replace("\u2013", "-")

    # strip Unicode accents (e.g., é → e, ü → u)
    s = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )

    # remove common punctuation and normalize
    result = s.strip().strip('"').strip("'").rstrip(".,:;!?").lower()

    # handle common title variations
    result = re.sub(r"\s+", " ", result)  # normalize whitespace
    return result.replace("&", "and")  # handle & vs and


def map_output_to_option(output: str, options_mapping: dict[str, str]) -> str | None:
    """Return the mapped option for the given output.

    Args:
        output: LLM output string.
        options_mapping: Mapping from canonicalized output strings to valid options.

    Returns:
        Mapped option string or None if not found.

    """
    output_canon = canonicalize(output)
    if output_canon is None:
        return None
    return options_mapping.get(output_canon)


def parse_output(
    output: str | None,
    options_mapping: dict[str, str],
    n_recommendations: int = 5,
) -> list[str] | None:
    """Parse LLM output to extract ordered list of movie recommendations.

    Args:
        output: LLM output string.
        options_mapping: Mapping from canonicalized output strings to valid options.
        n_recommendations: Number of recommendations to extract.

    Returns:
        List of mapped recommendations or None if parsing fails.

    Raises:
        No exceptions are raised; parsing errors return None and are logged.
        Internally handles AttributeError, TypeError, KeyError, and IndexError.

    """
    if not output or not isinstance(output, str) or output.strip() == "":
        return None

    recommendations: list[str] = []
    cached: set[str] = set()
    try:
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        for line in lines[:n_recommendations]:  # only take first N lines
            if len(recommendations) >= n_recommendations:
                break

            # Remove leading numbering or bullets
            cleaned_line = re.sub(r"^[\d\.\-\*•\s]+", "", line).strip()
            if not cleaned_line:
                continue

            # map each line to a valid option
            mapped = map_output_to_option(cleaned_line, options_mapping)
            if mapped and mapped not in cached:  # to avoid duplicates
                recommendations.append(mapped)
                cached.add(mapped)
    except AttributeError:
        logger.warning("Could not parse output (type: %s)", type(output))
        logger.debug("Output content: %s", output)
        return None
    except (TypeError, KeyError, IndexError) as e:
        logger.warning("Error parsing output: %s", e)
        logger.debug("Output content: %s", output)
        return None
    else:
        return recommendations if recommendations else None


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 1000,
    ci: float = 95.0,
) -> tuple[float, float, float]:
    """Return mean and bootstrap confidence interval for the given values.

    Args:
        values: Array of values to compute the mean and CI for.
        rng: Numpy random generator for bootstrapping.
        n_boot: Number of bootstrap samples.
        ci: Confidence interval percentage (e.g., 95.0 for 95% CI).

    Returns:
        Tuple of (mean, lower bound, upper bound).

    """
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(np.mean(vals))
    if len(vals) == 1:
        return mean_val, mean_val, mean_val

    boots = [
        rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)
    ]
    low = np.percentile(boots, (100 - ci) / 2)
    high = np.percentile(boots, 100 - (100 - ci) / 2)
    return mean_val, float(low), float(high)
