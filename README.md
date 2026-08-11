# CRS Performance and Stability

Code and artifacts for **Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation**, accepted as a CIKM 2026 short paper.

The repository evaluates proprietary, open-weight, and fine-tuned LLM rerankers alongside collaborative-filtering and sequential baselines on ReDial. It includes the prompt templates, run configurations, raw model outputs, trained RecBole checkpoints, and analysis artifacts used for the reported results.

## Requirements

- Python 3.10
- [`uv`](https://docs.astral.sh/uv/)
- CUDA is recommended for open-weight LLM inference and fine-tuning. RecBole uses CUDA when available and otherwise falls back to CPU.
- Access to the configured LiteLLM gateway is required only to rerun proprietary API models.

Install the environment from the repository root:

```bash
uv sync
```

If package installation fails because custom certificate environment variables are set, retry after unsetting them:

```bash
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
uv sync
```

## API configuration

Copy the example file and add your own API key:

```bash
cp .env.example .env
```

The file should contain:

```dotenv
LITELLM_ENDPOINT=https://cody.ib-inet.com/
LITELLM_API_KEY=your-api-key
```

`.env` is ignored by Git. Do not commit API keys. The evaluation notebooks load this file with `python-dotenv` and append `/v1` to the endpoint when needed.

The provided endpoint and model aliases are organization-specific. External users can point `LITELLM_ENDPOINT` at a compatible OpenAI-style gateway and update the API model identifiers in `scripts/04_evaluation.ipynb` and `scripts/05_stability.ipynb`.

## Repository contents

```text
src/stability/        Shared preprocessing, generation, evaluation, metric, and plotting code
scripts/              Numbered experiment workflow
main.ipynb            Exact analysis used for the paper tables and figures
data/input/           ReDial source archive
data/processed/       Prompt templates, prompt examples, metadata, and embeddings
data/output/          Raw LLM evaluations, stability runs, and paper-derived artifacts
data/recbole/         RecBole datasets, HPO settings, checkpoints, and evaluation artifacts
models/               Split archive of the fine-tuned Qwen2.5-7B LoRA adapter
tests/                Tests for shared code
```

The LaTeX manuscript source is intentionally not included.

## Reproduce the paper analysis

All inputs required by `main.ipynb` are included. Run Jupyter from the repository root so its paths resolve correctly:

```bash
uv run --with jupyter jupyter lab
```

Open and run `main.ipynb`. It regenerates the paper summaries, significance test, tables, and figures under `data/output/paper/`.

The older `scripts/14_paper_analysis.ipynb` contains an earlier, broader RecBole analysis workflow. `main.ipynb` is the authoritative notebook for the final paper.

## Reproduce the experiments

The numbered workflow is:

1. `scripts/00_build_prompt_templates.py` preprocesses ReDial dialogues.
2. `scripts/01_fetch_tmdb_metadata.py` optionally rebuilds enriched movie metadata. The generated metadata used in the paper is already included, so this step is not required. Rebuilding it requires a separate `TMDB_API_KEY`.
3. `scripts/02_build_prompt_examples.py` builds semantic candidate pools and prompt files.
4. `scripts/03_run_finetuning.ipynb` fine-tunes the Qwen2.5-7B LoRA adapter.
5. `scripts/04_evaluation.ipynb` evaluates deterministic LLM runs and retriever variants.
6. `scripts/05_stability.ipynb` runs repeated generations across temperatures.
7. `scripts/06_prepare_recbole_data.ipynb` through `scripts/11_recbole_evaluation_sequential.ipynb` prepare, tune, and evaluate collaborative-filtering and sequential baselines.
8. `scripts/12_inference.py` loads trained RecBole checkpoints for inference.
9. `scripts/13_generate_candidate_pools.py` creates EASE and SASRec candidate pools for LLM reranking.
10. `main.ipynb` produces the final paper analysis.

Run Python scripts from the repository root. Run notebooks in `scripts/` with the working directory set to `scripts/`, as their paths use `../data` and `../models`.

### Rebuild the fine-tuned adapter

The adapter archive and full-catalog prompt file are split to stay below GitHub's 100 MB per-file limit. Reassemble them from the repository root:

```bash
cat models/Qwen2.5-7B-Instruct-FT.tar.gz.part-* > models/Qwen2.5-7B-Instruct-FT.tar.gz
echo "4240856a21b26f7cec9ca0cb1f0063c0aa7335f941af7912fffaea158bd453ae  models/Qwen2.5-7B-Instruct-FT.tar.gz" | shasum -a 256 -c -
tar -xzf models/Qwen2.5-7B-Instruct-FT.tar.gz -C models/

cat data/processed/test_prompt_examples_cALL_r10.jsonl.part-* > data/processed/test_prompt_examples_cALL_r10.jsonl
echo "ce7e5f36ea4677d5fa1807a303ddbca04e29d14132b368c4afc10865df78f64c  data/processed/test_prompt_examples_cALL_r10.jsonl" | shasum -a 256 -c -
```

Alternatively, rerun `scripts/03_run_finetuning.ipynb` to rebuild the adapter and `scripts/02_build_prompt_examples.py` to rebuild all prompt files. The evaluation notebook expects the extracted adapter at `models/Qwen2.5-7B-Instruct-FT/`.

### Generate RecBole candidate pools

After training or restoring the EASE and SASRec checkpoints:

```bash
uv run python scripts/13_generate_candidate_pools.py --models EASE SASRec --top-k 250
```

## Tests and code quality

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ty check .
```

## Data and model notes

- The ReDial source archive is redistributed for reproducibility. ReDial remains subject to its original dataset terms and citation requirements.
- `data/processed/movies_metadata_tmdb.csv` contains derived TMDb metadata. TMDb attribution and terms apply.
- Open-weight base models are downloaded from their original providers and remain subject to their respective licenses. The repository contains only the paper's fine-tuned LoRA adapter, not base-model weights.
- Proprietary API outputs are archived because provider models and serving behavior can change over time.

## Citation

Citation metadata will be added when the final ACM proceedings record is available. Paper DOI: [10.1145/3799682.3840066](https://doi.org/10.1145/3799682.3840066).
