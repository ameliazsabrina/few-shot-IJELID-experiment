# Few-Shot Language Identification for IJELID

Thesis experiments on token-level language identification in Indonesian–Javanese–English tweets using large language models. The notebooks compare **0, 1, 3, and 5-shot prompting**, using training tweets as demonstrations and evaluating validation and test splits separately, without fine-tuning.

## Project contents

- `qwen.ipynb` — experiments with `Qwen/Qwen2.5-1.5B-Instruct`.
- `meta-llama.ipynb` — experiments with `meta-llama/Llama-3.2-1B-Instruct`.
- `tests/test_qwen_decoder.py` — offline checks for constrained decoding and output parsing.
- `outputs/` — generated predictions, metrics, confusion matrices, error reports, and run manifests.

Labels: `ID` (Indonesian), `JV` (Javanese), `EN` (English), `MIX_ID_EN`, `MIX_ID_JV`, `MIX_JV_EN` (mixed morphemes within a token), and `OTH` (other).

## Getting started

From this project directory, create a Python environment and install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab ipykernel torch transformers pandas tqdm huggingface-hub
jupyter lab
```

1. Place `train.tsv`, `val.tsv`, and `test.tsv` in `../dataset/`, or update `DATA_DIR` in the notebook. Each line must contain a token and label separated by a tab; blank lines separate tweets.
2. Open a notebook with the environment's Python kernel and run its cells in order from this directory. Model weights are downloaded from Hugging Face on first use. Llama requires approved model access and a read token, entered through the notebook's hidden prompt.
3. Review the configuration before running. CPU is the default; set `LIMIT` to a small positive number for a pilot or `None` for the full splits.

Results are saved under `outputs/`, with comparison tables in `outputs/comparisons/`. Evaluation reports accuracy, macro precision/recall/F1, weighted F1, per-label scores, and invalid-output rate. **Macro F1** is the primary comparison metric. Runs with the same configuration can resume from saved predictions.

## Tests

```bash
python -m unittest discover -s tests
```

These checks use a tiny randomly initialized Qwen model and do not require the dataset or downloaded model weights.
