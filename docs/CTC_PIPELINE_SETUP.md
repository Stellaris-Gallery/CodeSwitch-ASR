# CodeSwitch-ASR — CTC Data Loader and GPU Smoke Test

This pack implements `src/codeswitch_asr/data.py`, adds `scripts/smoke_ctc_data.py`, and adds offline tests. It deliberately leaves `finetune.py` untouched; a smoke test is not a training experiment.

## Preconditions

- Python 3.12 and the main repository's `uv sync` environment
- `D:\Deeplearning Final Project\Datasets\CS-Dialogue` populated with the official `data/index` and extracted `audio/short_wav`
- `processed/train_ctc_manifest.csv` and `processed/vocab.json` generated earlier
- 25,568 retained train utterances / 2,951 vocabulary tokens

## Commands (PowerShell from the main repository root)

```powershell
$env:CS_DIALOGUE_ROOT = "D:\Deeplearning Final Project\Datasets\CS-Dialogue"
uv run pytest
uv run python scripts/smoke_ctc_data.py --dataset-root "$env:CS_DIALOGUE_ROOT"
```

This first pass constructs a `dev_ctc_manifest.csv` in the external dataset directory, checks out-of-vocabulary labels without changing the train-only vocabulary, and tests two WAV files, tokenizer, and dynamic padding.

If it passes, optionally perform a real XLS-R CUDA forward/backward sanity check (downloads a sizable model):

```powershell
uv run python scripts/smoke_ctc_data.py --dataset-root "$env:CS_DIALOGUE_ROOT" --gpu-smoke
```

## Expected invariants

- Train: 25,568 retained rows, zero train OOV characters
- Dev: 6,186 official rows, **not** filtered for `**` or duration
- Vocabulary length: exactly 2,951, including `|`, `[UNK]`, `<blank>`
- No implicit `<s>`/`</s>` additions (the tokenizer explicitly sets `bos_token=None`, `eos_token=None`)
- `pad_token_id == vocab["<blank>"]`; CTC labels never contain the blank token
- Variable-length audio padded dynamically; padded label positions set to `-100`
- For dev/test, unseen characters are counted and mapped to `[UNK]` by the tokenizer (do not rebuild vocabulary using dev/test)

## Next milestone

`finetune.py` now provides the 1-GPU CTC training entry point. It loads the
train-only vocabulary, fine-tunes the XLS-R CTC head, evaluates the official
dev split for model selection, logs metrics/config/checkpoints to W&B, and runs
the official public test split only after training. Private `ours_test`
checkpoints remain outside this repository.

Run a 1-GPU experiment with:

```powershell
$env:CS_DIALOGUE_ROOT = "D:\Deeplearning Final Project\Datasets\CS-Dialogue"
uv run python -m codeswitch_asr.finetune --config configs/ft_cs_dialogue.yaml
```

Use `--disable-wandb` for local-only runs or set `wandb.enabled: false` in the
experiment YAML. Resume a previous run with
`--resume-from-checkpoint checkpoints/ft_cs_dialogue/best`.

Do not commit any `.wav`, processed manifest, model weight, or HF cache to the public repository.
