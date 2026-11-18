# CS-Dialogue Config Migration

This bundle updates the three legacy ASCEND fine-tuning YAML configs and `configs/data.yaml`.

## Verified preparation output (workstation, 2026-11-16)

- Official train: 26,239
- Dropped `**`: 671 (CN 26; EN 477; MIX 168)
- Dropped empty normalized transcript: 0
- Retained train: 25,568
- Vocabulary size: 2,951 including `|`, `[UNK]`, `<blank>`
- Dev: 6,186; test: 6,257

## Portable dataset path

The environment variable `CS_DIALOGUE_ROOT` identifies the dataset root. The YAML file paths under `public` and `vocab` are relative to that directory; they are **not** currently expanded by a working training loader. Keep operating-system-specific absolute paths out of tracked configs.

Windows PowerShell:

```powershell
$env:CS_DIALOGUE_ROOT = "D:\Deeplearning Final Project\Datasets\CS-Dialogue"
```

## CTC tokenizer invariant

When the data loader and tokenizer are implemented, instantiate `Wav2Vec2CTCTokenizer` using the generated vocabulary and explicitly set `pad_token="<blank>"`, `unk_token="[UNK]"`, and `word_delimiter_token="|"`. Set `model.config.pad_token_id = tokenizer.pad_token_id` and `model.config.vocab_size = len(tokenizer)`. The CTC blank must match `pad_token_id`.

`ctc_text` in the generated manifest is a sequence of normalized characters with `|` between MER tokens. `normalize.tokenize()` is the single source of normalization rules; vocabulary is built exclusively from kept official train utterances.

## Experiment evaluation split

- `eval_sets: [public_dev]` is used for model selection.
- `final_eval_sets` are held out for final benchmarking after model selection; `ours_test` is unavailable until new recordings are collected.
- `finetune.py` and `data.py` are currently placeholder modules, so these YAML files do **not** make training executable yet.
- Numeric train counts and vocabulary size are snapshots from the processed dataset; regenerate if normalization or filtering changes.

## Old to new filenames

- `ft_ascend.yaml` -> `ft_cs_dialogue.yaml`
- `ft_ascend_noise.yaml` -> `ft_cs_dialogue_noise.yaml`
- `ft_ascend_noise_ours.yaml` -> `ft_cs_dialogue_noise_ours.yaml`

Do not commit original WAVs, ZIPs, caches, or generated manifests to the public repository.
