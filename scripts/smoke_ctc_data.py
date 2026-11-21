"""Validate the local CS-Dialogue CTC data pipeline before any long training run.

Examples (VS Code PowerShell):
    uv run python scripts/smoke_ctc_data.py --dataset-root "D:\\Deeplearning Final Project\\Datasets\\CS-Dialogue"
    uv run python scripts/smoke_ctc_data.py --dataset-root "D:\\Deeplearning Final Project\\Datasets\\CS-Dialogue" --gpu-smoke

The default mode reads two WAVs and tests tokenization/padding. --gpu-smoke
additionally downloads XLS-R 300M and runs ONE GPU forward/backward smoke test.
It does NOT start an epoch of training or write a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from codeswitch_asr.data import (
    CSDialogueCTCDataset, CTCDataCollator, build_processor,
    load_official_eval_split, load_train_manifest, load_vocab,
    vocab_coverage, write_eval_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("CS_DIALOGUE_ROOT"))
    parser.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--ctc-loss-reduction", choices=("mean", "sum"), default="mean")
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error("Supply --dataset-root or set CS_DIALOGUE_ROOT")
    root = args.dataset_root.resolve()
    vocab_path = root / "processed" / "vocab.json"

    train = load_train_manifest(root, check_files=False)
    dev = load_official_eval_split(root, "dev")
    dev_manifest = write_eval_manifest(dev, root)
    vocab = load_vocab(vocab_path)

    print("\n=== CTC DATA CHECK ===")
    print("Train rows:", len(train))
    print("Dev rows (unfiltered):", len(dev))
    print("Vocabulary size:", len(vocab))
    for label, rows in [("train", train), ("dev", dev)]:
        print(label + " coverage:", json.dumps(vocab_coverage(rows, vocab), ensure_ascii=False))
    assert vocab_coverage(train, vocab)["oov_characters"] == 0
    print("Dev manifest:", dev_manifest)

    print("\nLoading local CTC tokenizer and XLS-R feature extractor...")
    processor = build_processor(vocab_path, args.model)
    print("Tokenizer size:", len(processor.tokenizer))
    print("CTC blank/pad ID:", processor.tokenizer.pad_token_id)
    print("XLS-R sample rate:", processor.feature_extractor.sampling_rate)

    # Smoke-test a couple of shorter train files. This does not filter the
    # actual train set; the full manifest remains 25,568 utterances.
    candidates = sorted(
        (row for row in train if 1.5 <= row.duration <= 5.0 and row.ctc_text
        and len(row.ctc_text) < int(row.duration * 50 * 0.55)),
        key=lambda row: (row.duration, row.id),
    )
    if len(candidates) < 2:
        raise RuntimeError("Could not find two short train WAVs for the smoke test")
    data = CSDialogueCTCDataset(candidates[:2], processor)
    batch = CTCDataCollator(processor)([data[0], data[1]])
    print("Padded audio batch:", tuple(batch["input_values"].shape))
    print("Padded label batch:", tuple(batch["labels"].shape))
    print("Label IDs valid:", bool(((batch["labels"] == -100) | ((batch["labels"] >= 0) & (batch["labels"] < len(vocab)))).all()))
    print("Data/Tokenizer/Collator: PASS")

    if not args.gpu_smoke:
        print("GPU model load was skipped. Add --gpu-smoke for one forward/backward pass.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("--gpu-smoke requires CUDA")
    from transformers import Wav2Vec2ForCTC

    # Load the pretrained encoder and initialize a NEW task-specific CTC head.
    print("\nDownloading/loading XLS-R 300M and initializing the CTC head...")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        vocab_size=len(vocab),
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
        ctc_zero_infinity=True,
        ctc_loss_reduction=args.ctc_loss_reduction,
    ).to("cuda:0")
    model.train()
    # The experiment's eventual freeze/unfreeze policy remains to be decided.
    # For the smoke pass, freezing the feature encoder saves VRAM.
    model.freeze_feature_encoder()
    single = CTCDataCollator(processor)([data[0]])
    single = {key: value.to("cuda:0") for key, value in single.items()}
    output = model(**single)
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise RuntimeError(f"Non-finite CTC loss: {loss}")
    if float(loss.detach().cpu()) <= 0:
        raise RuntimeError("CTC loss is zero; possible unalignable label sequence")
    loss.backward()
    has_grad = model.lm_head.weight.grad is not None
    if not has_grad or not bool(torch.isfinite(model.lm_head.weight.grad).all()):
        raise RuntimeError("Missing/non-finite gradient on CTC head")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CTC loss:", float(loss.detach().cpu()))
    print("CTC head gradient: PASS")
    print("GPU smoke: PASS (one batch; no optimizer step; no checkpoint)")


if __name__ == "__main__":
    main()
