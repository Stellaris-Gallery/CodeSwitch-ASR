"""XLS-R CTC fine-tuning entry point for CS-Dialogue."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2ForCTC,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from codeswitch_asr.data import (
    CSDialogueCTCDataset,
    CTCDataCollator,
    build_processor,
    load_official_eval_split,
    load_train_manifest,
)
from codeswitch_asr.evaluate import compute_mer_by_group, greedy_ctc_decode


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an experiment YAML file."""
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    required = {"run_name", "data_config", "base_model", "head", "train_sets", "eval_sets"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if config["head"].lower() != "ctc":
        raise ValueError("Only head: ctc is supported")
    if not config["train_sets"]:
        raise ValueError("train_sets must not be empty")
    return config


def set_seed(seed: int) -> None:
    """Make CPU/Python/GPU random operations reproducible where supported."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Short aliases kept for scripts and external callers.
load_config = load_yaml_config
configure_determinism = set_seed


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def save_checkpoint(
    output_dir: str | Path,
    name: str,
    model,
    processor=None,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    global_step: int = 0,
    best_mer: float = float("inf"),
) -> Path:
    """Save a loadable Hugging Face model plus compact training state."""
    destination = _safe_output_dir(Path(output_dir) / name)
    model.save_pretrained(destination)
    if processor is not None:
        processor.save_pretrained(destination)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_mer": best_mer,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
            },
        },
        destination / "training_state.pt",
    )
    return destination


def evaluate(
    model,
    dataloader,
    processor,
    device: torch.device,
    *,
    mixed_precision: str = "fp32",
) -> dict[str, Any]:
    """Evaluate CTC loss and corpus-level MER on a single official split."""
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    refs: list[str] = []
    hyps: list[str] = []
    groups: list[str] = []
    use_cuda_autocast = device.type == "cuda" and mixed_precision in ("fp16", "bf16")

    with torch.no_grad():
        for batch in dataloader:
            ids = batch.pop("_ids", None)
            texts = batch.pop("_texts", None)
            tags = batch.pop("_tags", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            if use_cuda_autocast:
                with torch.amp.autocast("cuda", dtype=dtype):
                    output = model(**batch)
            else:
                output = model(**batch)
            loss_sum += float(output.loss.detach().cpu()) * batch["input_values"].size(0)
            sample_count += batch["input_values"].size(0)

            logits = model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask"),
            ).logits
            hyps.extend(greedy_ctc_decode(logits, processor))
            if texts is not None:
                refs.extend(texts)
            if tags is not None:
                groups.extend(tags)

    if not refs:
        raise ValueError("Evaluation dataloader must retain text references")
    mer = compute_mer_by_group(refs, hyps, groups) if groups else compute_mer_by_group(
        refs, hyps, ["ALL"] * len(refs)
    )
    return {
        "loss": loss_sum / max(sample_count, 1),
        "mer": mer["ALL"]["mer"],
        "groups": mer,
    }


def _batch_metadata(dataset, indices: list[int]) -> dict[str, list[str]]:
    rows = [dataset.records[index] for index in indices]
    return {"texts": [row.text for row in rows], "tags": [row.tag for row in rows]}


def train(config: dict[str, Any], *, dataset_root: str | Path | None = None) -> dict[str, Any]:
    """Run one-GPU CTC training from a validated experiment config."""
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = get_device()

    data_config_path = Path(config["data_config"])
    if not data_config_path.is_absolute():
        data_config_path = data_config_path.resolve()
    with data_config_path.open(encoding="utf-8") as source:
        data_config = yaml.safe_load(source)
    public = data_config["public"]
    vocab_config = data_config["vocab"]

    if dataset_root is None:
        dataset_root = os.environ.get(public["root_env"])
    if dataset_root is None:
        raise ValueError(f"Supply --dataset-root or set {public['root_env']}")
    root = Path(dataset_root).resolve()
    vocab_path = root / vocab_config["path"]

    processor = build_processor(vocab_path, config["base_model"])
    train_records = load_train_manifest(root, check_files=True)
    dev_records = load_official_eval_split(root, "dev")
    train_dataset = CSDialogueCTCDataset(train_records, processor)
    dev_dataset = CSDialogueCTCDataset(dev_records, processor, allow_oov=True)
    collator = CTCDataCollator(processor)

    batch_size = int(config.get("batch_size", 8))
    workers = int(config.get("dataloader_num_workers", 0))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    def eval_collate(features):
        batch = collator(features)
        batch["_texts"] = [item["text"] for item in features]
        batch["_tags"] = [
            next(row.tag for row in dev_dataset.records if row.id == item["id"])
            for item in features
        ]
        return batch

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=eval_collate,
        pin_memory=device.type == "cuda",
    )

    resume_path = config.get("resume_from_checkpoint")
    if resume_path:
        model = Wav2Vec2ForCTC.from_pretrained(
            resume_path,
            ctc_zero_infinity=True,
        ).to(device)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(
            config["base_model"],
            vocab_size=len(processor.tokenizer),
            pad_token_id=processor.tokenizer.pad_token_id,
            ignore_mismatched_sizes=True,
            ctc_zero_infinity=True,
        ).to(device)

    if bool(config.get("freeze_feature_encoder", True)):
        model.freeze_feature_encoder()

    optimizer_config = config.get("optimizer", {})
    learning_rate = float(optimizer_config.get("lr", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epochs = int(config.get("epochs", 5))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    total_steps = max(1, (len(train_dataloader) + accumulation - 1) // accumulation * epochs)
    warmup_steps = int(config.get("warmup_steps", 0))
    warmup_ratio = float(config.get("warmup_ratio", 0.0))
    if warmup_steps == 0 and warmup_ratio:
        warmup_steps = int(total_steps * warmup_ratio)
    scheduler_name = str(config.get("scheduler", "linear")).lower()
    if scheduler_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    output_dir = _safe_output_dir(config.get("output_dir", f"checkpoints/{config['run_name']}"))
    mixed_precision = str(config.get("mixed_precision", "fp16")).lower()
    if device.type != "cuda":
        mixed_precision = "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16" and device.type == "cuda")

    wandb_config = config.get("wandb", {})
    wandb = None
    cli_disable = config.get("_disable_wandb", False)
    if wandb_config.get("enabled", True) and not cli_disable:
        try:
            import wandb as wandb_module
            wandb = wandb_module
            wandb.init(
                project=wandb_config.get("project", "codeswitch-asr"),
                name=config["run_name"],
                config=config,
                mode=wandb_config.get("mode"),
                dir=str(output_dir),
            )
        except Exception as exc:
            print(f"WARNING: W&B disabled: {exc}")
            wandb = None

    start_epoch = 0
    global_step = 0
    best_mer = float("inf")
    if resume_path:
        state_path = Path(resume_path) / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state["global_step"])
            best_mer = float(state["best_mer"])
            optimizer.load_state_dict(state["optimizer"])
            if state["scheduler"] is not None:
                scheduler.load_state_dict(state["scheduler"])

    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    unfreeze_epoch = config.get("feature_encoder_unfreeze_epoch")
    if unfreeze_epoch is not None:
        unfreeze_epoch = int(unfreeze_epoch)

    for epoch in range(start_epoch, epochs):
        if unfreeze_epoch is not None and epoch >= unfreeze_epoch:
            for parameter in model.wav2vec2.feature_encoder.parameters():
                parameter.requires_grad = True
            model.wav2vec2.feature_encoder.train()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(device) for key, value in batch.items()}
            dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            if device.type == "cuda" and mixed_precision in ("fp16", "bf16"):
                with torch.amp.autocast("cuda", dtype=dtype):
                    loss = model(**batch).loss / accumulation
            else:
                loss = model(**batch).loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}")

            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * accumulation

            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if wandb is not None:
                    wandb.log({
                        "train/loss": running_loss / ((step % accumulation) + 1),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                    })

        # Perform an optimizer step for a partial final accumulation block.
        if len(train_dataloader) % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        dev_result = evaluate(
            model, dev_dataloader, processor, device,
            mixed_precision=mixed_precision,
        )
        print(
            f"epoch={epoch + 1} dev_loss={dev_result['loss']:.6f} "
            f"dev_mer={dev_result['mer']:.6f}"
        )
        if wandb is not None:
            wandb.log({
                "epoch": epoch + 1,
                "dev/loss": dev_result["loss"],
                "dev/MER": dev_result["mer"],
                "dev/MER/CN": dev_result["groups"].get("<CN>", {}).get("mer"),
                "dev/MER/EN": dev_result["groups"].get("<EN>", {}).get("mer"),
                "dev/MER/MIX": dev_result["groups"].get("<MIX>", {}).get("mer"),
            })

        if dev_result["mer"] < best_mer:
            best_mer = dev_result["mer"]
            save_checkpoint(
                output_dir, "best", model, processor,
                optimizer, scheduler, epoch, global_step, best_mer,
            )
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as target:
                json.dump({"epoch": epoch + 1, **dev_result}, target, ensure_ascii=False, indent=2)
        save_checkpoint(
            output_dir, f"epoch-{epoch + 1}", model, processor,
            optimizer, scheduler, epoch, global_step, best_mer,
        )

    final_results: dict[str, Any] = {}
    if "public_test" in set(config.get("final_eval_sets", [])):
        test_records = load_official_eval_split(root, "test")
        test_dataset = CSDialogueCTCDataset(test_records, processor, allow_oov=True)

        def test_collate(features):
            batch = collator(features)
            batch["_texts"] = [item["text"] for item in features]
            batch["_tags"] = [
                next(row.tag for row in test_dataset.records if row.id == item["id"])
                for item in features
            ]
            return batch

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=test_collate,
            pin_memory=device.type == "cuda",
        )
        best_path = output_dir / "best"
        if best_path.is_dir():
            final_model = Wav2Vec2ForCTC.from_pretrained(
                best_path, ctc_zero_infinity=True
            ).to(device)
            test_result = evaluate(
                final_model, test_dataloader, processor, device,
                mixed_precision=mixed_precision,
            )
            final_results["public_test"] = test_result
            print(
                f"public_test_loss={test_result['loss']:.6f} "
                f"public_test_mer={test_result['mer']:.6f}"
            )
            with (output_dir / "public_test_metrics.json").open(
                "w", encoding="utf-8"
            ) as target:
                json.dump(test_result, target, ensure_ascii=False, indent=2)
            if wandb is not None:
                wandb.log({
                    "test/MER": test_result["mer"],
                    "test/MER/CN": test_result["groups"].get("<CN>", {}).get("mer"),
                    "test/MER/EN": test_result["groups"].get("<EN>", {}).get("mer"),
                    "test/MER/MIX": test_result["groups"].get("<MIX>", {}).get("mer"),
                })

    if wandb is not None:
        artifact = wandb.Artifact(
            f"{config['run_name']}-best",
            type="model",
        )
        artifact.add_dir(str(output_dir / "best"))
        wandb.log_artifact(artifact)
        wandb.finish()

    return {
        "output_dir": str(output_dir),
        "best_mer": best_mer,
        "global_step": global_step,
        "final_eval": final_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ft_cs_dialogue.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.resume_from_checkpoint:
        config["resume_from_checkpoint"] = str(args.resume_from_checkpoint)
    if args.disable_wandb:
        config["_disable_wandb"] = True
    result = train(config, dataset_root=args.dataset_root)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
