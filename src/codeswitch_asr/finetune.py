"""XLS-R CTC fine-tuning entry point for CS-Dialogue."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
from codeswitch_asr.evaluate import (
    compute_mer_by_group,
    greedy_ctc_decode_with_stats,
)


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
    validate_train_sets(config["train_sets"])
    return config


def validate_train_sets(train_sets: list[str]) -> None:
    """Reject experiment conditions that this PR does not implement."""
    unsupported = set(train_sets) - {"public_train"}
    if unsupported or train_sets != ["public_train"]:
        raise ValueError(
            "This pipeline supports only train_sets: [public_train]; "
            f"got {train_sets}. Public noise/private-data conditions are not implemented."
        )


def validate_ctc_loss_reduction(value: str) -> str:
    value = str(value).lower()
    if value not in ("mean", "sum"):
        raise ValueError("ctc_loss_reduction must be 'mean' or 'sum'")
    return value


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


def get_device(selection: str = "auto") -> torch.device:
    selection = selection.lower()
    if selection == "cpu":
        return torch.device("cpu")
    if selection == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return torch.device("cuda")
    if selection == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was explicitly requested but is unavailable")
        return torch.device("mps")
    if selection != "auto":
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
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
    best_loss: float = float("inf"),
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
            "best_loss": best_loss,
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


def load_training_state(path: str | Path) -> dict[str, Any] | None:
    state_path = Path(path) / "training_state.pt"
    if not state_path.is_file():
        return None
    return torch.load(state_path, map_location="cpu", weights_only=False)


def should_replace_best_checkpoint(
    candidate_mer: float,
    best_mer: float,
    candidate_loss: float = float("inf"),
    best_loss: float = float("inf"),
) -> bool:
    """Select by lower dev MER, then lower dev loss on exact MER ties."""
    if candidate_mer < best_mer:
        return True
    if candidate_mer > best_mer:
        return False
    return candidate_loss < best_loss


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
    unknown_token_count = 0
    unknown_utterance_count = 0
    use_cuda_autocast = device.type == "cuda" and mixed_precision in ("fp16", "bf16")

    with torch.no_grad():
        for batch in dataloader:
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

            output_lengths = None
            if batch.get("attention_mask") is not None:
                input_lengths = batch["attention_mask"].sum(dim=-1)
                output_lengths = model._get_feat_extract_output_lengths(input_lengths)
            decoded = greedy_ctc_decode_with_stats(
                output.logits,
                processor,
                output_lengths=output_lengths,
            )
            hyps.extend(decoded["hypotheses"])
            unknown_token_count += decoded["unknown_token_count"]
            unknown_utterance_count += decoded["unknown_utterance_count"]
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
        "unknown_token_count": unknown_token_count,
        "unknown_utterance_count": unknown_utterance_count,
    }


def _batch_metadata(dataset, indices: list[int]) -> dict[str, list[str]]:
    rows = [dataset.records[index] for index in indices]
    return {"texts": [row.text for row in rows], "tags": [row.tag for row in rows]}


def apply_duration_limit(
    records: list[Any],
    max_duration_seconds: float | None,
) -> tuple[list[Any], list[Any]]:
    """Filter only caller-provided training records; never filter evaluation records."""
    if max_duration_seconds is None:
        return records, []
    if max_duration_seconds <= 0:
        raise ValueError("max_train_duration_seconds must be positive")
    kept = [row for row in records if float(row.duration) <= max_duration_seconds]
    excluded = [row for row in records if float(row.duration) > max_duration_seconds]
    return kept, excluded


def run_optimizer_step(
    model,
    optimizer,
    scaler,
    scheduler,
    max_grad_norm: float,
) -> int:
    """Run one clipped optimizer/scheduler step and return the step count."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return 1


def ctc_diagnostics(
    model,
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    blank_id: int,
) -> dict[str, Any]:
    """Report CTC frame, target, and blank-collapse diagnostics."""
    predictions = logits.argmax(dim=-1)
    if batch.get("attention_mask") is not None:
        input_lengths = batch["attention_mask"].sum(dim=-1)
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)
    else:
        output_lengths = torch.full(
            (logits.size(0),),
            logits.size(1),
            dtype=torch.long,
            device=logits.device,
        )
    output_lengths_cpu = output_lengths.detach().cpu()
    blank_counts = torch.zeros(logits.size(0), dtype=torch.long)
    nonblank_counts = torch.zeros(logits.size(0), dtype=torch.long)
    blank_ratios: list[float] = []
    for row_index, frame_count in enumerate(output_lengths_cpu.tolist()):
        frame_count = int(frame_count)
        row_predictions = predictions[row_index, :frame_count]
        blanks = int((row_predictions == blank_id).sum().cpu())
        blank_counts[row_index] = blanks
        nonblank_counts[row_index] = frame_count - blanks
        blank_ratios.append(blanks / frame_count if frame_count else 0.0)

    target_lengths = (batch["labels"] != -100).sum(dim=-1).detach().cpu()
    return {
        "ctc_input_lengths": [int(value) for value in output_lengths_cpu.tolist()],
        "ctc_target_lengths": [int(value) for value in target_lengths.tolist()],
        "blank_prediction_ratio_mean": float(sum(blank_ratios) / max(len(blank_ratios), 1)),
        "blank_prediction_counts": [int(value) for value in blank_counts.tolist()],
        "nonblank_prediction_counts": [int(value) for value in nonblank_counts.tolist()],
    }


def gradient_diagnostics(model) -> dict[str, Any]:
    """Return finite/nonzero gradient diagnostics for CTC training."""
    total_squared = 0.0
    head_grad_norm = 0.0
    lm_head = getattr(model, "lm_head", None)
    head_grad: torch.Tensor | None = getattr(lm_head, "weight", None)
    if head_grad is None:
        head_grad = getattr(model, "weight", None)
    if head_grad is not None and head_grad.grad is not None:
        gradient = head_grad.grad.detach().float()
        head_grad_norm = float(torch.linalg.vector_norm(gradient).cpu())
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            return {
                "total_gradient_norm": float("inf"),
                "head_gradient_norm": head_grad_norm,
                "head_has_finite_gradient": False,
                "head_has_nonzero_gradient": False,
            }
        total_squared += float(torch.sum(gradient * gradient).cpu())
    finite_head = bool(torch.isfinite(torch.tensor(head_grad_norm)))
    return {
        "total_gradient_norm": float(total_squared ** 0.5),
        "head_gradient_norm": head_grad_norm,
        "head_has_finite_gradient": finite_head,
        "head_has_nonzero_gradient": head_grad_norm > 0.0,
    }


def ctc_head_parameters_changed(model, before: dict[str, torch.Tensor]) -> bool:
    for name, parameter in model.lm_head.named_parameters():
        if name in before and not torch.equal(parameter.detach().cpu(), before[name]):
            return True
    return False


def train(
    config: dict[str, Any],
    *,
    dataset_root: str | Path | None = None,
    max_train_samples: int | None = None,
    max_steps: int | None = None,
    dev_subset_samples: int | None = None,
    dry_run: bool = False,
    device_selection: str = "auto",
) -> dict[str, Any]:
    """Run one-GPU CTC training from a validated experiment config."""
    seed = int(config.get("seed", 0))
    set_seed(seed)
    ctc_loss_reduction = validate_ctc_loss_reduction(
        config.get("ctc_loss_reduction", "sum")
    )
    if config.get("require_fresh_checkpoint", False) and config.get("resume_from_checkpoint"):
        raise ValueError(
            "This experiment requires a fresh XLS-R-300M/pretrained CTC head; "
            "resume-from-checkpoint is not allowed."
        )
    device = get_device(device_selection)

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

    train_records = load_train_manifest(root, check_files=True)
    dev_records = load_official_eval_split(root, "dev")

    max_duration = config.get("max_train_duration_seconds")
    if max_duration is not None:
        max_duration = float(max_duration)
    train_records, excluded_records = apply_duration_limit(train_records, max_duration)
    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive")
        train_records = train_records[:max_train_samples]
    elif config.get("train_sample_limit") is not None:
        train_sample_limit = int(config["train_sample_limit"])
        if train_sample_limit <= 0:
            raise ValueError("train_sample_limit must be positive")
        train_records = train_records[:train_sample_limit]
    if not train_records:
        raise ValueError("No training samples remain after limits")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if dev_subset_samples is not None:
        if dev_subset_samples <= 0:
            raise ValueError("dev_subset_samples must be positive")
        dev_subset_samples = min(dev_subset_samples, len(dev_records))
    elif config.get("dev_subset_samples") is not None:
        dev_subset_samples = int(config["dev_subset_samples"])
        if dev_subset_samples <= 0:
            raise ValueError("dev_subset_samples must be positive")
        dev_subset_samples = min(dev_subset_samples, len(dev_records))
    output_dir = _safe_output_dir(config.get("output_dir", f"checkpoints/{config['run_name']}"))
    if excluded_records:
        exclusion_report = {
            "criterion": "duration",
            "max_train_duration_seconds": max_duration,
            "excluded_count": len(excluded_records),
            "excluded_ids": [row.id for row in excluded_records],
        }
        with (output_dir / "train_exclusions.json").open("w", encoding="utf-8") as target:
            json.dump(exclusion_report, target, ensure_ascii=False, indent=2)

    if dry_run:
        plan = {
            "dry_run": True,
            "device": str(device),
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "dev_subset_samples": dev_subset_samples,
            "excluded_train_duration": len(excluded_records),
            "max_train_samples": max_train_samples,
            "max_steps": max_steps,
            "base_model": config["base_model"],
            "mixed_precision": config.get("mixed_precision", "bf16"),
            "public_test": "not evaluated unless a separate evaluation command is used",
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    processor = build_processor(vocab_path, config["base_model"])
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

    train_subset_dataloader = None
    if max_train_samples is not None:
        train_subset_dataset = CSDialogueCTCDataset(train_records, processor)

        def train_subset_collate(features):
            batch = collator(features)
            batch["_texts"] = [item["text"] for item in features]
            batch["_tags"] = [
                next(row.tag for row in train_subset_dataset.records if row.id == item["id"])
                for item in features
            ]
            return batch

        train_subset_dataloader = DataLoader(
            train_subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=train_subset_collate,
            pin_memory=device.type == "cuda",
        )

    dev_subset_dataloader = None
    if dev_subset_samples is not None:
        dev_subset_dataset = CSDialogueCTCDataset(
            dev_records[:dev_subset_samples], processor, allow_oov=True
        )

        def dev_subset_collate(features):
            batch = collator(features)
            batch["_texts"] = [item["text"] for item in features]
            batch["_tags"] = [
                next(row.tag for row in dev_subset_dataset.records if row.id == item["id"])
                for item in features
            ]
            return batch

        dev_subset_dataloader = DataLoader(
            dev_subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=dev_subset_collate,
            pin_memory=device.type == "cuda",
        )

    resume_path = config.get("resume_from_checkpoint")
    if resume_path:
        model = Wav2Vec2ForCTC.from_pretrained(
            resume_path,
            ctc_zero_infinity=True,
            ctc_loss_reduction=ctc_loss_reduction,
        ).to(device)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(
            config["base_model"],
            vocab_size=len(processor.tokenizer),
            pad_token_id=processor.tokenizer.pad_token_id,
            ignore_mismatched_sizes=True,
            ctc_zero_infinity=True,
            ctc_loss_reduction=ctc_loss_reduction,
        ).to(device)

    if bool(config.get("freeze_feature_encoder", True)):
        model.freeze_feature_encoder()
    model.config.ctc_loss_reduction = ctc_loss_reduction

    optimizer_config = config.get("optimizer", {})
    learning_rate = float(optimizer_config.get("lr", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epochs = int(config.get("epochs", 5))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    steps_per_epoch = (len(train_dataloader) + accumulation - 1) // accumulation
    if max_steps is not None:
        scheduled_epochs = min(epochs, (max_steps + steps_per_epoch - 1) // steps_per_epoch)
        total_steps = min(max_steps, steps_per_epoch * epochs)
    else:
        scheduled_epochs = epochs
        total_steps = steps_per_epoch * epochs
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

    mixed_precision = str(config.get("mixed_precision", "fp16")).lower()
    if device.type != "cuda":
        mixed_precision = "fp32"
    elif mixed_precision == "bf16":
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) < (8, 0):
            mixed_precision = "fp16"
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
    best_loss = float("inf")
    if resume_path:
        state = load_training_state(resume_path)
        if state is not None:
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state["global_step"])
            best_mer = float(state["best_mer"])
            best_loss = float(state.get("best_loss", float("inf")))
            if state.get("optimizer") is not None:
                optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            rng = state.get("rng")
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.random.set_rng_state(rng["torch"])

    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    unfreeze_epoch = config.get("feature_encoder_unfreeze_epoch")
    if unfreeze_epoch is not None:
        unfreeze_epoch = int(unfreeze_epoch)

    overfit_history: list[dict[str, Any]] = []
    stop_training = False
    for epoch in range(start_epoch, scheduled_epochs):
        if stop_training:
            break
        if unfreeze_epoch is not None and epoch >= unfreeze_epoch:
            for parameter in model.wav2vec2.feature_encoder.parameters():
                parameter.requires_grad = True
            model.wav2vec2.feature_encoder.train()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_micro_batches = 0
        running_loss = 0.0
        running_micro_batches = 0
        optimizer_steps_this_epoch = 0
        diagnostics_last: dict[str, Any] | None = None
        head_changed_any = False
        for step, batch in enumerate(train_dataloader):
            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            if device.type == "cuda" and mixed_precision in ("fp16", "bf16"):
                with torch.amp.autocast("cuda", dtype=dtype):
                    output = model(**batch)
                    micro_loss = output.loss
                    loss = micro_loss / accumulation
            else:
                output = model(**batch)
                micro_loss = output.loss
                loss = micro_loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}")

            scaler.scale(loss).backward()
            diagnostics_last = ctc_diagnostics(
                model, output.logits, batch, processor.tokenizer.pad_token_id
            )
            diagnostics_last.update(gradient_diagnostics(model))
            diagnostics_last["ctc_loss_reduction"] = ctc_loss_reduction
            micro_loss_value = float(micro_loss.detach().cpu())
            running_loss += micro_loss_value
            epoch_loss_sum += micro_loss_value
            running_micro_batches += 1
            epoch_micro_batches += 1

            if (step + 1) % accumulation == 0:
                head_before = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.lm_head.named_parameters()
                }
                run_optimizer_step(
                    model, optimizer, scaler, scheduler, max_grad_norm
                )
                head_changed_any = (
                    head_changed_any or ctc_head_parameters_changed(model, head_before)
                )
                diagnostics_last["ctc_head_parameters_changed"] = head_changed_any
                global_step += 1
                optimizer_steps_this_epoch += 1
                if wandb is not None:
                    wandb.log({
                        "train/loss": running_loss / max(running_micro_batches, 1),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                    })
                running_loss = 0.0
                running_micro_batches = 0

        # Perform an optimizer step for a partial final accumulation block.
        if not stop_training and running_micro_batches > 0:
            head_before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.lm_head.named_parameters()
            }
            run_optimizer_step(
                model, optimizer, scaler, scheduler, max_grad_norm
            )
            head_changed_any = (
                head_changed_any or ctc_head_parameters_changed(model, head_before)
            )
            if diagnostics_last is not None:
                diagnostics_last["ctc_head_parameters_changed"] = head_changed_any
            global_step += 1
            optimizer_steps_this_epoch += 1
            if wandb is not None:
                wandb.log({
                    "train/loss": running_loss / max(running_micro_batches, 1),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                })

        dev_subset_result = None
        if dev_subset_dataloader is not None:
            dev_subset_result = evaluate(
                model, dev_subset_dataloader, processor, device,
                mixed_precision=mixed_precision,
            )
            dev_result = dev_subset_result
        else:
            dev_result = evaluate(
                model, dev_dataloader, processor, device,
                mixed_precision=mixed_precision,
            )
        train_subset_result = None
        if train_subset_dataloader is not None:
            train_subset_result = evaluate(
                model, train_subset_dataloader, processor, device,
                mixed_precision=mixed_precision,
            )
        train_epoch_loss = epoch_loss_sum / max(epoch_micro_batches, 1)
        print(
            f"epoch={epoch + 1} train_loss={train_epoch_loss:.6f} "
            f"dev_loss={dev_result['loss']:.6f} "
            f"dev_mer={dev_result['mer']:.6f}"
        )
        if diagnostics_last is not None:
            print(
                "ctc_diagnostics="
                + json.dumps(diagnostics_last, ensure_ascii=False)
            )
        history_row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_epoch_loss": train_epoch_loss,
            "dev_full_loss": dev_result["loss"],
            "dev_full_mer": dev_result["mer"],
        }
        if diagnostics_last is not None:
            history_row["ctc_diagnostics"] = diagnostics_last
        if train_subset_result is not None:
            history_row["train_subset_loss"] = train_subset_result["loss"]
            history_row["train_subset_mer"] = train_subset_result["mer"]
            history_row["train_subset_unknown_tokens"] = train_subset_result["unknown_token_count"]
            print(
                f"train_subset_loss={train_subset_result['loss']:.6f} "
                f"train_subset_mer={train_subset_result['mer']:.6f}"
            )
        if dev_subset_result is not None:
            history_row["dev_subset_loss"] = dev_subset_result["loss"]
            history_row["dev_subset_mer"] = dev_subset_result["mer"]
            history_row["dev_subset_unknown_tokens"] = dev_subset_result["unknown_token_count"]
            print(
                f"dev_subset_loss={dev_subset_result['loss']:.6f} "
                f"dev_subset_mer={dev_subset_result['mer']:.6f}"
            )
        overfit_history.append(history_row)
        with (output_dir / "validation_progress.json").open("w", encoding="utf-8") as target:
            json.dump(overfit_history, target, ensure_ascii=False, indent=2)
        if wandb is not None:
            log_row = {
                "epoch": epoch + 1,
                "train/epoch_loss": train_epoch_loss,
                "dev/loss": dev_result["loss"],
                "dev/MER": dev_result["mer"],
                "dev/MER/CN": dev_result["groups"].get("<CN>", {}).get("mer"),
                "dev/MER/EN": dev_result["groups"].get("<EN>", {}).get("mer"),
                "dev/MER/MIX": dev_result["groups"].get("<MIX>", {}).get("mer"),
            }
            if train_subset_result is not None:
                log_row["train_subset/loss"] = train_subset_result["loss"]
                log_row["train_subset/MER"] = train_subset_result["mer"]
            if dev_subset_result is not None:
                log_row["dev_subset/loss"] = dev_subset_result["loss"]
                log_row["dev_subset/MER"] = dev_subset_result["mer"]
            if diagnostics_last is not None:
                log_row["ctc/blank_prediction_ratio"] = diagnostics_last[
                    "blank_prediction_ratio_mean"
                ]
                log_row["ctc/head_gradient_norm"] = diagnostics_last[
                    "head_gradient_norm"
                ]
                log_row["ctc/total_gradient_norm"] = diagnostics_last[
                    "total_gradient_norm"
                ]
            wandb.log(log_row)

        if should_replace_best_checkpoint(
            dev_result["mer"], best_mer, dev_result["loss"], best_loss
        ):
            best_mer = dev_result["mer"]
            best_loss = dev_result["loss"]
            save_checkpoint(
                output_dir, "best", model, processor,
                optimizer, scheduler, epoch, global_step, best_mer, best_loss,
            )
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as target:
                json.dump({
                    "epoch": epoch + 1,
                    "selection_scope": "dev_subset" if dev_subset_result is not None else "dev_full",
                    **dev_result,
                }, target, ensure_ascii=False, indent=2)
        save_checkpoint(
            output_dir, f"epoch-{epoch + 1}", model, processor,
            optimizer, scheduler, epoch, global_step, best_mer, best_loss,
        )

    final_full_dev_result = None
    best_path = output_dir / "best"
    if best_path.is_dir():
        best_model = Wav2Vec2ForCTC.from_pretrained(
            best_path, ctc_zero_infinity=True
        ).to(device)
        final_full_dev_result = evaluate(
            best_model, dev_dataloader, processor, device,
            mixed_precision=mixed_precision,
        )
        with (output_dir / "final_full_dev_metrics.json").open(
            "w", encoding="utf-8"
        ) as target:
            json.dump(final_full_dev_result, target, ensure_ascii=False, indent=2)
        print(
            f"final_full_dev_loss={final_full_dev_result['loss']:.6f} "
            f"final_full_dev_mer={final_full_dev_result['mer']:.6f}"
        )

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
        "final_full_dev": final_full_dev_result,
    }


def evaluate_public_test(
    checkpoint: str | Path,
    *,
    dataset_root: str | Path | None = None,
    config_path: str | Path = "configs/ft_cs_dialogue.yaml",
    batch_size: int = 8,
    device_selection: str = "auto",
    mixed_precision: str = "bf16",
) -> dict[str, Any]:
    """Explicitly evaluate the held-out official public test split."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory not found: {checkpoint}")
    experiment_config = load_yaml_config(config_path)
    base_model = experiment_config["base_model"]
    data_config = experiment_config
    with Path(data_config["data_config"]).open(encoding="utf-8") as source:
        public_config = yaml.safe_load(source)
    public = public_config["public"]
    vocab_config = public_config["vocab"]
    if dataset_root is None:
        dataset_root = os.environ.get(public["root_env"])
    if dataset_root is None:
        raise ValueError(f"Supply --dataset-root or set {public['root_env']}")

    root = Path(dataset_root).resolve()
    processor = build_processor(root / vocab_config["path"], base_model)
    test_records = load_official_eval_split(root, "test")
    test_dataset = CSDialogueCTCDataset(test_records, processor, allow_oov=True)
    collator = CTCDataCollator(processor)

    def test_collate(features):
        batch = collator(features)
        batch["_texts"] = [item["text"] for item in features]
        batch["_tags"] = [
            next(row.tag for row in test_dataset.records if row.id == item["id"])
            for item in features
        ]
        return batch

    device = get_device(device_selection)
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_collate,
        pin_memory=device.type == "cuda",
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        checkpoint, ctc_zero_infinity=True
    ).to(device)
    result = evaluate(model, dataloader, processor, device, mixed_precision=mixed_precision)
    print(
        f"public_test_loss={result['loss']:.6f} "
        f"public_test_mer={result['mer']:.6f}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Fine-tune XLS-R-300M")
    train_parser.add_argument("--config", default="configs/ft_cs_dialogue.yaml")
    train_parser.add_argument("--dataset-root", type=Path)
    train_parser.add_argument("--output-dir", type=Path)
    train_parser.add_argument("--resume-from-checkpoint", type=Path)
    train_parser.add_argument("--disable-wandb", action="store_true")
    train_parser.add_argument("--max-train-samples", type=int)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--dev-subset-samples", type=int)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a checkpoint on an explicitly selected official split"
    )
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--config", default="configs/ft_cs_dialogue.yaml")
    evaluate_parser.add_argument("--dataset-root", type=Path)
    evaluate_parser.add_argument(
        "--split", choices=("public_test",), required=True
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    evaluate_parser.add_argument("--mixed-precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "evaluate"):
        argv = ["train", *argv]

    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_public_test(
            args.checkpoint,
            dataset_root=args.dataset_root,
            config_path=args.config,
            batch_size=args.batch_size,
            device_selection=args.device,
            mixed_precision=args.mixed_precision,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    config = load_yaml_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.resume_from_checkpoint:
        config["resume_from_checkpoint"] = str(args.resume_from_checkpoint)
    if args.disable_wandb:
        config["_disable_wandb"] = True
    result = train(
        config,
        dataset_root=args.dataset_root,
        max_train_samples=args.max_train_samples,
        max_steps=args.max_steps,
        dev_subset_samples=args.dev_subset_samples,
        dry_run=args.dry_run,
        device_selection=args.device,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
