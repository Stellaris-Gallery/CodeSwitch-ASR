"""Offline tests for CTC training configuration and entry-point validation."""

from pathlib import Path

import pytest
import torch

from codeswitch_asr.data import ASRRecord
from codeswitch_asr.evaluate import greedy_ctc_decode, greedy_ctc_decode_with_stats
from codeswitch_asr.finetune import (
    apply_duration_limit,
    build_parser,
    get_device,
    load_training_state,
    load_yaml_config,
    run_optimizer_step,
    save_checkpoint,
    should_replace_best_checkpoint,
    validate_train_sets,
)


def test_training_config_loads_and_validates():
    config = load_yaml_config(
        Path(__file__).parents[1] / "configs" / "ft_cs_dialogue.yaml"
    )
    assert config["head"] == "ctc"
    assert config["base_model"] == "facebook/wav2vec2-xls-r-300m"
    assert config["train_sets"] == ["public_train"]
    assert config["eval_sets"] == ["public_dev"]
    assert config["wandb"]["project"] == "codeswitch-asr"


def test_cli_train_limit_and_disable_wandb_flags():
    parser = build_parser()
    args = parser.parse_args([
        "train", "--config", "configs/ft_cs_dialogue.yaml",
        "--max-train-samples", "50", "--max-steps", "10",
        "--dry-run", "--disable-wandb", "--device", "cpu",
    ])
    assert args.max_train_samples == 50
    assert args.max_steps == 10
    assert args.dry_run
    assert args.disable_wandb
    assert args.device == "cpu"


def test_cli_accepts_configurable_dev_subset():
    args = build_parser().parse_args([
        "train", "--max-train-samples", "50",
        "--dev-subset-samples", "200",
    ])
    assert args.max_train_samples == 50
    assert args.dev_subset_samples == 200


@pytest.mark.parametrize(
    "train_sets",
    [
        ["public_train_noisy"],
        ["public_train", "public_train_noisy"],
        ["ours_train"],
        [],
    ],
)
def test_unsupported_training_sets_are_rejected(train_sets):
    with pytest.raises(ValueError, match="public_train"):
        validate_train_sets(train_sets)


def test_public_test_evaluation_is_a_separate_explicit_cli():
    args = build_parser().parse_args([
        "evaluate", "--checkpoint", "checkpoints/ft_cs_dialogue/best",
        "--split", "public_test",
    ])
    assert args.command == "evaluate"
    assert args.split == "public_test"


def test_duration_limit_only_filters_training_records(tmp_path):
    rows = [
        ASRRecord("short", tmp_path / "s.wav", "好", "<CN>", "1", "train", 2.0, "好"),
        ASRRecord("long", tmp_path / "l.wav", "你好", "<CN>", "1", "train", 20.0, "你好"),
    ]
    kept, excluded = apply_duration_limit(rows, 10.0)
    assert [row.id for row in kept] == ["short"]
    assert [row.id for row in excluded] == ["long"]


def test_device_validation():
    assert get_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="device"):
        get_device("gpu")


def test_best_checkpoint_selection_uses_lower_mer():
    assert should_replace_best_checkpoint(0.25, 0.30)
    assert not should_replace_best_checkpoint(0.35, 0.30)


class _Tokenizer:
    pad_token_id = 0
    word_delimiter_token = "|"

    def convert_tokens_to_ids(self, token):
        return {"|": 1, "[UNK]": 2}.get(token, -1)

    def convert_ids_to_tokens(self, token_id):
        return {1: "|", 2: "[UNK]", 3: "你", 4: "好"}.get(token_id, "<blank>")


class _Processor:
    tokenizer = _Tokenizer()


def test_variable_length_decoding_respects_output_lengths():
    predictions = torch.tensor([
        [3, 3, 4, 0, 0],
        [3, 1, 4, 4, 0],
    ])
    decoded = greedy_ctc_decode(
        predictions, _Processor(), output_lengths=torch.tensor([3, 4])
    )
    assert decoded == ["你好", "你 好"]


def test_unknown_tokens_are_reported_not_silently_discarded():
    predictions = torch.tensor([[2, 2, 0, 3]])
    decoded = greedy_ctc_decode_with_stats(predictions, _Processor())
    assert decoded["hypotheses"] == ["[UNK]你"]
    assert decoded["unknown_token_count"] == 1
    assert decoded["unknown_utterance_count"] == 1


def test_missing_training_state_is_safe(tmp_path):
    assert load_training_state(tmp_path) is None


class _DummyModel:
    def save_pretrained(self, destination):
        Path(destination).mkdir(parents=True, exist_ok=True)


def test_checkpoint_save_reload_training_state(tmp_path):
    destination = save_checkpoint(
        tmp_path,
        "best",
        _DummyModel(),
        epoch=2,
        global_step=17,
        best_mer=0.123,
    )
    state = load_training_state(destination)
    assert state is not None
    assert state["epoch"] == 2
    assert state["global_step"] == 17
    assert state["best_mer"] == pytest.approx(0.123)


class _NoOpScaler:
    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def test_synthetic_gradient_accumulation_and_checkpoint_reload(tmp_path):
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = _NoOpScaler()

    optimizer.zero_grad()
    initial = model.weight.detach().clone()
    (model(torch.tensor([[1.0, 0.0]])).sum() / 2).backward()
    assert model.weight.grad is not None
    assert torch.equal(initial, model.weight.detach())

    (model(torch.tensor([[0.0, 1.0]])).sum() / 2).backward()
    run_optimizer_step(model, optimizer, scaler, scheduler, max_grad_norm=1.0)
    assert not torch.equal(initial, model.weight.detach())
    assert scheduler.last_epoch == 1

    destination = save_checkpoint(
        tmp_path,
        "best",
        _DummyModel(),
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=1,
        best_mer=0.5,
    )
    state = load_training_state(destination)
    assert state["optimizer"] is not None
    assert state["scheduler"]["last_epoch"] == 1
