"""Offline tests for CTC training configuration and entry-point validation."""

from pathlib import Path

import pytest

from codeswitch_asr.finetune import load_yaml_config


def test_training_config_loads_and_validates():
    config = load_yaml_config(
        Path(__file__).parents[1] / "configs" / "ft_cs_dialogue.yaml"
    )
    assert config["head"] == "ctc"
    assert config["base_model"] == "facebook/wav2vec2-xls-r-300m"
    assert config["train_sets"] == ["public_train"]
    assert config["eval_sets"] == ["public_dev"]
    assert config["wandb"]["project"] == "codeswitch-asr"
